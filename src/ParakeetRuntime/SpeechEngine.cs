namespace ParakeetRuntime;

/// <summary>
/// Runs NVIDIA Parakeet TDT 0.6B v3 inside the calling process. It transcribes one utterance at a
/// time, and each transcription runs on the thread pool, so callers never block.
/// </summary>
public sealed class SpeechEngine : IDisposable
{
    private IUtteranceRecognizer? modelUntilDisposed;
    private readonly SemaphoreSlim oneAtATime = new(1, 1);

    internal SpeechEngine(IUtteranceRecognizer model)
    {
        modelUntilDisposed = model;
        Backend = model.BackendName;
        IsUsingGpu = model.IsUsingGpu;
    }

    /// <summary>
    /// Loads an installed model into memory. This reads the whole model file and takes a few seconds.
    /// </summary>
    /// <param name="speechModel">The installed model to load.</param>
    /// <param name="options">Where the model runs; on a GPU when one is usable, when omitted.</param>
    /// <param name="cancellationToken">Cancels the load before it starts.</param>
    /// <returns>An engine ready to transcribe.</returns>
    /// <exception cref="FileNotFoundException">The model is not installed; see <see cref="SpeechModel.IsInstalled"/>.</exception>
    /// <exception cref="InvalidDataException">The model file is damaged or is not a Parakeet GGUF file.</exception>
    public static async Task<SpeechEngine> LoadAsync(SpeechModel speechModel, SpeechEngineOptions? options = null, CancellationToken cancellationToken = default)
    {
        if (!speechModel.IsInstalled)
            throw new FileNotFoundException($"The speech model is not installed ({speechModel.Path}).", speechModel.Path);
        var chosen = options ?? new SpeechEngineOptions();
        var model = await Task.Run(() => new ParakeetModel(speechModel.Path, chosen.UseGpu, chosen.GpuNameContains), cancellationToken).ConfigureAwait(false);
        return new SpeechEngine(model);
    }

    /// <summary>
    /// Where the model runs, for example "GPU: NVIDIA GeForce RTX 5070 Ti (D3D12)" or "CPU (AvxVnniInt8)".
    /// It changes to the CPU after a transcription in which the GPU was lost, for example to a driver reset.
    /// </summary>
    public string Backend { get; private set; }

    /// <summary>
    /// Whether the model runs on a GPU. False when the options turned the GPU off, and also when no
    /// usable GPU, or none whose name matched, was found.
    /// </summary>
    public bool IsUsingGpu { get; private set; }

    /// <summary>
    /// Transcribes audio of any length. Audio of 20 seconds or less is transcribed in one piece. Longer audio
    /// is cut at its quietest points into pieces of 20 seconds or less, because the model drops whole sentences
    /// from long audio, and the texts of the pieces are joined with a space. Calls made while another
    /// transcription runs wait for all of its pieces to finish.
    /// </summary>
    /// <param name="samples16KhzMono">The audio: 16 kHz mono samples in the range -1 to 1.</param>
    /// <param name="cancellationToken">Cancels the wait for an earlier transcription, and stops between pieces. A piece that has started always finishes.</param>
    /// <returns>The text, with punctuation and capitals as the model writes them; empty for empty audio.</returns>
    /// <exception cref="ObjectDisposedException">The engine has been disposed.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled.</exception>
    public async Task<string> TranscribeAsync(float[] samples16KhzMono, CancellationToken cancellationToken = default)
    {
        await oneAtATime.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var running = modelUntilDisposed;
            ObjectDisposedException.ThrowIf(running is null, this);
            try
            {
                return await RunToCompletionWhileHoldingTheLock(running, samples16KhzMono, cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                Backend = running.BackendName;
                IsUsingGpu = running.IsUsingGpu;
            }
        }
        finally
        {
            oneAtATime.Release();
        }
    }

    private static Task<string> RunToCompletionWhileHoldingTheLock(IUtteranceRecognizer model, float[] samples16KhzMono, CancellationToken stopBetweenPieces) =>
        Task.Run(() => QuietestPointCutter.TranscribeInPieces(samples16KhzMono, model.Transcribe, stopBetweenPieces));

    /// <summary>
    /// Transcribes a WAV file of up to 10 minutes. Its audio is converted to 16 kHz mono first, and is then
    /// transcribed as <see cref="TranscribeAsync"/> does.
    /// </summary>
    /// <param name="path">The WAV file: PCM at 8, 16, 24 or 32 bits or float at 32 or 64 bits, 1 to 8 channels, 8,000 to 192,000 Hz.</param>
    /// <param name="cancellationToken">Cancels the wait for an earlier transcription, and stops between pieces of long audio.</param>
    /// <returns>The text, with punctuation and capitals as the model writes them.</returns>
    /// <exception cref="FileNotFoundException">There is no file at the path.</exception>
    /// <exception cref="DirectoryNotFoundException">The folder in the path does not exist.</exception>
    /// <exception cref="InvalidDataException">The file is not a supported WAV file, is damaged, or holds more than 10 minutes of audio. The message says which.</exception>
    /// <exception cref="ObjectDisposedException">The engine has been disposed.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled.</exception>
    public async Task<string> TranscribeWavAsync(string path, CancellationToken cancellationToken = default)
    {
        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.SequentialScan);
        return await TranscribeWavAsync(file, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Transcribes a WAV stream of up to 10 minutes, read from its current position to the end of the audio.
    /// The stream need not support seeking, and is left open.
    /// </summary>
    /// <param name="wav">The WAV audio, in the formats <see cref="TranscribeWavAsync(string, CancellationToken)"/> accepts.</param>
    /// <param name="cancellationToken">Cancels the wait for an earlier transcription, and stops between pieces of long audio.</param>
    /// <returns>The text, with punctuation and capitals as the model writes them.</returns>
    /// <exception cref="InvalidDataException">The stream is not a supported WAV file, is damaged, or holds more than 10 minutes of audio. The message says which.</exception>
    /// <exception cref="ObjectDisposedException">The engine has been disposed.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled.</exception>
    public async Task<string> TranscribeWavAsync(Stream wav, CancellationToken cancellationToken = default)
    {
        var samples = await Task.Run(() => WavAudio.Read16KhzMono(wav, WavAudio.LongestAudio), cancellationToken).ConfigureAwait(false);
        return await TranscribeAsync(samples, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Releases the model and its GPU resources, and returns the model's managed memory to the system
    /// right away. Waits for a running transcription to finish first. Calling it again does nothing.
    /// </summary>
    public void Dispose()
    {
        oneAtATime.Wait();
        try
        {
            if (modelUntilDisposed is null)
                return;
            modelUntilDisposed.Dispose();
            modelUntilDisposed = null;
        }
        finally
        {
            oneAtATime.Release();
        }
        ReturnTheModelsManagedMemoryNow();
    }

    private static void ReturnTheModelsManagedMemoryNow() =>
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);
}
