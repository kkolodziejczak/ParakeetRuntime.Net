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
    /// </summary>
    public string Backend { get; }

    /// <summary>
    /// Whether the model runs on a GPU. False when the options turned the GPU off, and also when no
    /// usable GPU, or none whose name matched, was found.
    /// </summary>
    public bool IsUsingGpu { get; }

    /// <summary>
    /// Transcribes one utterance. Calls made while another transcription runs wait for it to finish.
    /// </summary>
    /// <param name="samples16KhzMono">The audio: 16 kHz mono samples in the range -1 to 1.</param>
    /// <param name="cancellationToken">Cancels the wait for an earlier transcription. A transcription that has started always finishes.</param>
    /// <returns>The text, with punctuation and capitals as the model writes them; empty for empty audio.</returns>
    /// <exception cref="ArgumentException">The audio is longer than the model can transcribe in one piece.</exception>
    /// <exception cref="ObjectDisposedException">The engine has been disposed.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled while waiting.</exception>
    public async Task<string> TranscribeAsync(float[] samples16KhzMono, CancellationToken cancellationToken = default)
    {
        await oneAtATime.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var running = modelUntilDisposed;
            ObjectDisposedException.ThrowIf(running is null, this);
            return await RunToCompletionWhileHoldingTheLock(running, samples16KhzMono).ConfigureAwait(false);
        }
        finally
        {
            oneAtATime.Release();
        }
    }

    private static Task<string> RunToCompletionWhileHoldingTheLock(IUtteranceRecognizer model, float[] samples16KhzMono) =>
        Task.Run(() => model.Transcribe(samples16KhzMono));

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
