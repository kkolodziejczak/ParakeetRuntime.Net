namespace ParakeetRuntime;

/// <summary>
/// Runs NVIDIA Parakeet TDT 0.6B v3 inside the calling process. It transcribes one utterance at a
/// time, and each transcription runs on the thread pool, so callers never block.
/// </summary>
public sealed class SpeechEngine
{
    private ParakeetModel? modelUntilDisposed;
    private readonly SemaphoreSlim oneAtATime = new(1, 1);

    internal SpeechEngine(ParakeetModel model) => this.modelUntilDisposed = model;

    /// <summary>
    /// Loads an installed model into memory. This reads the whole model file and takes a few seconds.
    /// </summary>
    /// <param name="speechModel">The installed model to load.</param>
    /// <param name="cancellationToken">Cancels the load before it starts.</param>
    /// <returns>An engine ready to transcribe.</returns>
    /// <exception cref="FileNotFoundException">The model is not installed; see <see cref="SpeechModel.IsInstalled"/>.</exception>
    /// <exception cref="InvalidDataException">The model file is damaged or is not a Parakeet GGUF file.</exception>
    public static async Task<SpeechEngine> LoadAsync(SpeechModel speechModel, CancellationToken cancellationToken = default)
    {
        if (!speechModel.IsInstalled)
            throw new FileNotFoundException($"The speech model is not installed ({speechModel.Path}).", speechModel.Path);
        var model = await Task.Run(() => new ParakeetModel(speechModel.Path), cancellationToken);
        return new SpeechEngine(model);
    }

    internal static async Task<SpeechEngine> LoadUncheckedAsync(string modelPath) =>
        new(await Task.Run(() => new ParakeetModel(modelPath)));

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
        await oneAtATime.WaitAsync(cancellationToken);
        try
        {
            var running = modelUntilDisposed;
            ObjectDisposedException.ThrowIf(running is null, this);
            return await RunToCompletionWhileHoldingTheLock(running, samples16KhzMono);
        }
        finally
        {
            oneAtATime.Release();
        }
    }

    private static Task<string> RunToCompletionWhileHoldingTheLock(ParakeetModel model, float[] samples16KhzMono) =>
        Task.Run(() => model.Transcribe(samples16KhzMono));

    internal void Dispose()
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
