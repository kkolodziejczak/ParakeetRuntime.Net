namespace ParakeetRuntime;

// Parakeet TDT 0.6B v3 running inside LocalBuddy: no separate engine, no port. One utterance at a
// time; each runs on the thread pool so callers never block.
public sealed class SpeechEngine
{
    // Null once disposed, so a finished meeting that still holds its engine does not hold the model.
    private ParakeetModel? model;
    private readonly SemaphoreSlim oneAtATime = new(1, 1);

    internal SpeechEngine(ParakeetModel model) => this.model = model;

    // Reads the 714 MB model into memory, which takes a few seconds.
    public static async Task<SpeechEngine> LoadAsync(SpeechModel speechModel, CancellationToken cancellationToken = default)
    {
        if (!speechModel.IsInstalled)
            throw new FileNotFoundException($"The speech model is not installed ({speechModel.Path}).", speechModel.Path);
        var model = await Task.Run(() => new ParakeetModel(speechModel.Path), cancellationToken);
        return new SpeechEngine(model);
    }

    // For the Spike: any Parakeet GGUF, for comparing conversions. Nothing checks this file.
    internal static async Task<SpeechEngine> LoadUncheckedAsync(string modelPath) =>
        new(await Task.Run(() => new ParakeetModel(modelPath)));

    public async Task<string> TranscribeAsync(float[] samples16KhzMono, CancellationToken cancellationToken = default)
    {
        await oneAtATime.WaitAsync(cancellationToken);
        try
        {
            var running = model;
            ObjectDisposedException.ThrowIf(running is null, this);
            // No token here: a cancelled wait would release the lock while the model still runs.
            return await Task.Run(() => running.Transcribe(samples16KhzMono));
        }
        finally
        {
            oneAtATime.Release();
        }
    }

    internal void Dispose()
    {
        oneAtATime.Wait();
        try
        {
            if (model is null)
                return;
            model.Dispose();
            model = null;
        }
        finally
        {
            oneAtATime.Release();
        }
        // The CPU weights are about 860 MB of managed arrays. Without this they stay in the process
        // until some later full collection, which an idle app may not run for a long time.
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);
    }
}
