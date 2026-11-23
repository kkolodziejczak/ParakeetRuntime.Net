using ParakeetRuntime;

namespace ParakeetRuntime.Tests;

public sealed class SpeechEngineThreadingTests
{
    [Fact]
    public async Task Dispose_on_a_busy_UI_thread_does_not_wait_for_that_thread_to_run_the_transcription_continuation()
    {
        var model = new ModelThatFinishesWhenReleased();
        var engine = new SpeechEngine(model);
        StartTranscribingFrom(new UiThreadThatNeverPumps(), engine);

        model.Release();
        await model.Finished.Task.WaitAsync(TimeSpan.FromSeconds(5));
        var disposal = Task.Run(engine.Dispose);

        await disposal.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(model.Disposed);
    }

    private static void StartTranscribingFrom(SynchronizationContext uiThread, SpeechEngine engine)
    {
        var previous = SynchronizationContext.Current;
        SynchronizationContext.SetSynchronizationContext(uiThread);
        try
        {
            _ = engine.TranscribeAsync([0f]);
        }
        finally
        {
            SynchronizationContext.SetSynchronizationContext(previous);
        }
    }

    private sealed class ModelThatFinishesWhenReleased : IUtteranceRecognizer
    {
        private readonly ManualResetEventSlim released = new();
        public TaskCompletionSource Finished { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool Disposed { get; private set; }
        public string BackendName => "test";
        public bool IsUsingGpu => false;

        public void Release() => released.Set();

        public string Transcribe(float[] samples16KhzMono)
        {
            released.Wait();
            Finished.SetResult();
            return "";
        }

        public void Dispose() => Disposed = true;
    }

    private sealed class UiThreadThatNeverPumps : SynchronizationContext
    {
        public override void Post(SendOrPostCallback d, object? state) { }
    }
}
