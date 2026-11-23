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

    [Fact]
    public async Task After_the_GPU_is_lost_mid_transcription_the_engine_reports_the_CPU()
    {
        var model = new ModelThatLosesItsGpu();
        using var engine = new SpeechEngine(model);
        Assert.True(engine.IsUsingGpu);

        await engine.TranscribeAsync([0f]);

        Assert.False(engine.IsUsingGpu);
        Assert.Equal("CPU", engine.Backend);
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

    private sealed class ModelThatLosesItsGpu : IUtteranceRecognizer
    {
        public string BackendName => IsUsingGpu ? "GPU: test" : "CPU";
        public bool IsUsingGpu { get; private set; } = true;

        public string Transcribe(float[] samples16KhzMono)
        {
            IsUsingGpu = false;
            return "";
        }

        public void Dispose() { }
    }

    private sealed class UiThreadThatNeverPumps : SynchronizationContext
    {
        public override void Post(SendOrPostCallback d, object? state) { }
    }
}
