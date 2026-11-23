using System.Runtime.CompilerServices;
using ParakeetRuntime;

namespace ParakeetRuntime.Tests;

// Needs the real model (714 MB), so it runs only where LocalBuddy installed it.
internal sealed class InstalledModelFactAttribute : FactAttribute
{
    public InstalledModelFactAttribute()
    {
        if (!SpeechModel.Default().IsInstalled)
            Skip = "The speech model is not installed on this machine.";
    }
}

public sealed class SpeechEngineTests
{
    // A finished meeting still holds its LiveTranscription, and so its engine, until the app lets go of it.
    // The model's 860 MB of CPU weights must not wait for that.
    [InstalledModelFact]
    public void A_disposed_engine_no_longer_holds_the_model_even_while_the_engine_is_still_referenced()
    {
        var (engine, model) = LoadOnCpu();

        engine.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        Assert.False(model.IsAlive);
        GC.KeepAlive(engine);
    }

    [InstalledModelFact]
    public async Task A_disposed_engine_refuses_to_transcribe()
    {
        var (engine, _) = LoadOnCpu();

        engine.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(() => engine.TranscribeAsync(new float[16_000]));
    }

    // Not inlined, so no local in the test method keeps the model alive.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static (SpeechEngine Engine, WeakReference Model) LoadOnCpu()
    {
        var model = new ParakeetModel(SpeechModel.Default().Path, allowGpu: false);
        return (new SpeechEngine(model), new WeakReference(model));
    }
}
