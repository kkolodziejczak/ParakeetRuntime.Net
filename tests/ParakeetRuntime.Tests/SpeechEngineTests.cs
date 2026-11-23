using System.Runtime.CompilerServices;
using ParakeetRuntime;

namespace ParakeetRuntime.Tests;

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
    [InstalledModelFact]
    public void A_disposed_engine_no_longer_holds_the_model_even_while_the_engine_is_still_referenced()
    {
        var (engine, model) = LoadOnCpuInItsOwnFrameSoNoTestLocalKeepsTheModel();

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
        var (engine, _) = LoadOnCpuInItsOwnFrameSoNoTestLocalKeepsTheModel();

        engine.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(() => engine.TranscribeAsync(new float[16_000]));
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static (SpeechEngine Engine, WeakReference Model) LoadOnCpuInItsOwnFrameSoNoTestLocalKeepsTheModel()
    {
        var model = new ParakeetModel(SpeechModel.Default().Path, allowGpu: false);
        return (new SpeechEngine(model), new WeakReference(model));
    }
}
