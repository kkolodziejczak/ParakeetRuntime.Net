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

internal sealed class InstalledModelAndGpuFactAttribute : FactAttribute
{
    public InstalledModelAndGpuFactAttribute()
    {
        if (!SpeechModel.Default().IsInstalled)
            Skip = "The speech model is not installed on this machine.";
        else if (!GpuTestSupport.HasHardwareGpu)
            Skip = "No eligible hardware D3D12 GPU available on this machine.";
    }
}

public sealed class SpeechEngineTests
{
    [InstalledModelFact]
    public async Task An_engine_told_not_to_use_the_GPU_runs_on_the_CPU_and_says_so()
    {
        using var engine = await SpeechEngine.LoadAsync(SpeechModel.Default(), new SpeechEngineOptions(UseGpu: false));

        Assert.False(engine.IsUsingGpu);
        Assert.StartsWith("CPU", engine.Backend);
    }

    [InstalledModelAndGpuFact]
    public async Task An_engine_loaded_with_defaults_uses_a_GPU_and_names_it()
    {
        using var engine = await SpeechEngine.LoadAsync(SpeechModel.Default());

        Assert.True(engine.IsUsingGpu);
        Assert.StartsWith("GPU: ", engine.Backend);
    }

    [InstalledModelFact]
    public async Task An_engine_asked_for_a_GPU_no_adapter_matches_falls_back_to_the_CPU()
    {
        using var engine = await SpeechEngine.LoadAsync(SpeechModel.Default(), new SpeechEngineOptions(GpuNameContains: "no adapter is called this"));

        Assert.False(engine.IsUsingGpu);
        Assert.StartsWith("CPU", engine.Backend);
    }

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
