namespace ParakeetRuntime;

/// <summary>
/// How <see cref="SpeechEngine.LoadAsync"/> chooses where the model runs.
/// </summary>
/// <param name="UseGpu">Run on a Direct3D 12 GPU when one is usable; otherwise, or when false, run on the CPU.</param>
/// <param name="GpuNameContains">Consider only GPUs whose name contains this text, ignoring case, for example "RTX" or "Intel"; null considers every GPU.</param>
public sealed record SpeechEngineOptions(bool UseGpu = true, string? GpuNameContains = null);
