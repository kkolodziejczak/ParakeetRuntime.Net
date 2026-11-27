namespace ParakeetRuntime.Net;

/// <summary>
/// How <see cref="SpeechEngine.LoadAsync"/> chooses where the model runs.
/// </summary>
/// <param name="UseGpu">Run on a Direct3D 12 GPU when one is usable; otherwise, or when false, run on the CPU.</param>
/// <param name="GpuNameContains">Consider only GPUs whose name contains this text, ignoring case, for example "RTX" or "Intel"; null considers every GPU except integrated ones, which share memory with the CPU and are slower than the CPU path on the machines measured. Name an integrated GPU here to use it.</param>
public sealed record SpeechEngineOptions(bool UseGpu = true, string? GpuNameContains = null);
