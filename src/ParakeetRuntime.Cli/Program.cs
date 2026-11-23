using System.Reflection;

var version = typeof(Program).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";
if (args is ["--version"])
{
    Console.WriteLine(version);
    return 0;
}
if (args.Contains("--corpus") || args.Contains("--list-gpus"))
    return await CorpusBench.RunAsync(args);
if (args.Contains("--gemm-bench"))
    return GemmBench.Run(args);
if (args.Contains("--cpu-bench"))
    return CpuBench.Run(args);
if (args.Contains("--session-memory"))
    return await SessionMemory.RunAsync(args);

Console.Error.WriteLine($"parakeet-runtime {version}");
Console.Error.WriteLine("""
    Usage:
      parakeet-runtime --version
      parakeet-runtime --list-gpus
      parakeet-runtime --corpus <list.txt | wav folder> [options]   speed, word error rate and memory on a list of WAVs
      parakeet-runtime --cpu-bench [options]                       the CPU encoder's kernels per layer
      parakeet-runtime --gemm-bench [options]                      the encoder's Q8 matrix multiplications on every GPU
      parakeet-runtime --session-memory <clip.wav> [options]       whether memory goes back after repeated sessions
    """);
return args.Length == 0 ? 0 : 2;
