using System.Reflection;
using System.Text;
using ParakeetRuntime;
using ParakeetRuntime.Cli;

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

Console.OutputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, press) =>
{
    press.Cancel = !cancellation.IsCancellationRequested;
    cancellation.Cancel();
};
return await ConsoleCommands.RunAsync(args, SpeechModel.Default(), Console.In, Console.Out, Console.Error, cancellation.Token);
