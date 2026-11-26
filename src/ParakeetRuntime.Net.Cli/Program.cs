using System.ComponentModel;
using System.Diagnostics;
using System.Reflection;
using System.Text;
using ParakeetRuntime.Net;
using ParakeetRuntime.Net.Cli;

var version = typeof(Program).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";
if (args is ["--version"])
{
    Console.WriteLine(version);
    return 0;
}
if (!ConsoleCommands.IsUserCommand(args))
{
    if (args.Contains("--corpus") || args.Contains("--list-gpus"))
        return await CorpusBench.RunAsync(args);
    if (args.Contains("--gemm-bench"))
        return GemmBench.Run(args);
    if (args.Contains("--cpu-bench"))
        return CpuBench.Run(args);
    if (args.Contains("--session-memory"))
        return await SessionMemory.RunAsync(args);
}

var consoleEncoding = Console.OutputEncoding;
Console.OutputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, press) =>
{
    press.Cancel = !cancellation.IsCancellationRequested;
    cancellation.Cancel();
};
try
{
    return await ConsoleCommands.RunAsync(args, SpeechModel.Default(), Console.In, Console.Out, Console.Error, OpenInBrowser, cancellation.Token);
}
finally
{
    Console.OutputEncoding = consoleEncoding;
}

static void OpenInBrowser(string url)
{
    try
    {
        Process.Start(new ProcessStartInfo(url) { UseShellExecute = true })?.Dispose();
    }
    catch (Win32Exception)
    {
        Console.Error.WriteLine($"Open {url} in your browser.");
    }
}
