using ParakeetRuntime.Net;

namespace ParakeetRuntime.Net.Cli;

internal static class ConsoleCommands
{
    public const string Usage = """
        Usage:
          parakeet-runtime                                                       the same as serve: opens the page in the browser
          parakeet-runtime transcribe <file.wav> [--device cpu|gpu|gpu:<name>]   prints the text of a WAV file
          parakeet-runtime record [--device cpu|gpu|gpu:<name>]                  records the default microphone until Enter, prints the text
          parakeet-runtime download [--from <file.gguf>]                         installs the speech model (714 MB), or from a local copy
          parakeet-runtime serve [--port 5092] [--device cpu|gpu|gpu:<name>] [--no-browser]
                                                                                 serves the page and the HTTP API on 127.0.0.1 until Ctrl+C
          parakeet-runtime --help | --version

        The text goes to standard output; messages go to standard error. --device defaults to a GPU
        when one is usable, else the CPU.

        HTTP API of serve:
          GET  /                          the page: drop a WAV, read its text.
          POST /v1/audio/transcriptions   multipart/form-data with one 'file' part (a WAV); optional
                                          response_format=text. Answers {"text": ..., "duration": seconds}.
          GET  /health                    the model's status: no-model, downloading, loading, ready or failed.

        Benchmarks:
          parakeet-runtime --list-gpus
          parakeet-runtime --corpus <list.txt | wav folder> [options]   speed, word error rate and memory on a list of WAVs
          parakeet-runtime --cpu-bench [options]                       the CPU encoder's kernels per layer
          parakeet-runtime --gemm-bench [options]                      the encoder's Q8 matrix multiplications on every GPU
          parakeet-runtime --session-memory <clip.wav> [options]       whether memory goes back after repeated sessions
        """;

    private const int Done = 0, Failed = 1, NotUnderstood = 2;
    private const int DefaultPort = 5092;
    private const string NoBrowser = "--no-browser";
    private const int BrowsersDropThisPortFromTheHostHeader = 80;

    public static bool IsUserCommand(string[] args) => args is [] or ["transcribe" or "record" or "download" or "serve" or "--help" or "-h" or "/?", ..];

    public static async Task<int> RunAsync(string[] args, SpeechModel model, TextReader input, TextWriter output, TextWriter errors,
        Action<string> openPage, CancellationToken cancellationToken)
    {
        try
        {
            return args switch
            {
                [] => await KeptOnScreenIfFailedAsync(ServeAsync(DefaultPort, new SpeechEngineOptions(), openPage, model, errors, cancellationToken), input, errors, cancellationToken),
                ["--help" or "-h" or "/?"] => ShowUsage(output, Done),
                ["transcribe", var path, .. var rest] when IsPath(path) && DeviceOption(rest) is { } options =>
                    await TranscribeAsync(path, options, model, output, errors, cancellationToken),
                ["record", .. var rest] when DeviceOption(rest) is { } options =>
                    await RecordAsync(options, model, input, output, errors, cancellationToken),
                ["download"] => await DownloadAsync(from: null, model, errors, cancellationToken),
                ["download", "--from", var from] when IsPath(from) => await DownloadAsync(from, model, errors, cancellationToken),
                ["serve", .. var rest] when ServeOptions(rest.Where(arg => arg != NoBrowser).ToArray()) is var (port, options) =>
                    await ServeAsync(port, options, rest.Contains(NoBrowser) ? _ => { } : openPage, model, errors, cancellationToken),
                _ => ShowUsage(errors, NotUnderstood)
            };
        }
        catch (OperationCanceledException)
        {
            errors.WriteLine("Cancelled.");
            return Failed;
        }
    }

    private static async Task<int> KeptOnScreenIfFailedAsync(Task<int> doubleClickedRun, TextReader input, TextWriter errors, CancellationToken cancellationToken)
    {
        var exit = await doubleClickedRun;
        if (exit == Failed)
        {
            errors.WriteLine("Press Enter to close.");
            await input.ReadLineAsync(cancellationToken);
        }
        return exit;
    }

    private static bool IsPath(string arg) => !string.IsNullOrWhiteSpace(arg) && !arg.StartsWith("--", StringComparison.Ordinal);

    private static int ShowUsage(TextWriter to, int exitCode)
    {
        to.WriteLine(Usage);
        return exitCode;
    }

    private static SpeechEngineOptions? DeviceOption(string[] rest) => rest switch
    {
        [] => new SpeechEngineOptions(),
        ["--device", "cpu"] => new SpeechEngineOptions(UseGpu: false),
        ["--device", "gpu"] => new SpeechEngineOptions(),
        ["--device", var device] when device.StartsWith("gpu:", StringComparison.Ordinal) && device.Length > 4 =>
            new SpeechEngineOptions(GpuNameContains: device[4..]),
        _ => null
    };

    private static (int Port, SpeechEngineOptions Engine)? ServeOptions(string[] rest) => rest switch
    {
        ["--port", var port, .. var more] when IsPort(port, out var number) && DeviceOption(more) is { } engine => (number, engine),
        [.. var device, "--port", var port] when IsPort(port, out var number) && DeviceOption(device) is { } engine => (number, engine),
        _ when DeviceOption(rest) is { } engine => (DefaultPort, engine),
        _ => null
    };

    private static bool IsPort(string text, out int port) =>
        int.TryParse(text, out port) && port is > 0 and <= ushort.MaxValue && port != BrowsersDropThisPortFromTheHostHeader;

    private static async Task<int> ServeAsync(int port, SpeechEngineOptions options, Action<string> openPage, SpeechModel model, TextWriter errors, CancellationToken cancellationToken)
    {
        var settings = new ServerSettings(port, model,
            loading => SpeechEngine.LoadAsync(model, options, loading),
            (progress, installing) => model.EnsureInstalledAsync(progress, installing));
        try
        {
            await using var server = await LocalServer.StartAsync(settings, cancellationToken);
            errors.WriteLine($"Serving on {server.Origin}/ until Ctrl+C. Transcribe with:");
            errors.WriteLine($"  curl.exe -F file=@speech.wav {server.Origin}/v1/audio/transcriptions");
            if (!model.IsInstalled)
                ModelMissing(model, errors);
            openPage($"{server.Origin}/");
            await Task.Delay(Timeout.Infinite, cancellationToken);
            return Done;
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            errors.WriteLine("Stopped.");
            return Done;
        }
        catch (IOException exception)
        {
            errors.WriteLine(exception.Message);
            return Failed;
        }
    }

    private static async Task<int> TranscribeAsync(string path, SpeechEngineOptions options, SpeechModel model, TextWriter output, TextWriter errors, CancellationToken cancellationToken)
    {
        if (!model.IsInstalled)
            return ModelMissing(model, errors);
        try
        {
            using var wav = File.OpenRead(path);
            using var engine = await SpeechEngine.LoadAsync(model, options, cancellationToken);
            errors.WriteLine($"Transcribing {Path.GetFileName(path)} on {engine.Backend}.");
            output.WriteLine(await engine.TranscribeWavAsync(wav, cancellationToken));
            return Done;
        }
        catch (Exception exception) when (exception is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException)
        {
            errors.WriteLine(exception is FileNotFoundException or DirectoryNotFoundException ? $"There is no file {Path.GetFullPath(path)}." : exception.Message);
            return Failed;
        }
    }

    private static async Task<int> RecordAsync(SpeechEngineOptions options, SpeechModel model, TextReader input, TextWriter output, TextWriter errors, CancellationToken cancellationToken)
    {
        if (!model.IsInstalled)
            return ModelMissing(model, errors);
        try
        {
            using var recording = Microphone.StartRecording();
            var loading = SpeechEngine.LoadAsync(model, options, cancellationToken);
            float[] samples;
            try
            {
                errors.WriteLine("Recording from the default microphone (at most 10 minutes). Press Enter to stop.");
                await input.ReadLineAsync(cancellationToken).AsTask().WaitAsync(cancellationToken);
                samples = recording.Stop();
            }
            catch
            {
                _ = loading.ContinueWith(loaded => loaded.Result.Dispose(), TaskContinuationOptions.OnlyOnRanToCompletion);
                throw;
            }
            using var engine = await loading;
            errors.WriteLine($"Transcribing {samples.Length / 16_000.0:0.0} s on {engine.Backend}.");
            output.WriteLine(await engine.TranscribeAsync(samples, cancellationToken));
            return Done;
        }
        catch (Exception exception) when (exception is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            errors.WriteLine(exception.Message);
            return Failed;
        }
    }

    private static int ModelMissing(SpeechModel model, TextWriter errors)
    {
        errors.WriteLine($"""
            The speech model is not installed. Install it first with:
              parakeet-runtime download
            It downloads {model.Pinned.Bytes:N0} bytes from {model.Pinned.Source}
            into {model.Folder}.
            """);
        return Failed;
    }

    private static async Task<int> DownloadAsync(string? from, SpeechModel model, TextWriter errors, CancellationToken cancellationToken)
    {
        if (from is null && model.IsInstalled)
        {
            errors.WriteLine($"The speech model is already installed at {model.Path}.");
            return Done;
        }
        var progress = new PercentProgress(errors, from is null ? "Downloading the speech model" : "Installing the speech model");
        try
        {
            if (from is null)
                await model.EnsureInstalledAsync(progress, cancellationToken);
            else
                await model.InstallFromFileAsync(from, progress, cancellationToken);
            errors.WriteLine();
            errors.WriteLine($"The speech model is installed at {model.Path}.");
            return Done;
        }
        catch (Exception exception) when (exception is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException or HttpRequestException or TimeoutException)
        {
            errors.WriteLine();
            errors.WriteLine(exception is FileNotFoundException or DirectoryNotFoundException && from is not null ? $"There is no file {Path.GetFullPath(from)}." : exception.Message);
            return Failed;
        }
    }

    private sealed class PercentProgress(TextWriter errors, string what) : IProgress<double>
    {
        private int shown = -1;

        public void Report(double fraction)
        {
            var percent = (int)(fraction * 100);
            if (percent == shown)
                return;
            shown = percent;
            errors.Write($"\r{what}: {percent}%");
        }
    }
}
