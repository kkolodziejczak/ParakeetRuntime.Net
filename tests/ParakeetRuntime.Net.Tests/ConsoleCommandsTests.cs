using ParakeetRuntime.Net;
using ParakeetRuntime.Net.Cli;

namespace ParakeetRuntime.Net.Tests;

internal sealed class InstalledModelAndMicrophoneFactAttribute : FactAttribute
{
    public InstalledModelAndMicrophoneFactAttribute()
    {
        if (!SpeechModel.Default().IsInstalled)
            Skip = "The speech model is not installed on this machine.";
        else if (new MicrophoneFactAttribute().Skip is { } noMicrophone)
            Skip = noMicrophone;
    }
}

internal sealed class DefaultPortFreeFactAttribute : FactAttribute
{
    public DefaultPortFreeFactAttribute()
    {
        try
        {
            using var probe = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 5092);
            probe.Start();
        }
        catch (System.Net.Sockets.SocketException)
        {
            Skip = "Port 5092 is in use on this machine.";
        }
    }
}

public sealed class ConsoleCommandsTests : IDisposable
{
    private readonly string folder = Path.Combine(Path.GetTempPath(), $"parakeet-cli-{Guid.NewGuid():N}");
    private readonly StringWriter output = new();
    private readonly StringWriter errors = new();
    private readonly List<string> openedPages = [];

    public ConsoleCommandsTests() => Directory.CreateDirectory(folder);

    public void Dispose() => Directory.Delete(folder, recursive: true);

    private SpeechModel MissingModel => new(Path.Combine(folder, "models"), SpeechModel.Parakeet);

    private Task<int> Run(SpeechModel model, string input, params string[] args) =>
        ConsoleCommands.RunAsync(args, model, new StringReader(input), output, errors, openedPages.Add, CancellationToken.None);

    [Fact]
    public async Task Help_lists_every_command_on_standard_output()
    {
        var exit = await Run(MissingModel, "", "--help");

        Assert.Equal(0, exit);
        Assert.Contains("transcribe <file.wav>", output.ToString());
        Assert.Contains("record", output.ToString());
        Assert.Contains("download", output.ToString());
        Assert.Contains("--no-browser", output.ToString());
    }

    [Theory]
    [InlineData("translate")]
    [InlineData("transcribe")]
    [InlineData("transcribe", "a.wav", "--device", "tpu")]
    [InlineData("transcribe", "a.wav", "--colour", "red")]
    [InlineData("download", "--from")]
    [InlineData("transcribe", "")]
    [InlineData("serve", "--port", "http")]
    [InlineData("serve", "--port", "70000")]
    [InlineData("serve", "--port", "80")]
    [InlineData("download", "--from", " ")]
    public async Task A_command_line_that_is_not_understood_shows_the_usage_and_exits_with_2(params string[] args)
    {
        var exit = await Run(MissingModel, "", args);

        Assert.Equal(2, exit);
        Assert.Contains("transcribe <file.wav>", errors.ToString());
        Assert.Empty(output.ToString());
    }

    [Theory]
    [InlineData("transcribe", "speech.wav")]
    [InlineData("record")]
    public async Task Without_the_model_a_command_says_how_to_download_it(params string[] args)
    {
        var exit = await Run(MissingModel, "\n", args);

        Assert.Equal(1, exit);
        Assert.Contains("parakeet-runtime download", errors.ToString());
        Assert.Contains(SpeechModel.Parakeet.Source.ToString(), errors.ToString());
        Assert.Empty(output.ToString());
    }

    [InstalledModelFact]
    public async Task Transcribing_a_missing_file_names_it_and_exits_with_1()
    {
        var missing = Path.Combine(folder, "missing.wav");

        var exit = await Run(SpeechModel.Default(), "", "transcribe", missing, "--device", "cpu");

        Assert.Equal(1, exit);
        Assert.Contains("missing.wav", errors.ToString());
        Assert.Empty(output.ToString());
    }

    [InstalledModelFact]
    public async Task Transcribing_a_file_that_is_not_WAV_says_so_and_exits_with_1()
    {
        var mp3 = Path.Combine(folder, "song.wav");
        await File.WriteAllBytesAsync(mp3, "ID3 an mp3 file"u8.ToArray());

        var exit = await Run(SpeechModel.Default(), "", "transcribe", mp3, "--device", "cpu");

        Assert.Equal(1, exit);
        Assert.Contains("not a WAV file", errors.ToString());
        Assert.Empty(output.ToString());
    }

    [InstalledModelFact]
    public async Task Transcribe_prints_exactly_what_the_library_returns_for_the_file()
    {
        var wav = Path.Combine(folder, "tone.wav");
        await File.WriteAllBytesAsync(wav, ToneWav(seconds: 2));
        using var engine = await SpeechEngine.LoadAsync(SpeechModel.Default(), new SpeechEngineOptions(UseGpu: false));
        var expected = await engine.TranscribeWavAsync(wav);

        var exit = await Run(SpeechModel.Default(), "", "transcribe", wav, "--device", "cpu");

        Assert.Equal(0, exit);
        Assert.Equal(expected + Environment.NewLine, output.ToString());
    }

    [InstalledModelFact]
    public async Task Download_with_the_model_installed_says_so_and_downloads_nothing()
    {
        var exit = await Run(SpeechModel.Default(), "", "download");

        Assert.Equal(0, exit);
        Assert.Contains("already installed", errors.ToString());
    }

    [Fact]
    public async Task Download_from_a_file_that_is_not_the_model_installs_nothing_and_exits_with_1()
    {
        var notTheModel = Path.Combine(folder, "other.gguf");
        await File.WriteAllBytesAsync(notTheModel, new byte[1024]);
        var model = MissingModel;

        var exit = await Run(model, "", "download", "--from", notTheModel);

        Assert.Equal(1, exit);
        Assert.Contains("does not match", errors.ToString());
        Assert.False(model.IsInstalled);
    }

    [InstalledModelFact]
    public async Task Download_from_a_local_copy_installs_the_model()
    {
        var model = MissingModel;

        var exit = await Run(model, "", "download", "--from", SpeechModel.Default().Path);

        Assert.Equal(0, exit);
        Assert.True(model.IsInstalled);
        Assert.Contains(model.Path, errors.ToString());
    }

    [InstalledModelAndMicrophoneFact]
    public async Task Record_records_until_Enter_and_prints_what_it_heard()
    {
        var exit = await Run(SpeechModel.Default(), "\n", "record", "--device", "cpu");

        Assert.Equal(0, exit);
        Assert.Contains("Press Enter", errors.ToString());
        Assert.EndsWith(Environment.NewLine, output.ToString());
    }

    [InstalledModelAndMicrophoneFact]
    public async Task Cancelling_record_while_it_waits_for_Enter_ends_it_at_once()
    {
        using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(500));

        var exit = await ConsoleCommands.RunAsync(["record", "--device", "cpu"], SpeechModel.Default(), new ReaderThatNeverAnswers(), output, errors, openedPages.Add, cancellation.Token)
            .WaitAsync(TimeSpan.FromSeconds(15));

        Assert.Equal(1, exit);
        Assert.Contains("Cancelled.", errors.ToString());
        Assert.Empty(output.ToString());
    }

    [Fact]
    public async Task Serve_answers_on_its_port_until_cancelled_and_says_how_to_get_the_model()
    {
        var port = FreePort();

        var (health, exit) = await ServeUntilItAnswersAsync(port, ["serve", "--port", port.ToString(), "--device", "cpu", "--no-browser"]);

        Assert.Equal(System.Net.HttpStatusCode.ServiceUnavailable, health);
        Assert.Equal(0, exit);
        Assert.Contains($"http://127.0.0.1:{port}/", errors.ToString());
        Assert.Contains("parakeet-runtime download", errors.ToString());
        Assert.Empty(openedPages);
    }

    [Fact]
    public async Task Serve_opens_its_page_in_the_browser()
    {
        var port = FreePort();

        await ServeUntilItAnswersAsync(port, ["serve", "--device", "cpu", "--port", port.ToString()]);

        Assert.Equal([$"http://127.0.0.1:{port}/"], openedPages);
    }

    [Fact]
    public async Task Serve_with_no_browser_opens_nothing()
    {
        var port = FreePort();

        await ServeUntilItAnswersAsync(port, ["serve", "--no-browser", "--port", port.ToString()]);

        Assert.Empty(openedPages);
    }

    [DefaultPortFreeFact]
    public async Task Without_arguments_it_serves_on_port_5092_and_opens_the_page()
    {
        var (_, exit) = await ServeUntilItAnswersAsync(5092, []);

        Assert.Equal(0, exit);
        Assert.Equal(["http://127.0.0.1:5092/"], openedPages);
    }

    [DefaultPortFreeFact]
    public async Task Without_arguments_an_error_stays_on_screen_until_Enter()
    {
        using var taken = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 5092);
        taken.Start();
        var input = new ReaderThatAnswersWhenAsked();

        var running = ConsoleCommands.RunAsync([], MissingModel, input, output, errors, openedPages.Add, CancellationToken.None);
        await input.Asked.WaitAsync(TimeSpan.FromSeconds(15));
        var stillOpen = !running.IsCompleted;
        input.Answer();
        var exit = await running.WaitAsync(TimeSpan.FromSeconds(15));

        Assert.True(stillOpen);
        Assert.Equal(1, exit);
        Assert.Contains("Port 5092 is already in use", errors.ToString());
        Assert.Contains("Press Enter to close", errors.ToString());
        Assert.Empty(openedPages);
    }

    private sealed class ReaderThatAnswersWhenAsked : TextReader
    {
        private readonly TaskCompletionSource asked = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource<string?> answer = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public Task Asked => asked.Task;

        public void Answer() => answer.TrySetResult("");

        public override ValueTask<string?> ReadLineAsync(CancellationToken cancellationToken)
        {
            asked.TrySetResult();
            return new(answer.Task.WaitAsync(cancellationToken));
        }
    }

    private async Task<(System.Net.HttpStatusCode Health, int Exit)> ServeUntilItAnswersAsync(int port, string[] args)
    {
        using var stop = new CancellationTokenSource();
        var serving = ConsoleCommands.RunAsync(args, MissingModel, new StringReader(""), output, errors, openedPages.Add, stop.Token);
        using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };
        var health = await HealthWhenUpAsync(client);
        await stop.CancelAsync();
        return (health, await serving.WaitAsync(TimeSpan.FromSeconds(15)));
    }

    [Fact]
    public async Task Serve_on_a_port_in_use_says_so_and_exits_with_1()
    {
        using var taken = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
        taken.Start();
        var port = ((System.Net.IPEndPoint)taken.LocalEndpoint).Port;

        var exit = await Run(MissingModel, "", "serve", "--port", port.ToString());

        Assert.Equal(1, exit);
        Assert.Contains($"Port {port} is already in use", errors.ToString());
    }

    private static int FreePort()
    {
        using var listener = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
        listener.Start();
        return ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
    }

    private static async Task<System.Net.HttpStatusCode> HealthWhenUpAsync(HttpClient client)
    {
        var deadline = DateTime.UtcNow.AddSeconds(10);
        while (true)
        {
            try
            {
                using var response = await client.GetAsync("/health");
                return response.StatusCode;
            }
            catch (HttpRequestException) when (DateTime.UtcNow < deadline)
            {
                await Task.Delay(50);
            }
        }
    }

    [Theory]
    [InlineData("transcribe", "--list-gpus")]
    [InlineData("record", "--corpus", "list.txt")]
    public void A_user_command_with_a_benchmark_flag_is_not_run_as_a_benchmark(params string[] args)
    {
        var (exit, standardOutput) = RunTheExe(args);

        Assert.Equal(2, exit);
        Assert.Empty(standardOutput);
    }

    private static (int Exit, string StandardOutput) RunTheExe(string[] args)
    {
        var start = new System.Diagnostics.ProcessStartInfo(Path.Combine(AppContext.BaseDirectory, "parakeet-runtime.exe"))
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var arg in args)
            start.ArgumentList.Add(arg);
        using var exe = System.Diagnostics.Process.Start(start)!;
        var standardOutput = exe.StandardOutput.ReadToEndAsync();
        _ = exe.StandardError.ReadToEndAsync();
        if (!exe.WaitForExit(30_000))
        {
            exe.Kill(entireProcessTree: true);
            Assert.Fail("parakeet-runtime.exe did not exit within 30 seconds.");
        }
        return (exe.ExitCode, standardOutput.Result);
    }

    private sealed class ReaderThatNeverAnswers : TextReader
    {
        private readonly TaskCompletionSource<string?> never = new();

        public override Task<string?> ReadLineAsync() => never.Task;

        public override ValueTask<string?> ReadLineAsync(CancellationToken cancellationToken) => new(never.Task);
    }

    private static byte[] ToneWav(int seconds)
    {
        var samples = new short[seconds * 16_000];
        for (var i = 0; i < samples.Length; i++)
            samples[i] = (short)(8000 * Math.Sin(2 * Math.PI * 440 * i / 16_000.0));
        var data = new byte[samples.Length * 2];
        Buffer.BlockCopy(samples, 0, data, 0, data.Length);
        return [.. WavAudioTests.WavHeader(1, 16, 1, 16_000, (uint)data.Length), .. data];
    }
}
