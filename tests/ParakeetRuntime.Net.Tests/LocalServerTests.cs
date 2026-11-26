using System.IO.Abstractions.TestingHelpers;
using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using ParakeetRuntime.Net;
using ParakeetRuntime.Net.Cli;

namespace ParakeetRuntime.Net.Tests;

public sealed class LocalServerTests : IAsyncDisposable
{
    private static readonly byte[] ModelBytes = "a tiny stand-in for the model"u8.ToArray();
    private static readonly SpeechModelFile TinyModel = new(
        new Uri("https://models.example/tiny.gguf"), "tiny.gguf", ModelBytes.Length, Convert.ToHexStringLower(SHA256.HashData(ModelBytes)));

    private readonly MockFileSystem files = new();
    private readonly SpeechModel model;
    private readonly FakeRecognizer recognizer = new();
    private readonly List<IAsyncDisposable> servers = [];

    public LocalServerTests()
    {
        files.AddFile(@"C:\copy\tiny.gguf", new MockFileData(ModelBytes));
        model = new SpeechModel(@"C:\models", TinyModel, files);
    }

    public async ValueTask DisposeAsync()
    {
        recognizer.Release();
        foreach (var server in servers)
            await server.DisposeAsync();
    }

    [Fact]
    public async Task Health_without_the_model_is_503_and_says_where_the_model_comes_from()
    {
        using var client = await StartAsync();

        using var response = await client.GetAsync("/health");
        using var health = await JsonOf(response);

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        Assert.Equal("no-model", health.RootElement.GetProperty("status").GetString());
        Assert.Equal(TinyModel.Source.ToString(), health.RootElement.GetProperty("model").GetProperty("source").GetString());
        Assert.Equal(ModelBytes.Length, health.RootElement.GetProperty("model").GetProperty("bytes").GetInt64());
        Assert.Equal(@"C:\models", health.RootElement.GetProperty("model").GetProperty("folder").GetString());
        Assert.False(health.RootElement.GetProperty("model").GetProperty("installed").GetBoolean());
    }

    [Fact]
    public async Task Health_with_the_model_is_200_once_it_is_loaded_and_names_the_backend()
    {
        await InstallTheModelAsync();
        using var client = await StartAsync();

        using var health = await ReadyHealthAsync(client);

        Assert.Equal("CPU (test)", health.RootElement.GetProperty("backend").GetString());
        Assert.True(health.RootElement.GetProperty("model").GetProperty("installed").GetBoolean());
    }

    [Fact]
    public async Task The_page_is_one_HTML_file_at_the_root_that_loads_nothing_from_elsewhere()
    {
        using var client = await StartAsync();

        using var response = await client.GetAsync("/");
        var html = await response.Content.ReadAsStringAsync();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/html", response.Content.Headers.ContentType?.MediaType);
        Assert.Equal("utf-8", response.Content.Headers.ContentType?.CharSet);
        Assert.Contains("<title>Parakeet Runtime</title>", html);
        Assert.DoesNotMatch("""(src|href|url)\s*[=(]\s*["']?(https?:)?//""", html);
    }

    [Fact]
    public async Task The_page_may_not_be_framed_or_reach_other_servers()
    {
        using var client = await StartAsync();

        using var response = await client.GetAsync("/");
        var policy = response.Headers.GetValues("Content-Security-Policy").Single().Split(';', StringSplitOptions.TrimEntries)
            .Select(directive => directive.Split(' ', 2)).ToDictionary(parts => parts[0], parts => parts[1]);

        Assert.Equal("'none'", policy["default-src"]);
        Assert.Equal("'self'", policy["connect-src"]);
        Assert.Equal("'none'", policy["frame-ancestors"]);
        Assert.Matches("^'sha256-[A-Za-z0-9+/]{43}='$", policy["script-src"]);
        Assert.Equal("nosniff", response.Headers.GetValues("X-Content-Type-Options").Single());
    }

    [Fact]
    public async Task The_policy_allows_the_pages_script_as_a_browser_hashes_it()
    {
        using var client = await StartAsync();

        using var response = await client.GetAsync("/");
        var html = await response.Content.ReadAsStringAsync();
        var script = System.Text.RegularExpressions.Regex.Match(html, "<script>(.*?)</script>", System.Text.RegularExpressions.RegexOptions.Singleline).Groups[1].Value;
        var browsersHash = Convert.ToBase64String(SHA256.HashData(Encoding.UTF8.GetBytes(script.Replace("\r\n", "\n").Replace('\r', '\n'))));

        Assert.Contains($"script-src 'sha256-{browsersHash}'", response.Headers.GetValues("Content-Security-Policy").Single());
    }

    [Theory]
    [InlineData("\r\n")]
    [InlineData("\r")]
    public void A_page_checked_out_with_other_line_endings_still_matches_the_browsers_hash(string newline)
    {
        var lfHash = Convert.ToBase64String(SHA256.HashData("let a = 1;\nlet b = 2;\n"u8));

        var policy = Page.PolicyFor($"<p>x</p>{newline}<script>let a = 1;{newline}let b = 2;{newline}</script>");

        Assert.Contains($"script-src 'sha256-{lfHash}'", policy);
    }

    [Theory]
    [InlineData("'/health'")]
    [InlineData("'/model/download'")]
    [InlineData("'/v1/audio/transcriptions'")]
    [InlineData("'X-Parakeet-Request': '1'")]
    public async Task The_page_calls_the_servers_own_routes(string call)
    {
        using var client = await StartAsync();

        var html = await client.GetStringAsync("/");

        Assert.Contains(call, html);
    }

    [Fact]
    public async Task A_WAV_upload_returns_its_text_and_duration()
    {
        await InstallTheModelAsync();
        using var client = await StartReadyAsync();

        using var response = await client.PostAsync("/v1/audio/transcriptions", Upload(Wav(seconds: 1.5)));
        using var reply = await JsonOf(response);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("Hello there.", reply.RootElement.GetProperty("text").GetString());
        Assert.Equal(1.5, reply.RootElement.GetProperty("duration").GetDouble());
    }

    [Fact]
    public async Task Polish_text_comes_back_as_plain_UTF_8_in_the_JSON()
    {
        await InstallTheModelAsync();
        recognizer.Text = "Zażółć gęślą jaźń.";
        using var client = await StartReadyAsync();

        using var response = await client.PostAsync("/v1/audio/transcriptions", Upload(Wav(seconds: 1)));

        Assert.Contains("\"Zażółć gęślą jaźń.\"", Encoding.UTF8.GetString(await response.Content.ReadAsByteArrayAsync()));
    }

    [Fact]
    public async Task Response_format_text_returns_the_text_alone()
    {
        await InstallTheModelAsync();
        using var client = await StartReadyAsync();
        var form = Upload(Wav(seconds: 1));
        form.Add(new StringContent("text"), "response_format");

        using var response = await client.PostAsync("/v1/audio/transcriptions", form);

        Assert.Equal("text/plain", response.Content.Headers.ContentType?.MediaType);
        Assert.Equal("Hello there.", await response.Content.ReadAsStringAsync());
    }

    [Theory]
    [InlineData("localhost")]
    [InlineData("evil.example")]
    public async Task Requests_naming_another_host_are_refused(string host)
    {
        using var client = await StartAsync();
        using var request = new HttpRequestMessage(HttpMethod.Get, "/health");
        request.Headers.Host = $"{host}:{client.BaseAddress!.Port}";

        using var response = await client.SendAsync(request);

        Assert.Equal(HttpStatusCode.MisdirectedRequest, response.StatusCode);
    }

    [Theory]
    [InlineData("Origin", "http://evil.example")]
    [InlineData("Origin", "http://localhost:{port}")]
    [InlineData("Sec-Fetch-Site", "cross-site")]
    public async Task An_upload_from_another_web_page_is_refused_without_CORS_headers(string header, string value)
    {
        await InstallTheModelAsync();
        using var client = await StartReadyAsync();
        using var request = new HttpRequestMessage(HttpMethod.Post, "/v1/audio/transcriptions") { Content = Upload(Wav(seconds: 1)) };
        request.Headers.Add(header, value.Replace("{port}", client.BaseAddress!.Port.ToString()));

        using var response = await client.SendAsync(request);

        Assert.Equal(HttpStatusCode.Forbidden, response.StatusCode);
        Assert.False(response.Headers.Contains("Access-Control-Allow-Origin"));
        Assert.Empty(recognizer.Calls);
    }

    [Theory]
    [InlineData("Origin")]
    [InlineData("Sec-Fetch-Site")]
    public async Task A_page_header_that_is_present_but_empty_is_refused(string header)
    {
        await InstallTheModelAsync();
        using var client = await StartReadyAsync();
        using var request = new HttpRequestMessage(HttpMethod.Post, "/v1/audio/transcriptions") { Content = Upload(Wav(seconds: 1)) };
        Assert.True(request.Headers.TryAddWithoutValidation(header, ""));

        using var response = await client.SendAsync(request);

        Assert.Equal(HttpStatusCode.Forbidden, response.StatusCode);
    }

    [Fact]
    public async Task An_upload_slower_than_the_deadline_gives_up_its_place()
    {
        await InstallTheModelAsync();
        using var client = await StartReadyAsync(uploadDeadline: TimeSpan.FromMilliseconds(500));
        using var stopTrickling = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var trickles = Enumerable.Range(0, 2).Select(_ => SendQuietlyAsync(client, new TricklingUpload(), stopTrickling.Token)).ToList();
        await Task.Delay(1500);

        using var response = await client.PostAsync("/v1/audio/transcriptions", Upload(Wav(seconds: 1)));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        await stopTrickling.CancelAsync();
        await Task.WhenAll(trickles);
    }

    private static async Task SendQuietlyAsync(HttpClient client, HttpContent content, CancellationToken cancellationToken)
    {
        try
        {
            using var response = await client.PostAsync("/v1/audio/transcriptions", content, cancellationToken);
        }
        catch (Exception exception) when (exception is HttpRequestException or OperationCanceledException)
        {
        }
    }

    private sealed class TricklingUpload : HttpContent
    {
        public TricklingUpload() => Headers.ContentType = MediaTypeHeaderValue.Parse("multipart/form-data; boundary=slow");

        protected override async Task SerializeToStreamAsync(Stream stream, TransportContext? context)
        {
            await stream.WriteAsync("--slow\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n"u8.ToArray());
            for (var i = 0; i < 100; i++)
            {
                await stream.WriteAsync(new byte[16]);
                await stream.FlushAsync();
                await Task.Delay(100);
            }
        }

        protected override bool TryComputeLength(out long length)
        {
            length = 0;
            return false;
        }
    }

    [Fact]
    public async Task An_upload_from_the_servers_own_page_is_accepted()
    {
        await InstallTheModelAsync();
        using var client = await StartReadyAsync();
        using var request = new HttpRequestMessage(HttpMethod.Post, "/v1/audio/transcriptions") { Content = Upload(Wav(seconds: 1)) };
        request.Headers.Add("Origin", client.BaseAddress!.GetLeftPart(UriPartial.Authority));
        request.Headers.Add("Sec-Fetch-Site", "same-origin");

        using var response = await client.SendAsync(request);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    public static TheoryData<string, int, string> BadUploads => new()
    {
        { "not-multipart", 415, "not-multipart" },
        { "no-file", 400, "no-file" },
        { "two-files", 400, "two-files" },
        { "not-wav", 415, "not-wav" },
        { "unsupported-wav", 415, "unsupported-wav" },
        { "damaged-wav", 400, "damaged-wav" },
        { "too-long", 422, "too-long" },
        { "huge-boundary", 400, "bad-request" },
        { "unknown-format", 400, "bad-format" },
        { "empty-format", 400, "bad-format" },
        { "format-with-a-long-tail", 400, "bad-format" },
    };

    [Theory]
    [MemberData(nameof(BadUploads))]
    public async Task A_bad_upload_gets_its_own_status_and_error_code(string upload, int status, string code)
    {
        await InstallTheModelAsync();
        using var client = await StartReadyAsync();

        using var response = await client.PostAsync("/v1/audio/transcriptions", BadUpload(upload));
        using var reply = await JsonOf(response);

        Assert.Equal(status, (int)response.StatusCode);
        Assert.Equal(code, reply.RootElement.GetProperty("error").GetProperty("code").GetString());
        Assert.False(string.IsNullOrWhiteSpace(reply.RootElement.GetProperty("error").GetProperty("message").GetString()));
        Assert.Empty(recognizer.Calls);
    }

    [Fact]
    public async Task An_upload_over_the_byte_limit_is_413()
    {
        await InstallTheModelAsync();
        using var client = await StartReadyAsync(uploadByteLimit: 64 * 1024);

        using var response = await client.PostAsync("/v1/audio/transcriptions", Upload(Wav(seconds: 4)));
        using var reply = await JsonOf(response);

        Assert.Equal(HttpStatusCode.RequestEntityTooLarge, response.StatusCode);
        Assert.Equal("too-large", reply.RootElement.GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public async Task With_one_transcription_running_and_one_waiting_a_third_is_told_the_server_is_busy()
    {
        await InstallTheModelAsync();
        using var client = await StartReadyAsync();
        recognizer.HoldEachCall();

        var running = client.PostAsync("/v1/audio/transcriptions", Upload(Wav(seconds: 1)));
        await recognizer.FirstCallStarted.WaitAsync(TimeSpan.FromSeconds(10));
        var waiting = client.PostAsync("/v1/audio/transcriptions", Upload(Wav(seconds: 1)));
        await Task.Delay(300);
        using var third = await client.PostAsync("/v1/audio/transcriptions", Upload(Wav(seconds: 1)));
        using var busy = await JsonOf(third);
        recognizer.Release();

        Assert.Equal(HttpStatusCode.ServiceUnavailable, third.StatusCode);
        Assert.Equal("busy", busy.RootElement.GetProperty("error").GetProperty("code").GetString());
        Assert.Equal(HttpStatusCode.OK, (await running).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await waiting).StatusCode);
    }

    [Fact]
    public async Task A_waiting_client_that_gives_up_leaves_its_place_to_the_next()
    {
        await InstallTheModelAsync();
        using var client = await StartReadyAsync();
        recognizer.HoldEachCall();
        var running = client.PostAsync("/v1/audio/transcriptions", Upload(Wav(seconds: 1)));
        await recognizer.FirstCallStarted.WaitAsync(TimeSpan.FromSeconds(10));

        using (var givingUp = new CancellationTokenSource(TimeSpan.FromMilliseconds(300)))
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => client.PostAsync("/v1/audio/transcriptions", Upload(Wav(seconds: 1)), givingUp.Token));
        await Task.Delay(300);
        var next = client.PostAsync("/v1/audio/transcriptions", Upload(Wav(seconds: 1)));
        await Task.Delay(300);
        recognizer.Release();

        Assert.Equal(HttpStatusCode.OK, (await running).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await next).StatusCode);
    }

    [Fact]
    public async Task An_upload_before_the_model_is_ready_is_503()
    {
        using var client = await StartAsync();

        using var response = await client.PostAsync("/v1/audio/transcriptions", Upload(Wav(seconds: 1)));
        using var reply = await JsonOf(response);

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        Assert.Equal("not-ready", reply.RootElement.GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public async Task Downloading_the_model_needs_the_pages_origin_and_header_then_installs_and_loads_it()
    {
        using var client = await StartAsync();
        var origin = client.BaseAddress!.GetLeftPart(UriPartial.Authority);

        using var withoutHeader = await client.SendAsync(DownloadRequest(origin, withHeader: false));
        using var fromElsewhere = await client.SendAsync(DownloadRequest("http://evil.example", withHeader: true));
        using var started = await client.SendAsync(DownloadRequest(origin, withHeader: true));
        using var health = await ReadyHealthAsync(client);
        using var again = await client.SendAsync(DownloadRequest(origin, withHeader: true));

        Assert.Equal(HttpStatusCode.Forbidden, withoutHeader.StatusCode);
        Assert.Equal(HttpStatusCode.Forbidden, fromElsewhere.StatusCode);
        Assert.Equal(HttpStatusCode.Accepted, started.StatusCode);
        Assert.True(model.IsInstalled);
        Assert.Equal(HttpStatusCode.Conflict, again.StatusCode);
    }

    [Fact]
    public async Task A_model_that_fails_to_load_is_reported_in_health()
    {
        await InstallTheModelAsync();
        using var client = await StartAsync(load: _ => throw new InvalidDataException("The model file is damaged."));

        var health = await WaitForStatusAsync(client, "failed");

        Assert.Equal("The model file is damaged.", health.RootElement.GetProperty("error").GetString());
    }

    [Fact]
    public async Task After_a_failed_load_the_pages_request_loads_the_installed_model_again()
    {
        await InstallTheModelAsync();
        var loads = 0;
        using var client = await StartAsync(load: _ => ++loads == 1
            ? throw new InvalidDataException("The GPU was reset.")
            : Task.FromResult(new SpeechEngine(recognizer)));
        (await WaitForStatusAsync(client, "failed")).Dispose();

        using var again = await client.SendAsync(DownloadRequest(client.BaseAddress!.GetLeftPart(UriPartial.Authority), withHeader: true));
        using var health = await ReadyHealthAsync(client);

        Assert.Equal(HttpStatusCode.Accepted, again.StatusCode);
        Assert.Equal(2, loads);
    }

    private static HttpRequestMessage DownloadRequest(string origin, bool withHeader)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, "/model/download");
        request.Headers.Add("Origin", origin);
        if (withHeader)
            request.Headers.Add("X-Parakeet-Request", "1");
        return request;
    }

    private Task InstallTheModelAsync() => model.InstallFromFileAsync(@"C:\copy\tiny.gguf", null, CancellationToken.None);

    private async Task<HttpClient> StartReadyAsync(long uploadByteLimit = 256L << 20, TimeSpan? uploadDeadline = null)
    {
        var client = await StartAsync(uploadByteLimit: uploadByteLimit, uploadDeadline: uploadDeadline);
        (await ReadyHealthAsync(client)).Dispose();
        return client;
    }

    private async Task<HttpClient> StartAsync(Func<CancellationToken, Task<SpeechEngine>>? load = null, long uploadByteLimit = 256L << 20, TimeSpan? uploadDeadline = null)
    {
        var port = FreePort();
        var settings = new ServerSettings(
            port,
            model,
            load ?? (_ => Task.FromResult(new SpeechEngine(recognizer))),
            (progress, cancellationToken) => model.InstallFromFileAsync(@"C:\copy\tiny.gguf", progress, cancellationToken))
        {
            UploadByteLimit = uploadByteLimit,
            UploadDeadline = uploadDeadline ?? TimeSpan.FromMinutes(1),
        };
        servers.Add(await LocalServer.StartAsync(settings, CancellationToken.None));
        return new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}"), Timeout = TimeSpan.FromSeconds(30) };
    }

    private static int FreePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        return ((IPEndPoint)listener.LocalEndpoint).Port;
    }

    private static Task<JsonDocument> ReadyHealthAsync(HttpClient client) => WaitForStatusAsync(client, "ready");

    private static async Task<JsonDocument> WaitForStatusAsync(HttpClient client, string status)
    {
        var deadline = DateTime.UtcNow.AddSeconds(10);
        while (true)
        {
            using var response = await client.GetAsync("/health");
            var health = await JsonOf(response);
            if (health.RootElement.GetProperty("status").GetString() == status)
                return health;
            health.Dispose();
            Assert.True(DateTime.UtcNow < deadline, $"The server did not become {status} in time.");
            await Task.Delay(20);
        }
    }

    private static async Task<JsonDocument> JsonOf(HttpResponseMessage response) =>
        JsonDocument.Parse(await response.Content.ReadAsStringAsync());

    private static MultipartFormDataContent Upload(byte[] wav) => new() { { new ByteArrayContent(wav), "file", "audio.wav" } };

    private static HttpContent BadUpload(string kind) => kind switch
    {
        "not-multipart" => new ByteArrayContent(Wav(seconds: 1)) { Headers = { ContentType = new MediaTypeHeaderValue("audio/wav") } },
        "no-file" => new MultipartFormDataContent { { new StringContent("text"), "response_format" } },
        "two-files" => new MultipartFormDataContent { { new ByteArrayContent(Wav(seconds: 1)), "file", "a.wav" }, { new ByteArrayContent(Wav(seconds: 1)), "file", "b.wav" } },
        "not-wav" => Upload("ID3 an mp3 file"u8.ToArray()),
        "unsupported-wav" => Upload([.. WavAudioTests.WavHeader(1, 12, 1, 16_000, 64), .. new byte[64]]),
        "damaged-wav" => Upload(Wav(seconds: 1)[..^3]),
        "too-long" => Upload(WavAudioTests.WavHeader(1, 16, 1, 16_000, 16_000 * 2 * 601)),
        "huge-boundary" => new ByteArrayContent(Wav(seconds: 1)) { Headers = { ContentType = MediaTypeHeaderValue.Parse($"multipart/form-data; boundary={new string('x', 4096)}") } },
        "unknown-format" => WithFormat("xml"),
        "empty-format" => WithFormat(""),
        "format-with-a-long-tail" => WithFormat("text" + new string(' ', 60) + "xml"),
        _ => throw new ArgumentOutOfRangeException(nameof(kind))
    };

    private static MultipartFormDataContent WithFormat(string format)
    {
        var form = Upload(Wav(seconds: 1));
        form.Add(new StringContent(format), "response_format");
        return form;
    }

    private static byte[] Wav(double seconds)
    {
        var data = new byte[(int)(seconds * 16_000) * 2];
        return [.. WavAudioTests.WavHeader(1, 16, 1, 16_000, (uint)data.Length), .. data];
    }

    private sealed class FakeRecognizer : IUtteranceRecognizer
    {
        private readonly ManualResetEventSlim released = new(initialState: true);
        private readonly TaskCompletionSource firstCallStarted = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public List<int> Calls { get; } = [];
        public string Text { get; set; } = "Hello there.";
        public Task FirstCallStarted => firstCallStarted.Task;
        public string BackendName => "CPU (test)";
        public bool IsUsingGpu => false;

        public void HoldEachCall() => released.Reset();

        public void Release() => released.Set();

        public string Transcribe(float[] samples16KhzMono)
        {
            lock (Calls)
                Calls.Add(samples16KhzMono.Length);
            firstCallStarted.TrySetResult();
            released.Wait();
            return Text;
        }

        public void Dispose() { }
    }
}
