using System.Net;
using System.Reflection;
using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Connections;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.WebUtilities;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Net.Http.Headers;
using ParakeetRuntime;

namespace ParakeetRuntime.Cli;

internal sealed class LocalServer : IAsyncDisposable
{
    private const int MostRunning = 1, MostWaiting = 1;
    private const int LongestFieldValue = 64;
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web) { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    private readonly WebApplication app;
    private readonly ModelState state;
    private readonly string authority;
    private readonly long uploadByteLimit;
    private readonly SemaphoreSlim admitted = new(MostRunning + MostWaiting);
    private readonly SemaphoreSlim running = new(MostRunning);

    private LocalServer(ServerSettings settings)
    {
        authority = $"127.0.0.1:{settings.Port}";
        uploadByteLimit = settings.UploadByteLimit;
        Origin = $"http://{authority}";
        state = new ModelState(settings, Version);
        var builder = WebApplication.CreateEmptyBuilder(new WebApplicationOptions());
        builder.WebHost.UseKestrelCore().ConfigureKestrel(kestrel =>
        {
            kestrel.Listen(IPAddress.Loopback, settings.Port);
            kestrel.Limits.MaxRequestBodySize = settings.UploadByteLimit;
            kestrel.AddServerHeader = false;
        });
        builder.Services.AddRoutingCore();
        app = builder.Build();
        app.Use(RefuseOtherHostsAndPages);
        app.MapGet("/health", HealthAsync);
        app.MapPost("/model/download", DownloadAsync);
        app.MapPost("/v1/audio/transcriptions", TranscribeAsync);
    }

    public string Origin { get; }

    private static string Version =>
        Assembly.GetEntryAssembly()?.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";

    public static async Task<LocalServer> StartAsync(ServerSettings settings, CancellationToken cancellationToken)
    {
        var server = new LocalServer(settings);
        try
        {
            await server.app.StartAsync(cancellationToken);
        }
        catch (Exception bindFailure) when (bindFailure is AddressInUseException || bindFailure.InnerException is AddressInUseException)
        {
            await server.DisposeAsync();
            throw new IOException($"Port {settings.Port} is already in use. Choose another with --port.");
        }
        server.state.LoadIfInstalled();
        return server;
    }

    private async Task RefuseOtherHostsAndPages(HttpContext context, RequestDelegate next)
    {
        if (context.Request.Host.Value != authority)
        {
            await ErrorAsync(context, StatusCodes.Status421MisdirectedRequest, "wrong-host", $"Use {Origin}/.");
            return;
        }
        if (HttpMethods.IsPost(context.Request.Method) && !FromThisServersPageOrNoPage(context.Request))
        {
            await ErrorAsync(context, StatusCodes.Status403Forbidden, "cross-origin", "Requests from other web pages are refused.");
            return;
        }
        await next(context);
    }

    private bool FromThisServersPageOrNoPage(HttpRequest request)
    {
        var origin = request.Headers.Origin.ToString();
        var site = request.Headers["Sec-Fetch-Site"].ToString();
        return (origin.Length == 0 || origin == Origin) && (site.Length == 0 || site == "same-origin");
    }

    private async Task HealthAsync(HttpContext context)
    {
        var health = state.Snapshot();
        context.Response.StatusCode = health.Status == "ready" ? StatusCodes.Status200OK : StatusCodes.Status503ServiceUnavailable;
        await context.Response.WriteAsJsonAsync(health, Json);
    }

    private async Task DownloadAsync(HttpContext context)
    {
        if (context.Request.Headers["X-Parakeet-Request"] != "1" || context.Request.Headers.Origin != Origin)
        {
            await ErrorAsync(context, StatusCodes.Status403Forbidden, "page-only", "Only this server's page can start the download (its Origin and X-Parakeet-Request: 1).");
            return;
        }
        if (!state.TryStartInstall())
        {
            await ErrorAsync(context, StatusCodes.Status409Conflict, "not-now", $"The model is {state.Snapshot().Status}.");
            return;
        }
        context.Response.StatusCode = StatusCodes.Status202Accepted;
        await context.Response.WriteAsJsonAsync(state.Snapshot(), Json);
    }

    private async Task TranscribeAsync(HttpContext context)
    {
        if (state.Engine is not { } engine)
        {
            await ErrorAsync(context, StatusCodes.Status503ServiceUnavailable, "not-ready", $"The model is {state.Snapshot().Status}; see /health.");
            return;
        }
        if (!admitted.Wait(0))
        {
            await ErrorAsync(context, StatusCodes.Status503ServiceUnavailable, "busy", "One transcription is running and one is waiting. Try again shortly.");
            return;
        }
        try
        {
            await running.WaitAsync(context.RequestAborted);
            try
            {
                await TranscribeUploadAsync(context, engine);
            }
            finally
            {
                running.Release();
            }
        }
        catch (OperationCanceledException) when (context.RequestAborted.IsCancellationRequested)
        {
        }
        finally
        {
            admitted.Release();
        }
    }

    private async Task TranscribeUploadAsync(HttpContext context, SpeechEngine engine)
    {
        var aborted = context.RequestAborted;
        if (!MediaTypeHeaderValue.TryParse(context.Request.ContentType, out var type)
            || !type.MediaType.Equals("multipart/form-data", StringComparison.OrdinalIgnoreCase)
            || HeaderUtilities.RemoveQuotes(type.Boundary).Value is not { Length: > 0 } boundary)
        {
            await ErrorAsync(context, StatusCodes.Status415UnsupportedMediaType, "not-multipart", "Send multipart/form-data with one 'file' part holding a WAV.");
            return;
        }
        var reader = new MultipartReader(boundary, context.Request.Body);
        using var wav = new MemoryStream();
        var files = 0;
        var asText = false;
        try
        {
            while (await reader.ReadNextSectionAsync(aborted) is { } section)
            {
                var name = ContentDispositionHeaderValue.TryParse(section.ContentDisposition, out var disposition) ? disposition.Name.Value : null;
                if (name == "file" && ++files == 1)
                    await section.Body.CopyToAsync(wav, aborted);
                else if (name == "response_format")
                    asText = await ShortValueAsync(section.Body, aborted) == "text";
                else
                    await section.Body.DrainAsync(aborted);
            }
        }
        catch (BadHttpRequestException tooLarge) when (tooLarge.StatusCode == StatusCodes.Status413PayloadTooLarge)
        {
            await ErrorAsync(context, StatusCodes.Status413PayloadTooLarge, "too-large", $"Uploads are limited to {uploadByteLimit:N0} bytes.");
            return;
        }
        catch (Exception malformed) when (malformed is IOException or InvalidDataException && !aborted.IsCancellationRequested)
        {
            await ErrorAsync(context, StatusCodes.Status400BadRequest, "bad-request", "The multipart body is malformed.");
            return;
        }
        if (files != 1)
        {
            await ErrorAsync(context, StatusCodes.Status400BadRequest, files == 0 ? "no-file" : "two-files", "Send exactly one 'file' part holding a WAV.");
            return;
        }
        wav.Position = 0;
        float[] samples;
        try
        {
            samples = await Task.Run(() => WavAudio.Read16KhzMono(wav, WavAudio.LongestAudio), aborted);
        }
        catch (InvalidDataException rejected)
        {
            var (status, code) = WavAudio.ProblemOf(rejected) switch
            {
                WavProblem.NotWav => (StatusCodes.Status415UnsupportedMediaType, "not-wav"),
                WavProblem.Unsupported => (StatusCodes.Status415UnsupportedMediaType, "unsupported-wav"),
                WavProblem.TooLong => (StatusCodes.Status422UnprocessableEntity, "too-long"),
                _ => (StatusCodes.Status400BadRequest, "damaged-wav")
            };
            await ErrorAsync(context, status, code, rejected.Message);
            return;
        }
        var text = await engine.TranscribeAsync(samples, aborted);
        if (asText)
        {
            context.Response.ContentType = "text/plain; charset=utf-8";
            await context.Response.WriteAsync(text, aborted);
        }
        else
            await context.Response.WriteAsJsonAsync(new { text, duration = Math.Round(samples.Length / (double)Wav.SampleRate, 3) }, Json, aborted);
    }

    private static async Task<string> ShortValueAsync(Stream body, CancellationToken cancellationToken)
    {
        var buffer = new char[LongestFieldValue];
        using var reader = new StreamReader(body, leaveOpen: true);
        var read = await reader.ReadBlockAsync(buffer, cancellationToken);
        await body.DrainAsync(cancellationToken);
        return new string(buffer, 0, read).Trim();
    }

    private static Task ErrorAsync(HttpContext context, int status, string code, string message)
    {
        context.Response.StatusCode = status;
        return context.Response.WriteAsJsonAsync(new { error = new { code, message } }, Json);
    }

    public async ValueTask DisposeAsync()
    {
        await app.StopAsync();
        await app.DisposeAsync();
        await state.DisposeAsync();
    }
}
