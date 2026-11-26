using ParakeetRuntime.Net;

namespace ParakeetRuntime.Net.Cli;

internal sealed record ServerSettings(
    int Port,
    SpeechModel Model,
    Func<CancellationToken, Task<SpeechEngine>> LoadEngine,
    Func<IProgress<double>, CancellationToken, Task> InstallModel)
{
    public long UploadByteLimit { get; init; } = 256L * 1024 * 1024;

    public TimeSpan UploadDeadline { get; init; } = TimeSpan.FromMinutes(1);
}

internal sealed record ModelSource(string Source, long Bytes, string Folder, bool Installed);

internal sealed record HealthReply(string Status, double Progress, string Version, string? Backend, ModelSource Model, string? Error);

internal sealed class ModelState(ServerSettings settings, string version) : IAsyncDisposable
{
    private readonly Lock gate = new();
    private readonly CancellationTokenSource stopping = new();
    private string status = settings.Model.IsInstalled ? "loading" : "no-model";
    private double progress;
    private string? error;
    private Task work = Task.CompletedTask;
    private volatile SpeechEngine? engine;

    public SpeechEngine? Engine => engine;

    public HealthReply Snapshot()
    {
        var model = new ModelSource(settings.Model.Pinned.Source.ToString(), settings.Model.Pinned.Bytes, settings.Model.Folder, settings.Model.IsInstalled);
        lock (gate)
            return new HealthReply(status, Math.Round(progress, 3), version, engine?.Backend, model, error);
    }

    public void LoadIfInstalled()
    {
        if (settings.Model.IsInstalled)
            TryStart("loading", LoadAsync);
    }

    public bool TryStartInstall() => TryStart(settings.Model.IsInstalled ? "loading" : "downloading", async () =>
    {
        await settings.InstallModel(new ProgressInto(this), stopping.Token);
        await LoadAsync();
    });

    private bool TryStart(string newStatus, Func<Task> job)
    {
        lock (gate)
        {
            if (!work.IsCompleted || engine is not null || (newStatus == "downloading" && status is not ("no-model" or "failed")))
                return false;
            status = newStatus;
            progress = 0;
            error = null;
            work = Task.Run(async () =>
            {
                try
                {
                    await job();
                }
                catch (Exception failure)
                {
                    lock (gate)
                    {
                        status = "failed";
                        error = failure.Message;
                    }
                }
            });
            return true;
        }
    }

    private async Task LoadAsync()
    {
        lock (gate)
            status = "loading";
        var loaded = await settings.LoadEngine(stopping.Token);
        lock (gate)
        {
            engine = loaded;
            status = "ready";
        }
    }

    public async ValueTask DisposeAsync()
    {
        await stopping.CancelAsync();
        Task running;
        lock (gate)
            running = work;
        await running;
        engine?.Dispose();
        stopping.Dispose();
    }

    private sealed class ProgressInto(ModelState state) : IProgress<double>
    {
        public void Report(double value)
        {
            lock (state.gate)
                state.progress = value;
        }
    }
}
