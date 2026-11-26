using System.IO.Abstractions.TestingHelpers;
using System.Net;
using System.Security.Cryptography;
using ParakeetRuntime.Net;

namespace ParakeetRuntime.Net.Tests;

public sealed class SpeechModelTests
{
    private const string Folder = @"C:\Users\me\AppData\Local\ParakeetRuntime\models";
    private static readonly byte[] Weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    private static readonly SpeechModelFile Pinned = new(
        new Uri("https://example.test/model.gguf"), "model.gguf", Weights.Length, Convert.ToHexString(SHA256.HashData(Weights)));

    private readonly MockFileSystem fs = new();

    private SpeechModel Model() => new(Folder, Pinned, fs);

    private static HttpClient Serving(byte[] body) => new(new FixedResponse(body));

    [Fact]
    public void Without_a_file_the_model_is_not_installed()
    {
        Assert.False(Model().IsInstalled);
    }

    [Fact]
    public void The_default_model_is_the_pinned_Parakeet_file_in_ParakeetRuntimes_own_folder()
    {
        var model = SpeechModel.Default();

        Assert.EndsWith(@"ParakeetRuntime\models\parakeet-tdt-0.6b-v3.q8_0.gguf", model.Path);
        Assert.DoesNotContain("PrivateType", model.Path);
    }

    [Fact]
    public async Task A_download_that_matches_the_pin_installs_the_model()
    {
        var progress = new List<double>();
        var model = Model();

        await model.DownloadAsync(Serving(Weights), new InlineProgressInOrder(progress.Add), CancellationToken.None);

        Assert.True(model.IsInstalled);
        Assert.Equal(Weights, fs.File.ReadAllBytes(model.Path));
        Assert.Equal(1.0, progress[^1]);
        Assert.DoesNotContain(fs.Directory.GetFiles(Folder), file => file.EndsWith(".part"));
    }

    [Fact]
    public async Task Ensuring_a_missing_model_downloads_and_installs_it()
    {
        var progress = new List<double>();
        var model = Model();

        await model.EnsureInstalledAsync(Serving(Weights), new InlineProgressInOrder(progress.Add), CancellationToken.None);

        Assert.True(model.IsInstalled);
        Assert.Equal(1.0, progress[^1]);
    }

    [Fact]
    public async Task Ensuring_an_installed_model_downloads_nothing_and_reports_it_complete()
    {
        var model = Model();
        await model.DownloadAsync(Serving(Weights), null, CancellationToken.None);
        var progress = new List<double>();

        await model.EnsureInstalledAsync(new HttpClient(new NoNetwork()), new InlineProgressInOrder(progress.Add), CancellationToken.None);

        Assert.True(model.IsInstalled);
        Assert.Equal([1.0], progress);
    }

    [Fact]
    public async Task A_download_with_the_wrong_content_is_rejected_and_leaves_nothing_behind()
    {
        var model = Model();
        byte[] tampered = [.. Weights];
        tampered[0] = 99;

        await Assert.ThrowsAsync<InvalidDataException>(() => model.DownloadAsync(Serving(tampered), null, CancellationToken.None));

        Assert.False(model.IsInstalled);
        Assert.Empty(fs.Directory.GetFiles(Folder));
    }

    [Fact]
    public async Task A_download_of_the_wrong_size_is_rejected()
    {
        var model = Model();

        await Assert.ThrowsAsync<InvalidDataException>(() => model.DownloadAsync(Serving([.. Weights, 11]), null, CancellationToken.None));

        Assert.False(model.IsInstalled);
    }

    [Fact]
    public async Task A_download_left_behind_by_a_crash_is_cleaned_up_by_the_next_one()
    {
        fs.AddFile(fs.Path.Combine(Folder, "model.gguf.part"), new MockFileData([7, 7, 7]) { LastWriteTime = DateTimeOffset.Now.AddDays(-2) });
        var model = Model();

        await model.DownloadAsync(Serving(Weights), null, CancellationToken.None);

        Assert.True(model.IsInstalled);
        Assert.DoesNotContain(fs.Directory.GetFiles(Folder), file => file.EndsWith(".part"));
    }

    [Fact]
    public async Task A_copied_file_that_matches_the_pin_installs_the_model()
    {
        fs.AddFile(@"D:\usb\model.gguf", new MockFileData(Weights));
        var model = Model();

        await model.InstallFromFileAsync(@"D:\usb\model.gguf", null, CancellationToken.None);

        Assert.True(model.IsInstalled);
        Assert.True(fs.File.Exists(@"D:\usb\model.gguf"));
    }

    [Fact]
    public async Task A_copied_file_that_does_not_match_the_pin_is_rejected()
    {
        fs.AddFile(@"D:\usb\other.gguf", new MockFileData(new byte[Weights.Length]));
        var model = Model();

        await Assert.ThrowsAsync<InvalidDataException>(() => model.InstallFromFileAsync(@"D:\usb\other.gguf", null, CancellationToken.None));

        Assert.False(model.IsInstalled);
    }

    [Fact]
    public void A_file_put_in_place_by_hand_is_not_trusted_until_it_is_checked()
    {
        var model = Model();
        fs.AddFile(model.Path, new MockFileData(new byte[Weights.Length]));

        Assert.False(model.IsInstalled);
    }

    [Fact]
    public async Task A_checked_file_that_is_later_replaced_is_no_longer_installed()
    {
        var model = Model();
        await model.DownloadAsync(Serving(Weights), null, CancellationToken.None);

        fs.File.WriteAllBytes(model.Path, new byte[Weights.Length]);
        fs.File.SetLastWriteTimeUtc(model.Path, DateTime.UtcNow.AddMinutes(5));

        Assert.False(model.IsInstalled);
    }

    [Fact]
    public async Task A_part_file_still_being_written_by_another_install_is_left_alone()
    {
        var other = fs.Path.Combine(Folder, "model.gguf.0123.part");
        fs.AddFile(other, new MockFileData([1, 2]) { LastWriteTime = DateTimeOffset.Now });

        await Model().DownloadAsync(Serving(Weights), null, CancellationToken.None);

        Assert.True(fs.File.Exists(other));
    }

    [Fact]
    public async Task Picking_the_file_already_in_place_checks_it_and_keeps_it()
    {
        var model = Model();
        fs.AddFile(model.Path, new MockFileData(Weights));

        await model.InstallFromFileAsync(model.Path, null, CancellationToken.None);

        Assert.True(model.IsInstalled);
        Assert.Equal(Weights, fs.File.ReadAllBytes(model.Path));
    }

    [Fact]
    public async Task A_wrong_file_left_in_place_is_rejected_and_removed()
    {
        var model = Model();
        fs.AddFile(model.Path, new MockFileData(new byte[Weights.Length]));

        await Assert.ThrowsAsync<InvalidDataException>(() => model.InstallFromFileAsync(model.Path, null, CancellationToken.None));

        Assert.False(model.IsInstalled);
        Assert.False(fs.File.Exists(model.Path));
    }

    [Fact]
    public async Task A_cancel_that_lands_after_the_last_byte_still_installs_nothing()
    {
        using var cancel = new CancellationTokenSource();
        var model = Model();
        var http = new HttpClient(new StreamResponse(new StreamWhoseReadsFinishDespiteCancel(Weights)));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            model.DownloadAsync(http, new InlineProgressInOrder(done => { if (done >= 1) cancel.Cancel(); }), cancel.Token));

        Assert.False(model.IsInstalled);
    }

    private sealed class StreamResponse(Stream body) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StreamContent(body) });
    }

    private sealed class StreamWhoseReadsFinishDespiteCancel(byte[] data) : MemoryStream(data)
    {
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) =>
            base.ReadAsync(buffer, CancellationToken.None);

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            base.ReadAsync(buffer, offset, count, CancellationToken.None);
    }

    private sealed class FixedResponse(byte[] body) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(body) });
    }

    private sealed class NoNetwork : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            throw new InvalidOperationException("The test expected no download.");
    }

    private sealed class InlineProgressInOrder(Action<double> report) : IProgress<double>
    {
        public void Report(double value) => report(value);
    }
}
