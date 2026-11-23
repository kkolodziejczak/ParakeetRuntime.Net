using System.IO.Abstractions;
using System.Security.Cryptography;

namespace ParakeetRuntime;

// Where a model file comes from and what it must be, byte for byte.
public sealed record SpeechModelFile(Uri Source, string FileName, long Bytes, string Sha256);

// LocalBuddy's own copy of the Parakeet model, in %LOCALAPPDATA%\LocalBuddy\models. After a file's
// size and SHA-256 match the pin, a small ".verified" record notes its size and write time, so a
// start checks those instead of hashing 714 MB. A file put there by hand, or changed later, is not
// installed until it is checked again.
public sealed class SpeechModel(string folder, SpeechModelFile pinned, IFileSystem? fileSystem = null)
{
    // NVIDIA's own Q8_0 conversion at a fixed revision; licence CC BY 4.0.
    public static readonly SpeechModelFile Parakeet = new(
        new Uri("https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.q8_0.gguf"),
        "parakeet-tdt-0.6b-v3.q8_0.gguf",
        713_975_456,
        "e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e");

    private readonly IFileSystem fs = fileSystem ?? new FileSystem();

    public static SpeechModel Default(IFileSystem? fileSystem = null) => new(
        System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ParakeetRuntime", "models"), Parakeet, fileSystem);

    public string Folder { get; } = folder;

    public SpeechModelFile Pinned { get; } = pinned;

    public string Path => fs.Path.Combine(Folder, Pinned.FileName);

    private string VerifiedPath => Path + ".verified";

    // A download that sends nothing for this long stops on its own; a work proxy can hold a dead
    // connection open forever.
    public TimeSpan StallTimeout { get; init; } = TimeSpan.FromSeconds(60);

    public bool IsInstalled
    {
        get
        {
            try
            {
                return fs.FileInfo.New(Path) is { Exists: true } file && file.Length == Pinned.Bytes
                    && fs.File.Exists(VerifiedPath) && fs.File.ReadAllText(VerifiedPath) == VerifiedRecord(file);
            }
            // A scanner holding the record, or a file deleted in between: not usable right now.
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                return false;
            }
        }
    }

    private string VerifiedRecord(IFileInfo file) => $"{Pinned.Sha256.ToLowerInvariant()} {file.Length} {file.LastWriteTimeUtc.Ticks}";

    private void RecordVerified() => fs.File.WriteAllText(VerifiedPath, VerifiedRecord(fs.FileInfo.New(Path)));

    // Progress is the fraction of the file received, 0 to 1.
    public async Task DownloadAsync(HttpClient http, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        using var response = await http.GetAsync(Pinned.Source, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();
        await using var source = await response.Content.ReadAsStreamAsync(cancellationToken);
        await InstallAsync(source, progress, cancellationToken);
    }

    // For a computer that cannot reach the download: the user copies the file in by hand.
    public async Task InstallFromFileAsync(string sourcePath, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        if (fs.Path.GetFullPath(sourcePath).Equals(fs.Path.GetFullPath(Path), StringComparison.OrdinalIgnoreCase))
        {
            await CheckInPlaceAsync(progress, cancellationToken);
            return;
        }
        await using var source = fs.FileStream.New(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        await InstallAsync(source, progress, cancellationToken);
    }

    // A file the user put under the final name by hand: hashed where it is, and removed if wrong.
    private async Task CheckInPlaceAsync(IProgress<double>? progress, CancellationToken cancellationToken)
    {
        bool matches;
        await using (var file = fs.FileStream.New(Path, FileMode.Open, FileAccess.Read, FileShare.Read))
            matches = await MatchesPinAsync(file, part: null, progress, cancellationToken);
        if (!matches)
        {
            fs.File.Delete(Path);
            throw new InvalidDataException("The model file does not match NVIDIA's pinned size and SHA-256; it may be damaged or a different file.");
        }
        RecordVerified();
    }

    // Writes to a .part file while hashing, and renames it only when it matches the pin. A crash or
    // a failed check leaves no file under the final name; the next install starts the .part over.
    private async Task InstallAsync(Stream source, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        fs.Directory.CreateDirectory(Folder);
        DeleteLeftoverParts();
        // Its own name, so a second LocalBuddy installing at the same time cannot touch this one.
        var partPath = $"{Path}.{Guid.NewGuid():N}.part";
        try
        {
            bool matches;
            await using (var part = fs.FileStream.New(partPath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                matches = await MatchesPinAsync(source, part, progress, cancellationToken);
                // On disk before the rename, so a power cut cannot leave a short file under the final name.
                part.Flush(flushToDisk: true);
            }
            if (!matches)
                throw new InvalidDataException("The model file does not match NVIDIA's pinned size and SHA-256; it may be damaged or a different file.");
            fs.File.Move(partPath, Path, overwrite: true);
            RecordVerified();
        }
        finally
        {
            DeleteQuietly(partPath);
        }
    }

    // Reads the whole source, copying it to `part` when given. Stops early once it is too long.
    private async Task<bool> MatchesPinAsync(Stream source, Stream? part, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buffer = new byte[1 << 20];
        long read = 0;
        int count;
        while ((count = await ReadWithinStallTimeoutAsync(source, buffer, cancellationToken)) > 0)
        {
            read += count;
            if (read > Pinned.Bytes)
                return false;
            hash.AppendData(buffer, 0, count);
            if (part is not null)
                await part.WriteAsync(buffer.AsMemory(0, count), cancellationToken);
            progress?.Report((double)read / Pinned.Bytes);
        }
        // A cancel that arrives with the last bytes still means "keep nothing".
        cancellationToken.ThrowIfCancellationRequested();
        return read == Pinned.Bytes && Convert.ToHexString(hash.GetHashAndReset()).Equals(Pinned.Sha256, StringComparison.OrdinalIgnoreCase);
    }

    private async Task<int> ReadWithinStallTimeoutAsync(Stream source, byte[] buffer, CancellationToken cancellationToken)
    {
        using var stall = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        stall.CancelAfter(StallTimeout);
        try
        {
            return await source.ReadAsync(buffer, stall.Token);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException($"No data arrived for {StallTimeout.TotalSeconds:0} seconds.");
        }
    }

    // Parts left by a crash a while ago. A recent one may belong to another LocalBuddy that is
    // still writing or about to rename it, so it stays.
    private void DeleteLeftoverParts()
    {
        foreach (var part in fs.Directory.GetFiles(Folder, Pinned.FileName + ".*part"))
            if (fs.File.GetLastWriteTimeUtc(part) < DateTime.UtcNow.AddHours(-1))
                DeleteQuietly(part);
    }

    // Best effort: a scanner or a second LocalBuddy may still hold it, and that must not hide the
    // real error.
    private void DeleteQuietly(string path)
    {
        try
        {
            if (fs.File.Exists(path))
                fs.File.Delete(path);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
        }
    }
}
