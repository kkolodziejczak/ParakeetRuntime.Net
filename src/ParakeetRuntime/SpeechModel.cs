using System.IO.Abstractions;
using System.Security.Cryptography;

namespace ParakeetRuntime;

/// <summary>
/// Where a model file comes from and what it must be, byte for byte.
/// </summary>
/// <param name="Source">The URL to download the file from.</param>
/// <param name="FileName">The file's name in the model folder.</param>
/// <param name="Bytes">The file's exact size in bytes.</param>
/// <param name="Sha256">The file's SHA-256 hash, as hexadecimal text in either case.</param>
public sealed record SpeechModelFile(Uri Source, string FileName, long Bytes, string Sha256);

/// <summary>
/// A local copy of a speech model, pinned to one exact file. A file counts as installed only after
/// its size and SHA-256 have been checked against the pin; later starts compare its size and write
/// time with a small <c>.verified</c> record instead of hashing it again. A file put in the folder by
/// hand, or changed later, is not installed until it is checked again.
/// </summary>
/// <param name="folder">The folder that holds the model file.</param>
/// <param name="pinned">The file the folder must hold.</param>
/// <param name="fileSystem">The file system to use; the real one when omitted.</param>
public sealed class SpeechModel(string folder, SpeechModelFile pinned, IFileSystem? fileSystem = null)
{
    /// <summary>
    /// NVIDIA's own Q8_0 conversion of parakeet-tdt-0.6b-v3 at a fixed revision (713,975,456 bytes),
    /// licensed under CC BY 4.0.
    /// </summary>
    public static readonly SpeechModelFile Parakeet = new(
        new Uri("https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.q8_0.gguf"),
        "parakeet-tdt-0.6b-v3.q8_0.gguf",
        713_975_456,
        "e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e");

    private readonly IFileSystem fs = fileSystem ?? new FileSystem();

    /// <summary>
    /// The <see cref="Parakeet"/> model in <c>%LOCALAPPDATA%\ParakeetRuntime\models</c>.
    /// </summary>
    /// <param name="fileSystem">The file system to use; the real one when omitted.</param>
    public static SpeechModel Default(IFileSystem? fileSystem = null) => new(
        System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ParakeetRuntime", "models"), Parakeet, fileSystem);

    /// <summary>The folder that holds the model file.</summary>
    public string Folder { get; } = folder;

    /// <summary>The file the folder must hold.</summary>
    public SpeechModelFile Pinned { get; } = pinned;

    /// <summary>The full path of the model file, whether or not it is installed.</summary>
    public string Path => fs.Path.Combine(Folder, Pinned.FileName);

    private string VerifiedPath => Path + ".verified";

    /// <summary>
    /// How long a download or install may receive no data before it stops with a
    /// <see cref="TimeoutException"/>. A proxy can hold a dead connection open forever. Defaults to 60 seconds.
    /// </summary>
    public TimeSpan StallTimeout { get; init; } = TimeSpan.FromSeconds(60);

    /// <summary>
    /// Whether the pinned file is in the folder and has been checked. False while another process
    /// holds the file or its record, or while one of them is being deleted.
    /// </summary>
    public bool IsInstalled
    {
        get
        {
            try
            {
                return fs.FileInfo.New(Path) is { Exists: true } file && file.Length == Pinned.Bytes
                    && fs.File.Exists(VerifiedPath) && fs.File.ReadAllText(VerifiedPath) == VerifiedRecord(file);
            }
            catch (Exception exception) when (IsHeldOrGoneMeanwhile(exception))
            {
                return false;
            }
        }
    }

    private static bool IsHeldOrGoneMeanwhile(Exception exception) => exception is IOException or UnauthorizedAccessException;

    private string VerifiedRecord(IFileInfo file) => $"{Pinned.Sha256.ToLowerInvariant()} {file.Length} {file.LastWriteTimeUtc.Ticks}";

    private void RecordVerified() => fs.File.WriteAllText(VerifiedPath, VerifiedRecord(fs.FileInfo.New(Path)));

    /// <summary>
    /// Downloads the pinned file from its source and installs it. Nothing is left under the final
    /// name unless the download matches the pin.
    /// </summary>
    /// <param name="http">The client to download with.</param>
    /// <param name="progress">Receives the fraction of the file received, from 0 to 1; may be null.</param>
    /// <param name="cancellationToken">Stops the download and keeps nothing.</param>
    /// <exception cref="HttpRequestException">The server could not be reached or did not return the file.</exception>
    /// <exception cref="InvalidDataException">The downloaded file does not match the pinned size and SHA-256.</exception>
    /// <exception cref="TimeoutException">No data arrived for <see cref="StallTimeout"/>.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled.</exception>
    public async Task DownloadAsync(HttpClient http, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        using var response = await http.GetAsync(Pinned.Source, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();
        await using var source = await response.Content.ReadAsStreamAsync(cancellationToken);
        await InstallViaPartFileAsync(source, progress, cancellationToken);
    }

    /// <summary>
    /// Installs the pinned file from a local copy, for a computer that cannot reach the download.
    /// When <paramref name="sourcePath"/> is already <see cref="Path"/>, the file is checked where it
    /// is and deleted if it does not match.
    /// </summary>
    /// <param name="sourcePath">The local copy of the model file.</param>
    /// <param name="progress">Receives the fraction of the file read, from 0 to 1; may be null.</param>
    /// <param name="cancellationToken">Stops the install and keeps nothing.</param>
    /// <exception cref="InvalidDataException">The file does not match the pinned size and SHA-256.</exception>
    /// <exception cref="IOException">The file could not be read.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled.</exception>
    public async Task InstallFromFileAsync(string sourcePath, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        if (fs.Path.GetFullPath(sourcePath).Equals(fs.Path.GetFullPath(Path), StringComparison.OrdinalIgnoreCase))
        {
            await CheckFileAlreadyAtFinalPathAsync(progress, cancellationToken);
            return;
        }
        await using var source = fs.FileStream.New(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        await InstallViaPartFileAsync(source, progress, cancellationToken);
    }

    private async Task CheckFileAlreadyAtFinalPathAsync(IProgress<double>? progress, CancellationToken cancellationToken)
    {
        bool matches;
        await using (var file = fs.FileStream.New(Path, FileMode.Open, FileAccess.Read, FileShare.Read))
            matches = await MatchesPinAsync(file, copyTo: null, progress, cancellationToken);
        if (!matches)
        {
            fs.File.Delete(Path);
            throw new InvalidDataException("The model file does not match NVIDIA's pinned size and SHA-256; it may be damaged or a different file.");
        }
        RecordVerified();
    }

    private async Task InstallViaPartFileAsync(Stream source, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        fs.Directory.CreateDirectory(Folder);
        DeleteLeftoverPartsOlderThanAnHour();
        var partPathUniqueToThisInstall = $"{Path}.{Guid.NewGuid():N}.part";
        try
        {
            bool matches;
            await using (var part = fs.FileStream.New(partPathUniqueToThisInstall, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                matches = await MatchesPinAsync(source, part, progress, cancellationToken);
                part.Flush(flushToDisk: true);
            }
            if (!matches)
                throw new InvalidDataException("The model file does not match NVIDIA's pinned size and SHA-256; it may be damaged or a different file.");
            fs.File.Move(partPathUniqueToThisInstall, Path, overwrite: true);
            RecordVerified();
        }
        finally
        {
            DeleteQuietly(partPathUniqueToThisInstall);
        }
    }

    private async Task<bool> MatchesPinAsync(Stream source, Stream? copyTo, IProgress<double>? progress, CancellationToken cancellationToken)
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
            if (copyTo is not null)
                await copyTo.WriteAsync(buffer.AsMemory(0, count), cancellationToken);
            progress?.Report((double)read / Pinned.Bytes);
        }
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

    private void DeleteLeftoverPartsOlderThanAnHour()
    {
        foreach (var part in fs.Directory.GetFiles(Folder, Pinned.FileName + ".*part"))
            if (fs.File.GetLastWriteTimeUtc(part) < DateTime.UtcNow.AddHours(-1))
                DeleteQuietly(part);
    }

    private void DeleteQuietly(string path)
    {
        try
        {
            if (fs.File.Exists(path))
                fs.File.Delete(path);
        }
        catch (Exception exception) when (IsHeldOrGoneMeanwhile(exception))
        {
        }
    }
}
