namespace ParakeetRuntime.Net;

/// <summary>
/// A microphone Windows can record from.
/// </summary>
/// <param name="Id">Windows' id for the device, which stays the same while it is connected.</param>
/// <param name="Name">The name Windows shows for it, for example "Microphone (USB Audio Device)".</param>
public sealed record MicrophoneDevice(string Id, string Name);

/// <summary>
/// Records from a microphone into memory, as 16 kHz mono samples ready for <see cref="SpeechEngine.TranscribeAsync"/>.
/// Windows converts the microphone's own format; nothing is written to disk.
/// </summary>
public static class Microphone
{
    private static readonly TimeSpan DefaultLongest = TimeSpan.FromMinutes(10);
    private static readonly TimeSpan StartAndStopPatience = TimeSpan.FromSeconds(5);

    /// <summary>
    /// Lists the microphones that are connected and turned on.
    /// </summary>
    /// <returns>The microphones; empty when there are none.</returns>
    /// <exception cref="IOException">Windows' audio service could not be asked.</exception>
    public static IReadOnlyList<MicrophoneDevice> Devices() => Wasapi.OnItsOwnComThread(Wasapi.ActiveCaptureDevices);

    /// <summary>
    /// Starts recording. Recording runs on its own thread until <see cref="MicrophoneRecording.Stop"/>, until
    /// <paramref name="longest"/> is reached, or until the microphone fails.
    /// </summary>
    /// <param name="device">The microphone, from <see cref="Devices"/>; Windows' default microphone when omitted.</param>
    /// <param name="longest">The most audio to keep; 10 minutes when omitted. Recording ends on its own there.</param>
    /// <returns>The running recording.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="longest"/> is zero or negative.</exception>
    /// <exception cref="IOException">There is no microphone, the chosen one is no longer connected, or it did not start within 5 seconds. The message says which.</exception>
    /// <exception cref="UnauthorizedAccessException">Windows' privacy settings do not let this app use the microphone.</exception>
    public static MicrophoneRecording StartRecording(MicrophoneDevice? device = null, TimeSpan? longest = null)
    {
        var cap = longest ?? DefaultLongest;
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(cap, TimeSpan.Zero, nameof(longest));
        return new MicrophoneRecording(() => Wasapi.Capture.Open(device?.Id), cap, StartAndStopPatience);
    }
}
