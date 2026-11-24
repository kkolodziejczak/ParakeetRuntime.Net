namespace ParakeetRuntime;

/// <summary>
/// A recording from <see cref="Microphone.StartRecording"/>, kept in memory as 16 kHz mono samples.
/// </summary>
public sealed class MicrophoneRecording : IDisposable
{
    private static readonly TimeSpan PollInterval = TimeSpan.FromMilliseconds(20);

    private readonly Thread captureThread;
    private readonly ManualResetEventSlim stopRequested = new();
    private readonly TaskCompletionSource started = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly Lock gate = new();
    private readonly List<float[]> packets = [];
    private readonly long longestInSamples;
    private readonly TimeSpan patience;
    private long recordedSamples;
    private float levelOfTheLatestPacket;
    private Exception? failure;
    private bool capturing = true;
    private bool disposed;

    internal MicrophoneRecording(Func<IMicrophonePackets> open, TimeSpan longest, TimeSpan patience)
    {
        longestInSamples = (long)((Int128)longest.Ticks * Wav.SampleRate / TimeSpan.TicksPerSecond);
        this.patience = patience;
        captureThread = new Thread(() => CaptureUntilStoppedOrFull(open)) { IsBackground = true, Name = "ParakeetRuntime microphone" };
        captureThread.Start();
        try
        {
            started.Task.WaitAsync(patience).GetAwaiter().GetResult();
        }
        catch (TimeoutException)
        {
            DropTheAudio();
            stopRequested.Set();
            throw new IOException($"The microphone did not start within {patience.TotalSeconds:0.#} seconds.");
        }
    }

    /// <summary>
    /// How much audio has been kept so far.
    /// </summary>
    public TimeSpan Duration => TimeSpan.FromTicks(Interlocked.Read(ref recordedSamples) * TimeSpan.TicksPerSecond / Wav.SampleRate);

    /// <summary>
    /// The loudness (RMS) of the latest packet, from 0 for silence to 1 for full scale, for a level meter.
    /// </summary>
    public float Level => Volatile.Read(ref levelOfTheLatestPacket);

    /// <summary>
    /// Whether the microphone is still being recorded. False after <see cref="Stop"/>, after the longest duration,
    /// and after the microphone failed.
    /// </summary>
    public bool IsRecording => Volatile.Read(ref capturing);

    /// <summary>
    /// Stops recording and returns everything kept. Calling it again returns the same audio.
    /// </summary>
    /// <returns>16 kHz mono samples.</returns>
    /// <exception cref="IOException">The microphone failed during the recording, for example because it was unplugged, or it did not stop within 5 seconds.</exception>
    /// <exception cref="ObjectDisposedException">The recording has been disposed, and its audio dropped.</exception>
    public float[] Stop()
    {
        lock (gate)
            ObjectDisposedException.ThrowIf(disposed, this);
        stopRequested.Set();
        if (!captureThread.Join(patience))
            throw new IOException($"The microphone did not stop within {patience.TotalSeconds:0.#} seconds.");
        if (failure is not null)
            throw new IOException($"The microphone stopped working: {failure.Message}", failure);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return [.. packets.SelectMany(packet => packet)];
        }
    }

    /// <summary>
    /// Stops recording, if it still runs, and clears and drops the audio. Calling it again does nothing.
    /// </summary>
    public void Dispose()
    {
        lock (gate)
            if (disposed)
                return;
        stopRequested.Set();
        captureThread.Join(patience);
        DropTheAudio();
    }

    private void DropTheAudio()
    {
        lock (gate)
        {
            foreach (var packet in packets)
                Array.Clear(packet);
            packets.Clear();
            disposed = true;
        }
    }

    private void CaptureUntilStoppedOrFull(Func<IMicrophonePackets> open)
    {
        IMicrophonePackets? source = null;
        try
        {
            source = open();
            started.TrySetResult();
            do
                Keep(source);
            while (!IsFull() && !stopRequested.Wait(PollInterval));
            Keep(source);
        }
        catch (Exception exception)
        {
            failure = exception;
            started.TrySetException(exception);
        }
        finally
        {
            Volatile.Write(ref capturing, false);
            source?.Dispose();
        }
    }

    private bool IsFull() => Interlocked.Read(ref recordedSamples) >= longestInSamples;

    private void Keep(IMicrophonePackets source)
    {
        while (!IsFull() && source.NextPacket() is { } packet)
        {
            var room = longestInSamples - Interlocked.Read(ref recordedSamples);
            var kept = packet.Length <= room ? packet : packet[..(int)room];
            if (kept != packet)
                Array.Clear(packet);
            Volatile.Write(ref levelOfTheLatestPacket, Loudness(kept));
            lock (gate)
            {
                if (disposed)
                {
                    Array.Clear(kept);
                    return;
                }
                packets.Add(kept);
            }
            Interlocked.Add(ref recordedSamples, kept.Length);
        }
    }

    private static float Loudness(float[] samples)
    {
        if (samples.Length == 0)
            return 0;
        double sumOfSquares = 0;
        foreach (var sample in samples)
            sumOfSquares += sample * sample;
        return (float)Math.Min(1, Math.Sqrt(sumOfSquares / samples.Length));
    }
}
