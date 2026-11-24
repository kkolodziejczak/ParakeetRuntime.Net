using ParakeetRuntime;

namespace ParakeetRuntime.Tests;

public sealed class MicrophoneRecordingTests
{
    private static readonly TimeSpan TenMinutes = TimeSpan.FromMinutes(10);
    private static readonly TimeSpan Patience = TimeSpan.FromSeconds(5);

    [Fact]
    public void Stop_returns_every_packet_in_order()
    {
        using var recording = new MicrophoneRecording(() => new Packets([1f, 2f], [3f]), TenMinutes, Patience);

        Assert.Equal([1f, 2f, 3f], recording.Stop());
        Assert.False(recording.IsRecording);
    }

    [Fact]
    public void Stopping_twice_returns_the_same_audio()
    {
        using var recording = new MicrophoneRecording(() => new Packets([1f, 2f]), TenMinutes, Patience);

        Assert.Equal(recording.Stop(), recording.Stop());
    }

    [Fact]
    public void Recording_ends_on_its_own_at_exactly_the_longest_duration()
    {
        var fiveSamples = TimeSpan.FromSeconds(5 / 16_000.0);
        using var recording = new MicrophoneRecording(() => new Packets([1f, 2f, 3f], [4f, 5f, 6f], [7f]), fiveSamples, Patience);

        WaitUntil(() => !recording.IsRecording);

        Assert.Equal(fiveSamples, recording.Duration);
        Assert.Equal([1f, 2f, 3f, 4f, 5f], recording.Stop());
    }

    [Fact]
    public void Level_is_the_loudness_of_the_latest_packet()
    {
        using var recording = new MicrophoneRecording(() => new Packets([0.1f], [0.5f, -0.5f]), TenMinutes, Patience);

        WaitUntil(() => recording.Duration == TimeSpan.FromSeconds(3 / 16_000.0));

        Assert.Equal(0.5f, recording.Level, 0.0001f);
    }

    [Fact]
    public void A_microphone_that_cannot_start_throws_its_reason_from_StartRecording()
    {
        var failure = new IOException("Windows has no microphone to record from.");

        var thrown = Assert.Throws<IOException>(() => new MicrophoneRecording(() => throw failure, TenMinutes, Patience));

        Assert.Same(failure, thrown);
    }

    [Fact]
    public void A_microphone_that_does_not_start_in_time_is_reported_without_waiting_longer()
    {
        using var never = new ManualResetEventSlim();

        var thrown = Assert.Throws<IOException>(() => new MicrophoneRecording(() => { never.Wait(); return new Packets(); }, TenMinutes, TimeSpan.FromMilliseconds(200)));

        Assert.Contains("did not start", thrown.Message);
        never.Set();
    }

    [Fact]
    public void A_microphone_that_fails_while_recording_ends_the_recording_and_Stop_says_why()
    {
        using var recording = new MicrophoneRecording(() => new Packets([1f]) { ThenFail = new IOException("The microphone was disconnected.") }, TenMinutes, Patience);

        WaitUntil(() => !recording.IsRecording);

        var thrown = Assert.Throws<IOException>(() => recording.Stop());
        Assert.Contains("The microphone was disconnected.", thrown.Message);
    }

    [Fact]
    public void A_microphone_that_hangs_makes_Stop_fail_instead_of_waiting_for_ever()
    {
        using var released = new ManualResetEventSlim();
        var recording = new MicrophoneRecording(() => new Packets([1f]) { ThenHangUntil = released }, TenMinutes, TimeSpan.FromMilliseconds(200));

        var thrown = Assert.Throws<IOException>(() => recording.Stop());

        Assert.Contains("did not stop", thrown.Message);
        released.Set();
    }

    [Fact]
    public void A_disposed_recording_has_dropped_its_audio()
    {
        var packets = new Packets([0.5f, 0.25f]);
        var recording = new MicrophoneRecording(() => packets, TenMinutes, Patience);
        WaitUntil(() => recording.Duration > TimeSpan.Zero);

        recording.Dispose();
        recording.Dispose();

        Assert.Throws<ObjectDisposedException>(() => recording.Stop());
        Assert.True(packets.Disposed);
        Assert.All(packets.Handed, packet => Assert.All(packet, sample => Assert.Equal(0f, sample)));
    }

    [Fact]
    public void Audio_from_a_microphone_that_opens_after_the_start_timeout_is_cleared_and_dropped()
    {
        using var opening = new ManualResetEventSlim();
        var packets = new Packets([0.5f, 0.25f]);
        Assert.Throws<IOException>(() => new MicrophoneRecording(() => { opening.Wait(); return packets; }, TenMinutes, TimeSpan.FromMilliseconds(100)));

        opening.Set();
        WaitUntil(() => packets.Disposed);

        Assert.All(packets.Handed, packet => Assert.All(packet, sample => Assert.Equal(0f, sample)));
    }

    [Fact]
    public void A_recording_that_reached_its_longest_duration_is_not_recording_while_the_microphone_closes()
    {
        using var closing = new ManualResetEventSlim();
        using var recording = new MicrophoneRecording(() => new Packets([1f, 2f]) { DisposeWaitsFor = closing }, TimeSpan.FromSeconds(1 / 16_000.0), Patience);

        WaitUntil(() => !recording.IsRecording);

        closing.Set();
        Assert.Equal([1f], recording.Stop());
    }

    [Fact]
    public void The_longest_duration_holds_for_very_long_values()
    {
        using var recording = new MicrophoneRecording(() => new Packets([1f, 2f]), TimeSpan.FromDays(668), Patience);

        Assert.Equal([1f, 2f], recording.Stop());
    }

    [Fact]
    public void Level_stays_within_0_and_1_for_audio_beyond_full_scale()
    {
        using var recording = new MicrophoneRecording(() => new Packets([2f, -2f]), TenMinutes, Patience);

        WaitUntil(() => recording.Duration > TimeSpan.Zero);

        Assert.Equal(1f, recording.Level);
    }

    [Fact]
    public void Disposing_again_after_a_microphone_hung_returns_at_once()
    {
        using var released = new ManualResetEventSlim();
        var recording = new MicrophoneRecording(() => new Packets([1f]) { ThenHangUntil = released }, TenMinutes, TimeSpan.FromMilliseconds(300));
        recording.Dispose();

        var clock = System.Diagnostics.Stopwatch.StartNew();
        recording.Dispose();

        Assert.True(clock.Elapsed < TimeSpan.FromMilliseconds(150), $"The second Dispose took {clock.Elapsed.TotalMilliseconds:0} ms.");
        released.Set();
    }

    private static void WaitUntil(Func<bool> condition)
    {
        var deadline = DateTime.UtcNow + Patience;
        while (!condition())
        {
            Assert.True(DateTime.UtcNow < deadline, "The condition did not become true in time.");
            Thread.Sleep(5);
        }
    }

    private sealed class Packets(params float[][] packets) : IMicrophonePackets
    {
        private int next;
        public Exception? ThenFail { get; init; }
        public ManualResetEventSlim? ThenHangUntil { get; init; }
        public List<float[]> Handed { get; } = [];
        public bool Disposed { get; private set; }

        public float[]? NextPacket()
        {
            if (next < packets.Length)
            {
                var packet = packets[next++];
                Handed.Add(packet);
                return packet;
            }
            ThenHangUntil?.Wait();
            return ThenFail is null ? null : throw ThenFail;
        }

        public ManualResetEventSlim? DisposeWaitsFor { get; init; }

        public void Dispose()
        {
            DisposeWaitsFor?.Wait();
            Disposed = true;
        }
    }
}
