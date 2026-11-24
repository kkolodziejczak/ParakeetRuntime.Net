using ParakeetRuntime;

namespace ParakeetRuntime.Tests;

internal sealed class MicrophoneFactAttribute : FactAttribute
{
    public MicrophoneFactAttribute()
    {
        if (!HasMicrophone())
            Skip = "This machine has no microphone.";
    }

    private static bool HasMicrophone()
    {
        try
        {
            return Microphone.Devices().Count > 0;
        }
        catch (IOException)
        {
            return false;
        }
    }
}

public sealed class MicrophoneTests
{
    [MicrophoneFact]
    public void Every_microphone_has_an_id_and_a_name()
    {
        var devices = Microphone.Devices();

        Assert.All(devices, device =>
        {
            Assert.False(string.IsNullOrWhiteSpace(device.Id));
            Assert.False(string.IsNullOrWhiteSpace(device.Name));
        });
        Assert.Equal(devices.Count, devices.Select(device => device.Id).Distinct().Count());
    }

    [MicrophoneFact]
    public void One_second_from_the_default_microphone_is_about_16000_samples_of_finite_audio()
    {
        using var recording = Microphone.StartRecording();
        Thread.Sleep(1000);

        var samples = recording.Stop();

        Assert.InRange(samples.Length, 12_000, 20_000);
        Assert.All(samples, sample => Assert.True(float.IsFinite(sample)));
    }

    [MicrophoneFact]
    public void A_recording_from_a_named_microphone_stops_at_exactly_its_longest_duration()
    {
        using var recording = Microphone.StartRecording(Microphone.Devices()[0], TimeSpan.FromMilliseconds(500));
        var deadline = DateTime.UtcNow.AddSeconds(5);
        while (recording.IsRecording && DateTime.UtcNow < deadline)
            Thread.Sleep(20);

        Assert.False(recording.IsRecording);
        Assert.Equal(8_000, recording.Stop().Length);
    }

    [MicrophoneFact]
    public void A_microphone_that_is_no_longer_connected_is_named_as_such()
    {
        var gone = new MicrophoneDevice(@"{0.0.1.00000000}.{00000000-0000-0000-0000-000000000000}", "Unplugged");

        var thrown = Assert.Throws<IOException>(() => Microphone.StartRecording(gone));

        Assert.Contains("no longer connected", thrown.Message);
    }
}
