using ParakeetRuntime.Net;

namespace ParakeetRuntime.Net.Tests;

public sealed class EncoderLengthTests
{
    [Theory]
    [InlineData(20_480, 16)]
    [InlineData(20_640, 17)]
    [InlineData(320_000, 250)]
    [InlineData(100, 1)]
    public void The_encoder_runs_on_as_many_frames_as_NeMo_counts_from_the_valid_mel_frames(int samples, int expectedFrames)
    {
        int melFrames = samples / 160 + 1;
        int subsampled = melFrames;
        for (int stride2 = 0; stride2 < 3; stride2++)
            subsampled = (subsampled - 1) / 2 + 1;

        Assert.Equal(expectedFrames, ParakeetModel.EncoderFramesOfValidMelFrames(subsampled, melFrames - Mel.ZeroedLastFrames));
    }
}
