using ParakeetRuntime;

namespace ParakeetRuntime.Tests;

// How many GPU decoder steps go into one submit: long enough to save CPU round trips, short enough
// that a slow GPU never runs one submit anywhere near Windows' 2 s GPU timeout.
public sealed class DecodeBatchTests
{
    private const double Budget = 350;

    [Fact]
    public void Before_the_step_time_is_known_a_batch_stays_at_64_steps()
    {
        Assert.Equal(64, ParakeetGpu.DecodeBatchSteps(remainingFrames: 1500, Budget, msPerStep: 0));
    }

    [Fact]
    public void A_fast_GPU_gets_long_batches_on_long_audio()
    {
        Assert.Equal(512, ParakeetGpu.DecodeBatchSteps(remainingFrames: 1500, Budget, msPerStep: 0.05));
    }

    [Fact]
    public void A_slow_GPU_gets_only_as_many_steps_as_fit_the_budget()
    {
        Assert.Equal(70, ParakeetGpu.DecodeBatchSteps(remainingFrames: 1500, Budget, msPerStep: 5));
    }

    [Fact]
    public void A_very_slow_GPU_still_makes_progress()
    {
        var steps = ParakeetGpu.DecodeBatchSteps(remainingFrames: 1500, Budget, msPerStep: 200);

        Assert.InRange(steps, 1, 8);
    }

    [Fact]
    public void Short_audio_needs_short_batches()
    {
        Assert.True(ParakeetGpu.DecodeBatchSteps(remainingFrames: 40, Budget, msPerStep: 0.05) < 64);
    }

    [Fact]
    public void Without_budget_left_after_the_encoder_a_batch_takes_the_minimum()
    {
        Assert.Equal(ParakeetGpu.MinDecodeBatchSteps, ParakeetGpu.DecodeBatchSteps(remainingFrames: 1500, budgetMs: 0, msPerStep: 0.05));
    }
}
