using ParakeetRuntime.Net;

namespace ParakeetRuntime.Net.Tests;

public class Q8GemmTileTests
{
    private const long TwoGigabytes = 2L << 30;
    private const long SixteenGigabytes = 16L << 30;

    [Theory]
    [InlineData(48)]
    [InlineData(96)]
    public void A_small_gpu_takes_the_large_tile_from_48_rows(int rows)
    {
        Assert.Equal(ParakeetGpu.Q8GemmLargeTile, ParakeetGpu.Q8GemmTile(rows, TwoGigabytes));
    }

    [Fact]
    public void A_small_gpu_keeps_the_small_tile_for_a_few_rows()
    {
        Assert.Equal(ParakeetGpu.Q8GemmSmallTile, ParakeetGpu.Q8GemmTile(32, TwoGigabytes));
    }

    [Theory]
    [InlineData(48, ParakeetGpu.Q8GemmSmallTile)]
    [InlineData(96, ParakeetGpu.Q8GemmSmallTile)]
    [InlineData(97, ParakeetGpu.Q8GemmLargeTile)]
    public void A_large_gpu_takes_the_large_tile_above_96_rows(int rows, int tile)
    {
        Assert.Equal(tile, ParakeetGpu.Q8GemmTile(rows, SixteenGigabytes));
    }
}
