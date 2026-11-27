using ParakeetRuntime.Net;

namespace ParakeetRuntime.Net.Tests;

public class Q8GemmTileTests
{
    private const long TwoGigabytes = 2L << 30;
    private const long ThreeGigabytes = 3L << 30;
    private const long SixteenGigabytes = 16L << 30;

    [Theory]
    [InlineData(40, ParakeetGpu.Q8GemmSmallTile)]
    [InlineData(41, ParakeetGpu.Q8GemmLargeTile)]
    [InlineData(96, ParakeetGpu.Q8GemmLargeTile)]
    public void A_small_gpu_takes_the_large_tile_above_40_rows(int rows, int tile)
    {
        Assert.Equal(tile, ParakeetGpu.Q8GemmTile(rows, TwoGigabytes));
        Assert.Equal(tile, ParakeetGpu.Q8GemmTile(rows, ThreeGigabytes - 1));
    }

    [Fact]
    public void A_gpu_with_3_GiB_counts_as_large()
    {
        Assert.Equal(ParakeetGpu.Q8GemmSmallTile, ParakeetGpu.Q8GemmTile(96, ThreeGigabytes));
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
