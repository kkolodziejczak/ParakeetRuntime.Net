using System.Numerics;
using ParakeetRuntime.Net;

namespace ParakeetRuntime.Net.Tests;

public sealed class KernelsTests
{
    public static TheoryData<string> SupportedPathNames()
    {
        var paths = new TheoryData<string>();
        foreach (var path in Enum.GetValues<Int8Path>().Where(Kernels.Supports))
            paths.Add(path.ToString());
        return paths;
    }

    public static TheoryData<bool> QuantizeWithAndWithoutAvx2()
    {
        var paths = new TheoryData<bool> { false };
        if (Kernels.Supports(Int8Path.Avx2))
            paths.Add(true);
        return paths;
    }

    [Theory]
    [MemberData(nameof(QuantizeWithAndWithoutAvx2))]
    public void Activations_are_quantized_per_block_like_ggml_on_x86(bool avx2)
    {
        var random = new Random(7);
        float[] firstBlockWithScaleOneSoHalvesLandOnPointFive = [127f, 0.5f, -0.5f, 1.5f, -2.5f, 126.5f, 0.49999997f, -0.49999997f, 2.4999998f, 3.5f];
        var x = new float[3 * 64];
        for (var i = 0; i < x.Length; i++)
            x[i] = i < firstBlockWithScaleOneSoHalvesLandOnPointFive.Length ? firstBlockWithScaleOneSoHalvesLandOnPointFive[i] : (float)(random.NextDouble() * 6 - 3);

        var q = Kernels.Quantize(x, 3, 64, avx2);

        for (var block = 0; block < 6; block++)
        {
            var values = x.AsSpan(block * 32, 32).ToArray();
            var amax = values.Max(MathF.Abs);
            var expected = values.Select(v => (sbyte)MathF.Round(v * (127f / amax), MidpointRounding.ToEven)).ToArray();
            Assert.Equal((float)(Half)(amax / 127f), q.D[block]);
            Assert.Equal(expected, q.Q.AsSpan(block * 32, 32).ToArray());
            Assert.Equal(expected.Sum(v => v), q.Sums[block]);
        }
        Assert.Equal([127, 0, 0, 2, -2, 126, 0, 0, 2, 4], q.Q[..10].Select(v => (int)v));
    }

    [Theory]
    [MemberData(nameof(QuantizeWithAndWithoutAvx2))]
    public void Activations_quantize_by_the_formula_at_every_loudness(bool avx2)
    {
        const int rows = 5, k = 1024;
        var random = new Random(11);
        var x = new float[rows * k];
        for (var block = 0; block < x.Length / 32; block++)
        {
            var loudness = block switch { 8 => 1e7f, 9 => 1e38f, _ => MathF.Pow(10, random.Next(-7, 4)) };
            var sign = block % 7 == 3 ? -1f : 1f;
            for (var i = 0; i < 32; i++)
                x[block * 32 + i] = block == 10 || block % 11 == 5 && i % 2 == 0 ? -0f : sign * loudness * (float)(random.NextDouble() * (block % 2 == 0 ? 1 : 2) - (sign < 0 ? 0 : 0.5));
        }

        var q = Kernels.Quantize(x, rows, k, avx2);

        for (var block = 0; block < x.Length / 32; block++)
        {
            var values = x.AsSpan(block * 32, 32).ToArray();
            var amax = values.Max(MathF.Abs);
            var scale = (float)(Half)(amax / 127f);
            var expected = values.Select(v => scale == 0 ? (sbyte)0 : (sbyte)MathF.Round(v * (127f / amax), MidpointRounding.ToEven)).ToArray();
            Assert.Equal(scale, q.D[block]);
            Assert.Equal(expected, q.Q.AsSpan(block * 32, 32).ToArray());
            Assert.Equal(expected.Sum(v => v), q.Sums[block]);
        }
    }

    [Theory]
    [MemberData(nameof(SilentBlocks))]
    public void Silent_blocks_quantize_to_zero(float loudest, bool avx2)
    {
        var x = new float[32];
        x[3] = loudest;

        var q = QuantizeOverLouder(x, 1, 32, avx2);

        Assert.Equal(0f, q.D[0]);
        Assert.All(q.Q, v => Assert.Equal(0, v));
        Assert.Equal(0, q.Sums[0]);
    }

    public static TheoryData<float, bool> SilentBlocks()
    {
        var cases = new TheoryData<float, bool>();
        foreach (var avx2 in QuantizeWithAndWithoutAvx2())
            foreach (var loudest in new[] { 0f, 1e-38f, 1e-7f })
                cases.Add(loudest, avx2);
        return cases;
    }

    [Theory]
    [MemberData(nameof(QuantizeWithAndWithoutAvx2))]
    public void Blocks_holding_NaN_or_infinity_quantize_to_zero(bool avx2)
    {
        float[] bad = [float.NaN, -float.NaN, float.PositiveInfinity, float.NegativeInfinity];
        int[] placesInEveryLoadAndLane = [0, 5, 10, 18, 31];
        int badBlocks = bad.Length * placesInEveryLoadAndLane.Length + 1, blocks = badBlocks + 1;
        var x = new float[blocks * 32];
        for (var i = 0; i < x.Length; i++)
            x[i] = (i % 13 - 6) * 0.25f;
        for (var b = 0; b < badBlocks - 1; b++)
            x[b * 32 + placesInEveryLoadAndLane[b % placesInEveryLoadAndLane.Length]] = bad[b / placesInEveryLoadAndLane.Length];
        x[(badBlocks - 1) * 32 + 3] = float.PositiveInfinity;
        x[(badBlocks - 1) * 32 + 20] = float.NaN;

        var q = QuantizeOverLouder(x, 2, x.Length / 2, avx2);

        for (var b = 0; b < badBlocks; b++)
        {
            Assert.Equal(b < badBlocks - 1 && float.IsInfinity(bad[b / placesInEveryLoadAndLane.Length]) ? float.PositiveInfinity : float.NaN, q.D[b]);
            Assert.All(q.Q.AsSpan(b * 32, 32).ToArray(), v => Assert.Equal(0, v));
            Assert.Equal(0, q.Sums[b]);
        }
        var good = x.AsSpan(badBlocks * 32, 32).ToArray();
        var expected = good.Select(v => (sbyte)MathF.Round(v * (127f / 1.5f), MidpointRounding.ToEven)).ToArray();
        Assert.Equal((float)(Half)(1.5f / 127f), q.D[badBlocks]);
        Assert.Equal(expected, q.Q.AsSpan(badBlocks * 32, 32).ToArray());
        Assert.Equal(expected.Sum(v => v), q.Sums[badBlocks]);
    }

    private static Q8Activations QuantizeOverLouder(float[] x, int rows, int k, bool avx2)
    {
        var scratch = new EncoderScratch();
        using var utterance = scratch.BeginUtterance();
        var louder = Kernels.Quantize(Enumerable.Range(0, x.Length).Select(i => i % 7 + 1f).ToArray(), rows, k, avx2);
        var q = Kernels.Quantize(x, rows, k, avx2);
        Assert.Same(louder, q);
        return q;
    }

    [Theory]
    [MemberData(nameof(SupportedPathNames))]
    public void Every_int8_path_multiplies_exactly(string pathName)
    {
        var path = Enum.Parse<Int8Path>(pathName);
        foreach (var (rows, k, outputs) in new[] { (8, 64, 8), (5, 96, 13), (1, 32, 3), (11, 128, 21), (4, 64, 16), (7, 64, 72), (10, 32, 141) })
        foreach (var withBias in new[] { true, false })
        {
            var random = new Random(rows * 1000 + k + outputs);
            var x = Enumerable.Range(0, rows * k).Select(_ => (float)(random.NextDouble() * 4 - 2)).ToArray();
            var w = new Q8Matrix(outputs, k);
            for (var i = 0; i < w.Q.Length; i++)
                w.Q[i] = (sbyte)random.Next(-128, 128);
            for (var i = 0; i < w.D.Length; i++)
                w.D[i] = (float)random.NextDouble() / 100;
            var bias = withBias ? Enumerable.Range(0, outputs).Select(_ => (float)random.NextDouble() - 0.5f).ToArray() : null;
            var residual = Enumerable.Range(0, rows * outputs).Select(_ => (float)random.NextDouble()).ToArray();
            var expected = Product(Kernels.Quantize(x, rows, k), w, bias);
            var weights = Kernels.Pack(w, bias, path);

            var plain = Kernels.Linear(x, rows, weights).ToArray();
            var silu = Kernels.Linear(x, rows, weights, Activation.Silu).ToArray();
            var added = residual.ToArray();
            Kernels.LinearAdd(x, rows, weights, 0.5f, added);

            for (var i = 0; i < expected.Length; i++)
            {
                var tolerance = 1e-4 * Math.Max(1, Math.Abs(expected[i]));
                Assert.Equal(expected[i], plain[i], tolerance);
                Assert.Equal(expected[i] / (1 + Math.Exp(-expected[i])), silu[i], tolerance);
                Assert.Equal(residual[i] + 0.5 * expected[i], added[i], tolerance);
            }
        }
    }

    static double[] Product(Q8Activations xq, Q8Matrix w, float[]? bias)
    {
        int rows = xq.Rows, k = xq.K, outputs = w.Rows;
        var y = new double[rows * outputs];
        for (var m = 0; m < rows; m++)
            for (var n = 0; n < outputs; n++)
            {
                double sum = bias?[n] ?? 0;
                for (var b = 0; b < k / 32; b++)
                {
                    long dot = 0;
                    for (var i = b * 32; i < b * 32 + 32; i++)
                        dot += xq.Q[m * k + i] * w.Q[n * k + i];
                    sum += dot * (double)xq.D[m * (k / 32) + b] * w.D[n * (k / 32) + b];
                }
                y[m * outputs + n] = sum;
            }
        return y;
    }

    [Theory]
    [InlineData(4, 32, 16)]
    [InlineData(7, 96, 24)]
    [InlineData(1, 64, 8)]
    public void The_float_product_with_relu_matches_the_formula(int rows, int outputs, int k)
    {
        var random = new Random(rows + outputs + k);
        var x = Enumerable.Range(0, rows * k).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var w = Enumerable.Range(0, outputs * k).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var bias = Enumerable.Range(0, outputs).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
        var transposed = new float[k * outputs];
        for (var n = 0; n < outputs; n++)
            for (var i = 0; i < k; i++)
                transposed[i * outputs + n] = w[n * k + i];

        var y = Kernels.GemmRelu(x, rows, transposed, outputs, k, bias);

        for (var m = 0; m < rows; m++)
            for (var n = 0; n < outputs; n++)
            {
                double sum = bias[n];
                for (var i = 0; i < k; i++)
                    sum += (double)x[m * k + i] * w[n * k + i];
                Assert.Equal(Math.Max(0, sum), y[m * outputs + n], 1e-5);
            }
    }

    [Fact]
    public void The_float_product_refuses_a_width_that_is_not_whole_vector_pairs()
    {
        int outputs = 3 * Vector<float>.Count, k = 4;

        Assert.Throws<ArgumentException>(() => Kernels.GemmRelu(new float[k], 1, new float[k * outputs], outputs, k, new float[outputs]));
    }

    [Theory]
    [InlineData(8, 0)]
    [InlineData(7, 0)]
    [InlineData(5, 4)]
    [InlineData(1, 0)]
    [InlineData(37, 0)]
    [InlineData(37, 6)]
    [InlineData(70, 0)]
    public void Relative_attention_matches_the_formula(int T, int first)
    {
        int d = 64, heads = 2, dk = d / heads, P = 2 * (T + first) - 1;
        var random = new Random(T * 10 + first);
        float[] Random(int n) => Enumerable.Range(0, n).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var qkv = Random(T * 3 * d); var u = Random(d); var v = Random(d); var p = Random(P * d);
        var (positions, stride) = Kernels.TransposePositions(p, P, d);
        var ctx = new float[T * d];

        Kernels.RelativeAttention(qkv, T, d, heads, u, v, positions, stride, first, ctx);

        for (var h = 0; h < heads; h++)
            for (var i = 0; i < T; i++)
            {
                var scores = new double[T];
                for (var j = 0; j < T; j++)
                    for (var e = 0; e < dk; e++)
                    {
                        double q = qkv[i * 3 * d + h * dk + e];
                        scores[j] += ((q + u[h * dk + e]) * qkv[j * 3 * d + d + h * dk + e]
                                      + (q + v[h * dk + e]) * p[(first + T - 1 - i + j) * d + h * dk + e]) / Math.Sqrt(dk);
                    }
                var weights = scores.Select(s => Math.Exp(s - scores.Max())).ToArray();
                for (var e = 0; e < dk; e++)
                {
                    double expected = 0;
                    for (var j = 0; j < T; j++)
                        expected += weights[j] / weights.Sum() * qkv[j * 3 * d + 2 * d + h * dk + e];
                    Assert.Equal(expected, ctx[i * d + h * dk + e], 1e-5);
                }
            }
    }

    [FmaFact]
    public void Relative_attention_keeps_its_exact_float_sums()
    {
        var fingerprints = new[] { (1, 0), (7, 0), (37, 6), (70, 0), (300, 0) }.Select(shape =>
        {
            var (T, first) = shape;
            int d = 256, heads = 2, P = 2 * (T + first) - 1;
            var random = new Random(T * 10 + first);
            float[] Random(int n) => Enumerable.Range(0, n).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            var qkv = Random(T * 3 * d); var u = Random(d); var v = Random(d); var p = Random(P * d);
            var (positions, stride) = Kernels.TransposePositions(p, P, d);
            var ctx = new float[T * d];
            Kernels.RelativeAttention(qkv, T, d, heads, u, v, positions, stride, first, ctx);
            return ctx.Aggregate(14695981039346656037UL, (hash, value) => (hash ^ BitConverter.SingleToUInt32Bits(value)) * 1099511628211UL).ToString("x16");
        });

        Assert.Equal(["7bf0c7c2416f2297", "b878fbfc15f062a5", "291ec174ae74ef02", "9f1b503305969ac8", "f1df3a9dcb34f8a2"], fingerprints);
    }

    [Fact]
    public void Layer_norm_matches_the_textbook_formula()
    {
        var random = new Random(3);
        int rows = 3, d = 1024;
        var x = Enumerable.Range(0, rows * d).Select(_ => (float)(random.NextDouble() * 8 - 3)).ToArray();
        var gain = Enumerable.Range(0, d).Select(_ => (float)random.NextDouble() + 0.5f).ToArray();
        var shift = Enumerable.Range(0, d).Select(_ => (float)random.NextDouble() - 0.5f).ToArray();

        var y = Kernels.LayerNorm(x, rows, d, gain, shift);

        for (var r = 0; r < rows; r++)
        {
            var row = x.Skip(r * d).Take(d).Select(v => (double)v).ToArray();
            double mean = row.Average(), variance = row.Select(v => (v - mean) * (v - mean)).Average();
            for (var i = 0; i < d; i++)
                Assert.Equal((row[i] - mean) / Math.Sqrt(variance + 1e-5) * gain[i] + shift[i], y[r * d + i], 1e-5);
        }
    }
}

internal sealed class FmaFactAttribute : FactAttribute
{
    public FmaFactAttribute()
    {
        if (Vector<float>.Count != 8 || !System.Runtime.Intrinsics.X86.Fma.IsSupported)
            Skip = "The pinned bits are for 256-bit vectors with FMA.";
    }
}
