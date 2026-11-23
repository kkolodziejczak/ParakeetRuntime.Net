using System.Numerics;
using ParakeetRuntime;

namespace ParakeetRuntime.Tests;

public sealed class KernelsTests
{
    // By name: the engine's types are internal, xUnit theories are public.
    public static TheoryData<string> SupportedPaths()
    {
        var paths = new TheoryData<string>();
        foreach (var path in Enum.GetValues<Int8Path>().Where(Kernels.Supports))
            paths.Add(path.ToString());
        return paths;
    }

    // Quantizing's AVX2 path, where the CPU has it, and the 128-bit one of CPUs without it.
    public static TheoryData<bool> QuantizePaths()
    {
        var paths = new TheoryData<bool> { false };
        if (Kernels.Supports(Int8Path.Avx2))
            paths.Add(true);
        return paths;
    }

    // Like NeMo-Speech.cpp on x86 (ggml's AVX2 quantize_row_q8_0): scale amax / 127 rounded to
    // fp16, values times 127 / amax rounded half to even. The first block has amax 127 (scale 1),
    // so its halves land exactly on .5.
    [Theory]
    [MemberData(nameof(QuantizePaths))]
    public void Activations_are_quantized_per_block_like_ggml_on_x86(bool avx2)
    {
        var random = new Random(7);
        float[] halves = [127f, 0.5f, -0.5f, 1.5f, -2.5f, 126.5f, 0.49999997f, -0.49999997f, 2.4999998f, 3.5f];
        var x = new float[3 * 64];
        for (var i = 0; i < x.Length; i++)
            x[i] = i < halves.Length ? halves[i] : (float)(random.NextDouble() * 6 - 3);

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

    // The same formula over many blocks of very different loudness, from fp16-subnormal scales to
    // large ones and finite blocks too loud for fp16 (an infinite scale, values as usual), with
    // all-negative blocks, negative zeros and a block of nothing else: every scale, value and sum.
    [Theory]
    [MemberData(nameof(QuantizePaths))]
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

    // Silence gives all-zero blocks; a block so quiet that its fp16 scale is zero counts as
    // silent too, so its values and sum stay consistent with that zero scale.
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
        foreach (var avx2 in QuantizePaths())
            foreach (var loudest in new[] { 0f, 1e-38f, 1e-7f })
                cases.Add(loudest, avx2);
        return cases;
    }

    // A block holding NaN or infinity gets a NaN or infinite scale (NaN when it holds both), so
    // whatever it multiplies is lost anyway; its values are zeros and their sum is 0, wherever the
    // bad value sits. Its neighbours quantize as usual.
    [Theory]
    [MemberData(nameof(QuantizePaths))]
    public void Blocks_holding_NaN_or_infinity_quantize_to_zero(bool avx2)
    {
        float[] bad = [float.NaN, -float.NaN, float.PositiveInfinity, float.NegativeInfinity];
        int[] places = [0, 5, 10, 18, 31]; // every 8-value load, and every lane of a 4-value half
        int badBlocks = bad.Length * places.Length + 1, blocks = badBlocks + 1;
        var x = new float[blocks * 32];
        for (var i = 0; i < x.Length; i++)
            x[i] = (i % 13 - 6) * 0.25f;
        for (var b = 0; b < badBlocks - 1; b++)
            x[b * 32 + places[b % places.Length]] = bad[b / places.Length];
        x[(badBlocks - 1) * 32 + 3] = float.PositiveInfinity;
        x[(badBlocks - 1) * 32 + 20] = float.NaN;

        var q = QuantizeOverLouder(x, 2, x.Length / 2, avx2);

        for (var b = 0; b < badBlocks; b++)
        {
            Assert.Equal(b < badBlocks - 1 && float.IsInfinity(bad[b / places.Length]) ? float.PositiveInfinity : float.NaN, q.D[b]);
            Assert.All(q.Q.AsSpan(b * 32, 32).ToArray(), v => Assert.Equal(0, v));
            Assert.Equal(0, q.Sums[b]);
        }
        var good = x.AsSpan(badBlocks * 32, 32).ToArray();
        var expected = good.Select(v => (sbyte)MathF.Round(v * (127f / 1.5f), MidpointRounding.ToEven)).ToArray();
        Assert.Equal((float)(Half)(1.5f / 127f), q.D[badBlocks]);
        Assert.Equal(expected, q.Q.AsSpan(badBlocks * 32, 32).ToArray());
        Assert.Equal(expected.Sum(v => v), q.Sums[badBlocks]);
    }

    // Quantizes x into the buffers a louder input of the same shape just filled, as the encoder
    // reuses them layer after layer: blocks of zeros must be written, not left as they were.
    private static Q8Activations QuantizeOverLouder(float[] x, int rows, int k, bool avx2)
    {
        using var scratch = new EncoderScratch();
        var louder = Kernels.Quantize(Enumerable.Range(0, x.Length).Select(i => i % 7 + 1f).ToArray(), rows, k, avx2);
        var q = Kernels.Quantize(x, rows, k, avx2);
        Assert.Same(louder, q);
        return q;
    }

    // Every int8 path the CPU has must give the product of the quantized values, with or without
    // a bias, and apply SiLU or add a scaled product to a residual in the same pass. Weights span
    // the whole int8 range, -128 included: the file format allows it.
    // Shapes cover whole 4-row x 8-column and 4-row x 16-column tiles (with 4 threads, a thread
    // gets 2+ groups of 8 columns from 64 columns on) and the leftover rows and columns.
    [Theory]
    [MemberData(nameof(SupportedPaths))]
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

    // The subsampling's pointwise convolutions: whole 4-row tiles and leftover rows. Output
    // counts are multiples of 32, two vectors at any width up to 512 bits.
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

    // The float tiles write two whole vectors per step; a width they don't divide is refused
    // rather than written past.
    [Fact]
    public void The_float_product_refuses_a_width_that_is_not_whole_vector_pairs()
    {
        int outputs = 3 * Vector<float>.Count, k = 4;

        Assert.Throws<ArgumentException>(() => Kernels.GemmRelu(new float[k], 1, new float[k * outputs], outputs, k, new float[outputs]));
    }

    // FastConformer's relative-position attention, against the formula: for each head,
    // score[i][j] = ((q_i + u).k_j + (q_i + v).p[first + T-1-i+j]) / sqrt(dk), ctx_i = softmax(score_i).V.
    // first > 0 is a slice of a table made for a longer utterance; T = 7 and 37 leave a partial
    // row tile, and T = 37 needs several tiles of key columns, T = 70 several blocks of query rows. dk = 32 is two vectors at any width
    // up to 512 bits.
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

    // The exact bits of the attention output, pinned so that a faster loop order cannot slip in a
    // reordered sum: transcripts must stay identical. dk = 128 as in the model; T = 70 and 300 span
    // several blocks of query rows with a partial row tile at the end. The bits depend on 256-bit
    // FMA vectors and on .NET's Vector.Exp: update them only together with a corpus transcript
    // comparison (Spike --corpus ... --compare).
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

    // The engine's layer norm in float vectors must match a plain double-precision one.
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

// The CPU engine's own float path: 256-bit vectors with fused multiply-add (AVX2 and FMA).
internal sealed class FmaFactAttribute : FactAttribute
{
    public FmaFactAttribute()
    {
        if (Vector<float>.Count != 8 || !System.Runtime.Intrinsics.X86.Fma.IsSupported)
            Skip = "The pinned bits are for 256-bit vectors with FMA.";
    }
}
