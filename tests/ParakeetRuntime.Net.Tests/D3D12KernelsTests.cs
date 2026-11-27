using ParakeetRuntime.Net;
using static ParakeetRuntime.Net.D3D12Native;

namespace ParakeetRuntime.Net.Tests;

internal static class GpuTestSupport
{
    private static readonly Lazy<List<GpuAdapterInfo>> EligibleAdapters = new(() =>
        D3D12Device.EnumerateAdapters().Where(a => a.MeetsMinimumMemory).ToList());

    public static bool HasHardwareGpu => EligibleAdapters.Value.Count > 0;

    public static IReadOnlyList<GpuAdapterInfo> Adapters => EligibleAdapters.Value;

    public static void ForEachHardwareAdapter(Action<D3D12Device> run)
    {
        int ran = 0;
        foreach (var adapter in EligibleAdapters.Value)
        {
            using var gpu = D3D12Device.TryCreate(adapterNameFilter: adapter.Name);
            if (gpu is null)
                continue;
            ran++;
            run(gpu);
        }
        Assert.True(ran > 0, "Expected at least one eligible hardware D3D12 adapter.");
    }
}

internal sealed class GpuFactAttribute : FactAttribute
{
    public GpuFactAttribute()
    {
        if (!GpuTestSupport.HasHardwareGpu)
            Skip = "No eligible hardware D3D12 GPU available on this machine.";
    }
}

internal sealed class GpuTheoryAttribute : TheoryAttribute
{
    public GpuTheoryAttribute()
    {
        if (!GpuTestSupport.HasHardwareGpu)
            Skip = "No eligible hardware D3D12 GPU available on this machine.";
    }
}

public sealed unsafe class D3D12KernelsTests
{
    [GpuTheory]
    [InlineData(1, 32, 3, false)]
    [InlineData(7, 64, 19, true)]
    [InlineData(32, 128, 64, false)]
    [InlineData(65, 256, 130, true)]
    [InlineData(455, 128, 131, true)]
    [InlineData(97, 96, 200, false)]
    public void Gpu_q8_linear_matches_cpu_reference(int m, int k, int n, bool withBias)
    {
        foreach (var (entry, tile) in new[] { ("Q8GemmMain", ParakeetGpu.Q8GemmLargeTile), ("Q8GemmSmallMain", ParakeetGpu.Q8GemmSmallTile) })
            Q8LinearMatchesCpuReference(m, k, n, withBias, entry, tile);
    }

    private static void Q8LinearMatchesCpuReference(int m, int k, int n, bool withBias, string entry, int tile)
    {
        var random = new Random(m * 10000 + k * 100 + n + (withBias ? 1 : 0));
        var x = Enumerable.Range(0, m * k).Select(_ => (float)(random.NextDouble() * 4 - 2)).ToArray();
        var w = new Q8Matrix(n, k);
        for (var i = 0; i < w.Q.Length; i++)
            w.Q[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < w.D.Length; i++)
            w.D[i] = (float)(Half)((float)random.NextDouble() / 50f);
        float[]? bias = withBias
            ? Enumerable.Range(0, n).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray()
            : null;

        var expected = new float[m * n];
        for (var row = 0; row < m; row++)
        {
            for (var col = 0; col < n; col++)
            {
                double sum = 0;
                for (var b = 0; b < w.Blocks; b++)
                {
                    double blockSum = 0;
                    for (var i = 0; i < 32; i++)
                        blockSum += x[row * k + b * 32 + i] * w.Q[col * k + b * 32 + i];
                    sum += blockSum * w.D[col * w.Blocks + b];
                }
                expected[row * n + col] = (float)sum + (bias?[col] ?? 0f);
            }
        }

        GpuTestSupport.ForEachHardwareAdapter(gpu =>
        {
            using var runner = new Q8GpuLinearProbe(gpu, w, bias, entry, tile);
            var actual = runner.Linear(x, m);

            for (var i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[i])), $"{entry} on {gpu.AdapterName}: element {i} (row {i / n}, column {i % n}) is {actual[i]}, expected {expected[i]}");
        });
    }

    private static readonly (string Entry, int Tile)[] Q8Kernels =
        [("Q8GemmMain", ParakeetGpu.Q8GemmLargeTile), ("Q8GemmSmallMain", ParakeetGpu.Q8GemmSmallTile)];

    private static (float[] X, Q8Matrix W, float[] Bias) RandomQ8Linear(int seed, int m, int k, int n)
    {
        var random = new Random(seed);
        var x = Enumerable.Range(0, m * k).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var w = new Q8Matrix(n, k);
        for (var i = 0; i < w.Q.Length; i++)
            w.Q[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < w.D.Length; i++)
            w.D[i] = (float)(Half)((float)random.NextDouble() / 50f);
        var bias = Enumerable.Range(0, n).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        return (x, w, bias);
    }

    private static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();

    [GpuTheory]
    [InlineData(150, true)]
    [InlineData(170, false)]
    [InlineData(128, false)]
    public void Gpu_q8_linear_kernels_give_the_same_bits(int m, bool withBias)
    {
        const int k = 1024, n = 203;
        var (x, w, randomBias) = RandomQ8Linear(7, m, k, n);
        var bias = withBias ? randomBias : null;
        var residual = RandomQ8Linear(8, m, n, 32).X;

        GpuTestSupport.ForEachHardwareAdapter(gpu =>
        {
            using var large = new Q8GpuLinearProbe(gpu, w, bias, Q8Kernels[0].Entry, Q8Kernels[0].Tile);
            using var small = new Q8GpuLinearProbe(gpu, w, bias, Q8Kernels[1].Entry, Q8Kernels[1].Tile);

            foreach (var epilogue in Enum.GetValues<GpuEpilogue>())
                Assert.Equal(Bits(small.Linear(x, m, epilogue, residual)), Bits(large.Linear(x, m, epilogue, residual)));
        });
    }

    [GpuFact]
    public void Gpu_q8_linear_reads_its_input_at_an_srv_and_byte_offset()
    {
        const int m = 97, k = 64, n = 96;
        var (x, w, bias) = RandomQ8Linear(9, m, k, n);

        GpuTestSupport.ForEachHardwareAdapter(gpu =>
        {
            foreach (var (entry, tile) in Q8Kernels)
            {
                using var probe = new Q8GpuLinearProbe(gpu, w, bias, entry, tile);
                Assert.Equal(Bits(probe.Linear(x, m)), Bits(probe.Linear(x, m, inputOffset: (256, 3 * k * 4))));
            }
        });
    }

    [GpuFact]
    public void Gpu_q8_linear_epilogues_match_cpu_reference()
    {
        const int m = 67, k = 64, n = 67;
        var random = new Random(42);
        var x = Enumerable.Range(0, m * k).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var residual = Enumerable.Range(0, m * n).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var w = new Q8Matrix(n, k);
        for (var i = 0; i < w.Q.Length; i++)
            w.Q[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < w.D.Length; i++)
            w.D[i] = (float)(Half)((float)random.NextDouble() / 100f);

        var bias = Enumerable.Range(0, n).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();

        GpuTestSupport.ForEachHardwareAdapter(gpu =>
        {
            foreach (var (entry, tile) in Q8Kernels)
            {
                using var runner = new Q8GpuLinearProbe(gpu, w, bias, entry, tile);
                var plain = runner.Linear(x, m);
                var silu = runner.Linear(x, m, GpuEpilogue.Silu);
                var addRes = runner.Linear(x, m, GpuEpilogue.AddResidual, residual);
                var addHalf = runner.Linear(x, m, GpuEpilogue.AddHalfResidual, residual);

                for (var i = 0; i < plain.Length; i++)
                {
                    Assert.Equal(Kernels.Silu(plain[i]), silu[i], tolerance: 1e-4f * Math.Max(1f, Math.Abs(plain[i])));
                    Assert.Equal(residual[i] + plain[i], addRes[i], tolerance: 1e-4f * Math.Max(1f, Math.Abs(residual[i] + plain[i])));
                    Assert.Equal(residual[i] + 0.5f * plain[i], addHalf[i], tolerance: 1e-4f * Math.Max(1f, Math.Abs(residual[i] + 0.5f * plain[i])));
                }
            }
        });
    }

    [GpuFact]
    public void Gpu_encoder_ops_match_cpu_reference()
    {
        var random = new Random(12345);
        float[] Rand(int n) => Enumerable.Range(0, n).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();

        const int T = 17, D = 1024, heads = 8, dk = 128, kernel = 9;
        var x = Rand(T * D);
        var g = Rand(D);
        var b = Rand(D);
        var expectedLn = Kernels.LayerNorm(x, T, D, g, b);

        var qkv = Rand(T * 3 * D);
        var posP = Rand((2 * T - 1) * D);
        var u = Rand(D);
        var v = Rand(D);
        var expectedAtt = new float[T * D];
        float scale = 1f / MathF.Sqrt(dk);
        for (int hd = 0; hd < heads; hd++)
        {
            int ho = hd * dk;
            for (int i = 0; i < T; i++)
            {
                var scores = new float[T];
                float max = float.NegativeInfinity;
                for (int j = 0; j < T; j++)
                {
                    float s = 0;
                    for (int e = 0; e < dk; e++)
                    {
                        float q = qkv[i * 3 * D + ho + e];
                        s += (q + u[ho + e]) * qkv[j * 3 * D + D + ho + e] + (q + v[ho + e]) * posP[(T - 1 - i + j) * D + ho + e];
                    }
                    scores[j] = s * scale;
                    if (scores[j] > max) max = scores[j];
                }
                float sum = 0;
                for (int j = 0; j < T; j++) { scores[j] = MathF.Exp(scores[j] - max); sum += scores[j]; }
                for (int j = 0; j < T; j++)
                {
                    float a = scores[j] / sum;
                    for (int e = 0; e < dk; e++)
                        expectedAtt[i * D + ho + e] += a * qkv[j * 3 * D + 2 * D + ho + e];
                }
            }
        }

        var pw1 = Rand(T * 2 * D);
        var expectedGlu = new float[T * D];
        for (int t = 0; t < T; t++)
            for (int ch = 0; ch < D; ch++)
                expectedGlu[t * D + ch] = pw1[t * 2 * D + ch] * Kernels.Sigmoid(pw1[t * 2 * D + D + ch]);

        var dw = Rand(D * kernel);
        var bnScale = Rand(D);
        var bnShift = Rand(D);
        var expectedDw = new float[T * D];
        int pad = (kernel - 1) / 2;
        for (int t = 0; t < T; t++)
        {
            for (int ch = 0; ch < D; ch++)
            {
                float s = 0;
                for (int k = 0; k < kernel; k++)
                {
                    int ti = t + k - pad;
                    if (ti >= 0 && ti < T) s += expectedGlu[ti * D + ch] * dw[ch * kernel + k];
                }
                expectedDw[t * D + ch] = Kernels.Silu(s * bnScale[ch] + bnShift[ch]);
            }
        }

        GpuTestSupport.ForEachHardwareAdapter(gpu =>
        {
            using var probe = new EncoderOpsGpuProbe(gpu);
            var actualLn = probe.LayerNorm(x, T, D, g, b);
            for (int i = 0; i < expectedLn.Length; i++)
                Assert.Equal(expectedLn[i], actualLn[i], tolerance: 1e-4f * Math.Max(1f, Math.Abs(expectedLn[i])));

            var actualAtt = probe.RelPosAttention(qkv, posP, u, v, T, D, heads);
            for (int i = 0; i < expectedAtt.Length; i++)
                Assert.Equal(expectedAtt[i], actualAtt[i], tolerance: 1e-4f * Math.Max(1f, Math.Abs(expectedAtt[i])));

            var actualGlu = probe.Glu(pw1, T, D);
            for (int i = 0; i < expectedGlu.Length; i++)
                Assert.Equal(expectedGlu[i], actualGlu[i], tolerance: 1e-5f * Math.Max(1f, Math.Abs(expectedGlu[i])));

            var actualDw = probe.DepthwiseConvBnSilu(expectedGlu, dw, bnScale, bnShift, T, D, kernel);
            for (int i = 0; i < expectedDw.Length; i++)
                Assert.Equal(expectedDw[i], actualDw[i], tolerance: 1e-4f * Math.Max(1f, Math.Abs(expectedDw[i])));
        });
    }

    [GpuTheory]
    [InlineData(1, 256, 4)]
    [InlineData(32, 512, 4)]
    [InlineData(65, 512, 4)]
    [InlineData(71, 256, 4)]
    [InlineData(37, 288, 3)]
    public void Gpu_rel_pos_attention_tiles_match_cpu_reference(int t, int d, int heads)
    {
        int dk = d / heads;
        var random = new Random(t * 10000 + d * 10 + heads);
        float[] Rand(int n) => Enumerable.Range(0, n).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();

        var qkv = Rand(t * 3 * d);
        var posP = Rand((2 * t - 1) * d);
        var u = Rand(d);
        var v = Rand(d);
        var expected = new float[t * d];
        float scale = 1f / MathF.Sqrt(dk);
        for (int hd = 0; hd < heads; hd++)
        {
            int ho = hd * dk;
            for (int i = 0; i < t; i++)
            {
                var scores = new float[t];
                float max = float.NegativeInfinity;
                for (int j = 0; j < t; j++)
                {
                    double s = 0;
                    for (int e = 0; e < dk; e++)
                    {
                        float q = qkv[i * 3 * d + ho + e];
                        s += (q + u[ho + e]) * qkv[j * 3 * d + d + ho + e] + (q + v[ho + e]) * posP[(t - 1 - i + j) * d + ho + e];
                    }
                    scores[j] = (float)s * scale;
                    if (scores[j] > max) max = scores[j];
                }
                double sum = 0;
                for (int j = 0; j < t; j++) { scores[j] = MathF.Exp(scores[j] - max); sum += scores[j]; }
                for (int j = 0; j < t; j++)
                {
                    float a = (float)(scores[j] / sum);
                    for (int e = 0; e < dk; e++)
                        expected[i * d + ho + e] += a * qkv[j * 3 * d + 2 * d + ho + e];
                }
            }
        }

        GpuTestSupport.ForEachHardwareAdapter(gpu =>
        {
            using var probe = new EncoderOpsGpuProbe(gpu);
            var actual = probe.RelPosAttention(qkv, posP, u, v, t, d, heads);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i], tolerance: 1e-4f * Math.Max(1f, Math.Abs(expected[i])));
        });
    }

    [GpuFact]
    public void Gpu_attention_over_one_frame_returns_its_value_even_when_the_score_overflows()
    {
        const int d = 64, heads = 2;
        const int queryStart = 0, keyStart = d, valueStart = 2 * d;
        var qkv = new float[3 * d];
        qkv[queryStart] = float.PositiveInfinity;
        qkv[keyStart] = 1;
        for (int e = 0; e < d; e++) qkv[valueStart + e] = e + 1;
        var posP = new float[d];
        posP[0] = 1;
        var zero = new float[d];

        GpuTestSupport.ForEachHardwareAdapter(gpu =>
        {
            using var probe = new EncoderOpsGpuProbe(gpu);
            var actual = probe.RelPosAttention(qkv, posP, zero, zero, 1, d, heads);
            Assert.Equal(qkv[valueStart..], actual);
        });
    }

    [GpuFact]
    public void Gpu_decoder_ops_match_cpu_reference()
    {
        var random = new Random(999);

        const int k = 640, n = 130;
        var x = Enumerable.Range(0, k).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var w = new Q8Matrix(n, k);
        for (int i = 0; i < w.Q.Length; i++) w.Q[i] = (sbyte)random.Next(-127, 128);
        for (int i = 0; i < w.D.Length; i++) w.D[i] = (float)(Half)((float)random.NextDouble() / 50f);
        var bias = Enumerable.Range(0, n).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();

        var expected = new float[n];
        for (int c = 0; c < n; c++)
        {
            double sum = 0;
            for (int b = 0; b < w.Blocks; b++)
            {
                double bs = 0;
                for (int i = 0; i < 32; i++) bs += x[b * 32 + i] * w.Q[c * k + b * 32 + i];
                sum += bs * w.D[c * w.Blocks + b];
            }
            expected[c] = (float)sum + bias[c];
        }

        const int vocab = 8193, blank = 8192;
        int[] durations = [0, 1, 2, 3, 4];
        var logits = Enumerable.Range(0, vocab + durations.Length).Select(_ => (float)(random.NextDouble() * 10 - 5)).ToArray();
        logits[42] = 999f;
        logits[500] = 999f;
        logits[vocab + 2] = 100f;

        GpuTestSupport.ForEachHardwareAdapter(gpu =>
        {
            using var probe = new DecoderOpsGpuProbe(gpu);
            var actual = probe.Q8Gemv(x, w, bias);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i], tolerance: 1e-3f * Math.Max(1f, Math.Abs(expected[i])));

            var (tok, tNew, syms, needPred, slot) = probe.ArgMaxStep(logits, vocab, blank, durations, maxSymbols: 10, initialNeedPred: 1u, initialSlot: 0u);
            Assert.Equal(42u, tok);
            Assert.Equal(2u, tNew);
            Assert.Equal(0u, syms);
            Assert.Equal(1u, needPred);
            Assert.Equal(1u, slot);
        });
    }

    [GpuFact]
    public void Gpu_fused_lstm_and_joint_relu_match_cpu_reference()
    {
        var random = new Random(1337);
        const int h = 64, vocab = 8, label = 3, tFrame = 2, nOut = 35;

        float[] Rand(int len, float scale = 1f) => Enumerable.Range(0, len).Select(_ => (float)(random.NextDouble() * 2 - 1) * scale).ToArray();
        Q8Matrix RandQ8(int rows, int cols)
        {
            var m = new Q8Matrix(rows, cols);
            for (int i = 0; i < m.Q.Length; i++) m.Q[i] = (sbyte)random.Next(-127, 128);
            for (int i = 0; i < m.D.Length; i++) m.D[i] = (float)(Half)((float)random.NextDouble() / 50f);
            return m;
        }
        float[] Q8Vec(float[] x, Q8Matrix w)
        {
            var y = new float[w.Rows];
            for (int r = 0; r < w.Rows; r++)
            {
                double sum = 0;
                for (int b = 0; b < w.Blocks; b++)
                {
                    double bs = 0;
                    for (int i = 0; i < 32; i++) bs += x[b * 32 + i] * w.Q[r * w.K + b * 32 + i];
                    sum += bs * w.D[r * w.Blocks + b];
                }
                y[r] = (float)sum;
            }
            return y;
        }

        var embed = Rand(vocab * h);
        var hPrev = Rand(h);
        var cPrev = Rand(h);
        var ih = RandQ8(4 * h, h);
        var hh = RandQ8(4 * h, h);
        var bias = Rand(4 * h);

        var xIn = embed.AsSpan(label * h, h).ToArray();
        var gi = Q8Vec(xIn, ih);
        var gh = Q8Vec(hPrev, hh);
        var expectedH = new float[h];
        var expectedC = new float[h];
        for (int k = 0; k < h; k++)
        {
            float ig = Kernels.Sigmoid(gi[k] + gh[k] + bias[k]);
            float fg = Kernels.Sigmoid(gi[h + k] + gh[h + k] + bias[h + k]);
            float gg = MathF.Tanh(gi[2 * h + k] + gh[2 * h + k] + bias[2 * h + k]);
            float og = Kernels.Sigmoid(gi[3 * h + k] + gh[3 * h + k] + bias[3 * h + k]);
            expectedC[k] = fg * cPrev[k] + ig * gg;
            expectedH[k] = og * MathF.Tanh(expectedC[k]);
        }

        var encProj = Rand(4 * h);
        var predProj = Rand(h);
        var jointOut = RandQ8(nOut, h);
        var outBias = Rand(nOut);
        var jointIn = new float[h];
        for (int k = 0; k < h; k++)
            jointIn[k] = MathF.Max(0f, encProj[tFrame * h + k] + predProj[k]);
        var expectedLogits = Q8Vec(jointIn, jointOut);
        for (int i = 0; i < nOut; i++)
            expectedLogits[i] += outBias[i];

        GpuTestSupport.ForEachHardwareAdapter(gpu =>
        {
            using var probe = new DecoderOpsGpuProbe(gpu);
            var (actualH, actualC) = probe.FusedLstmEmbed(embed, hPrev, cPrev, ih, hh, bias, label, srcSlot: 0u);
            for (int k = 0; k < h; k++)
            {
                Assert.Equal(expectedH[k], actualH[k], tolerance: 1e-3f * Math.Max(1f, Math.Abs(expectedH[k])));
                Assert.Equal(expectedC[k], actualC[k], tolerance: 1e-3f * Math.Max(1f, Math.Abs(expectedC[k])));
            }

            var actualLogits = probe.Q8GemvFusedJointRelu(encProj, predProj, jointOut, outBias, tFrame, totalT: 4);
            for (int i = 0; i < nOut; i++)
                Assert.Equal(expectedLogits[i], actualLogits[i], tolerance: 1e-3f * Math.Max(1f, Math.Abs(expectedLogits[i])));

            if (gpu.HasTimestamps)
            {
                using var tsRb = gpu.CreateReadbackBuffer(256);
                gpu.BeginCommands();
                gpu.WriteTimestamp(0);
                gpu.WriteTimestamp(1);
                gpu.ResolveTimestamps(0, 2, tsRb, 0);
                gpu.ExecuteAndWait();
                var pTs = (ulong*)tsRb.Map(forRead: true);
                ulong t0 = pTs[0], t1 = pTs[1];
                tsRb.Unmap(wasWritten: false);
                Assert.True(t1 >= t0, $"Expected monotonic GPU timestamps, got t0={t0}, t1={t1}.");
            }
        });
    }

    [GpuFact]
    public void Gpu_lstm_saturates_instead_of_overflowing_when_cell_or_gate_values_pass_88()
    {
        const int h = 64, saturating = 30;
        var cPrev = new float[h];
        var bias = new float[4 * h];
        for (int k = 0; k < h; k++)
        {
            float magnitude = 88f + k % 32 * (k < 32 ? 0.4f : 4f);
            float sign = k % 2 == 0 ? 1f : -1f;
            bool cellCarriesTheValue = k < 32;
            bias[k] = cellCarriesTheValue ? -saturating : saturating;
            bias[h + k] = cellCarriesTheValue ? saturating : -saturating;
            bias[2 * h + k] = cellCarriesTheValue ? 0f : sign * magnitude;
            bias[3 * h + k] = saturating;
            cPrev[k] = cellCarriesTheValue ? sign * magnitude : 0f;
        }
        var expectedH = new float[h];
        for (int k = 0; k < h; k++)
        {
            float ig = Kernels.Sigmoid(bias[k]), fg = Kernels.Sigmoid(bias[h + k]), og = Kernels.Sigmoid(bias[3 * h + k]);
            float c = fg * cPrev[k] + ig * MathF.Tanh(bias[2 * h + k]);
            expectedH[k] = og * MathF.Tanh(c);
        }

        GpuTestSupport.ForEachHardwareAdapter(gpu =>
        {
            using var probe = new DecoderOpsGpuProbe(gpu);
            var (actualH, _) = probe.FusedLstmEmbed(new float[4 * h], new float[h], cPrev, new Q8Matrix(4 * h, h), new Q8Matrix(4 * h, h), bias, label: 1, srcSlot: 0u);
            var wrong = Enumerable.Range(0, h)
                .Where(k => !float.IsFinite(actualH[k]) || Math.Abs(expectedH[k] - actualH[k]) > 1e-4f)
                .Select(k => $"unit {k}: input {(k < 32 ? cPrev[k] : bias[2 * h + k])}, expected {expectedH[k]}, got {actualH[k]}");
            Assert.Empty(wrong);
        });
    }

    [GpuTheory]
    [InlineData(8000)]
    [InlineData(10)]
    public void Gpu_mel_matches_cpu_reference(int sampleCount)
    {
        var path = Path.Combine(Path.GetTempPath(), $"mel-test-{Guid.NewGuid():N}.gguf");
        try
        {
            var random = new Random(77);
            var fb = Enumerable.Range(0, 128 * 257).Select(_ => (float)random.NextDouble() * 0.1f).ToArray();
            using (var ms = new MemoryStream())
            using (var bw = new BinaryWriter(ms))
            {
                void WriteStr(string s) { var b = System.Text.Encoding.UTF8.GetBytes(s); bw.Write((long)b.Length); bw.Write(b); }
                bw.Write(0x46554747u); bw.Write(3u);
                bw.Write(1L);
                bw.Write(6L);
                WriteStr("asr.preprocessor.n_fft"); bw.Write(5); bw.Write(512);
                WriteStr("asr.preprocessor.sample_rate"); bw.Write(5); bw.Write(16000);
                WriteStr("asr.preprocessor.window_stride"); bw.Write(6); bw.Write(0.01f);
                WriteStr("asr.preprocessor.window_size"); bw.Write(6); bw.Write(0.025f);
                WriteStr("asr.preprocessor.features"); bw.Write(5); bw.Write(128);
                WriteStr("asr.preprocessor.preemph"); bw.Write(6); bw.Write(0.97f);
                WriteStr("preprocessor.fb"); bw.Write(2); bw.Write(257L); bw.Write(128L); bw.Write(0); bw.Write(0L);
                long pad = ((ms.Length + 31) / 32 * 32) - ms.Length;
                bw.Write(new byte[pad]);
                foreach (var v in fb) bw.Write(v);
                File.WriteAllBytes(path, ms.ToArray());
            }

            using var g = new Gguf(path);
            var mel = new Mel(g);
            var audio = Enumerable.Range(0, sampleCount).Select(i => MathF.Sin(i * 0.05f) * 0.3f + (float)(random.NextDouble() - 0.5) * 0.1f).ToArray();
            var expected = mel.Compute(audio, out int expectedFrames);

            GpuTestSupport.ForEachHardwareAdapter(gpu =>
            {
                using var probe = new MelOpsGpuProbe(gpu);
                var actual = probe.Compute(audio, mel, out int actualFrames);

                Assert.Equal(expectedFrames, actualFrames);
                for (int i = 0; i < expected.Length; i++)
                    Assert.Equal(expected[i], actual[i], tolerance: 1e-3f * Math.Max(1f, Math.Abs(expected[i])));
            });
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [GpuFact]
    public void Gpu_f32_gemm_relu_with_unaligned_dimensions_matches_cpu_reference()
    {
        const int m = 35, k = 64, n = 70;
        var random = new Random(314);
        var x = Enumerable.Range(0, m * k).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var w = Enumerable.Range(0, n * k).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var bias = Enumerable.Range(0, n).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var expected = new float[m * n];
        for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++)
            {
                double sum = bias[c];
                for (int j = 0; j < k; j++) sum += x[r * k + j] * w[c * k + j];
                expected[r * n + c] = MathF.Max(0f, (float)sum);
            }

        GpuTestSupport.ForEachHardwareAdapter(gpu =>
        {
            using var probe = new SubsampleOpsGpuProbe(gpu);
            var actual = probe.F32GemmRelu(x, w, bias, m, n, k);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i], tolerance: 1e-3f * Math.Max(1f, Math.Abs(expected[i])));
        });
    }

    [Fact]
    public void Gpu_device_initialization_failure_returns_null_instead_of_throwing()
    {
        using var gpu = D3D12Device.TryCreate(minimumMemoryBytes: long.MaxValue);
        Assert.Null(gpu);
        using var missing = D3D12Device.TryCreate(adapterNameFilter: "__nonexistent_gpu_adapter__");
        Assert.Null(missing);
    }

    [GpuFact]
    public void An_integrated_gpu_is_used_only_when_allowed()
    {
        foreach (var adapter in GpuTestSupport.Adapters)
        {
            using var allowed = D3D12Device.TryCreate(adapterNameFilter: adapter.Name, integratedGpuAllowed: true);
            if (allowed is null)
                continue;
            using var notAllowed = D3D12Device.TryCreate(adapterNameFilter: adapter.Name, integratedGpuAllowed: false);
            Assert.Equal(allowed.IsIntegrated, notAllowed is null);
            if (IsAnIntelIntegratedGpuName(adapter.Name))
                Assert.True(allowed.IsIntegrated, $"{adapter.Name} is an integrated GPU.");
        }
    }

    [GpuFact]
    public void Timed_operations_add_up_per_operation_across_submits()
    {
        GpuTestSupport.ForEachHardwareAdapter(gpu =>
        {
            var times = gpu.TimeEveryOperation();
            if (times is null)
                return;
            for (int submit = 0; submit < 2; submit++)
            {
                gpu.BeginCommands();
                gpu.MarkOperationEnd("first");
                gpu.MarkOperationEnd("second");
                gpu.ExecuteAndWait();
            }
            var totals = times.Totals.ToDictionary(t => t.Operation);
            Assert.Equal(["first", "second"], totals.Keys.Order());
            Assert.All(totals.Values, t => Assert.Equal(2, t.Count));
            Assert.All(totals.Values, t => Assert.True(t.Milliseconds >= 0));
            times.Clear();
            Assert.Empty(times.Totals);
        });
    }

    private static bool IsAnIntelIntegratedGpuName(string name) =>
        name.Contains("Intel", StringComparison.OrdinalIgnoreCase) &&
        (name.Contains("UHD", StringComparison.OrdinalIgnoreCase) || name.Contains("Iris", StringComparison.OrdinalIgnoreCase));

    private sealed class Q8GpuLinearProbe : IDisposable
    {
        private readonly D3D12Device gpu;
        private readonly Q8Matrix weight;
        private readonly float[]? bias;
        private readonly GpuBuffer wqBuf, wdBuf, biasBuf;
        private readonly int tile;
        private nint pso;

        public Q8GpuLinearProbe(D3D12Device gpu, Q8Matrix weight, float[]? bias = null, string entry = "Q8GemmMain", int tile = ParakeetGpu.Q8GemmLargeTile)
        {
            this.gpu = gpu;
            this.weight = weight;
            this.bias = bias;
            this.tile = tile;
            pso = gpu.EmbeddedPipeline("Q8Gemm.hlsl", entry);

            ulong qBytes = (ulong)weight.Q.Length;
            ulong dBytes = (ulong)((weight.D.Length + 1) / 2 * 4);
            ulong bBytes = (ulong)((bias?.Length ?? 1) * 4);

            wqBuf = gpu.CreateDefaultBuffer(qBytes, allowUav: false);
            wdBuf = gpu.CreateDefaultBuffer(dBytes, allowUav: false);
            biasBuf = gpu.CreateDefaultBuffer(bBytes, allowUav: false);

            using var upQ = gpu.CreateUploadBuffer(qBytes);
            using var upD = gpu.CreateUploadBuffer(dBytes);
            using var upB = gpu.CreateUploadBuffer(bBytes);

            weight.Q.AsSpan().CopyTo(new Span<sbyte>((sbyte*)upQ.Map(), weight.Q.Length)); upQ.Unmap();
            var pD = (ushort*)upD.Map();
            for (int i = 0; i < weight.D.Length; i++) pD[i] = BitConverter.HalfToUInt16Bits((Half)weight.D[i]);
            if ((weight.D.Length & 1) != 0) pD[weight.D.Length] = 0;
            upD.Unmap();
            var pB = (float*)upB.Map();
            if (bias is not null) bias.AsSpan().CopyTo(new Span<float>(pB, bias.Length)); else pB[0] = 0f;
            upB.Unmap();

            gpu.BeginCommands();
            gpu.Transition(wqBuf, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.Transition(wdBuf, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.Transition(biasBuf, D3D12_RESOURCE_STATE_COPY_DEST);
            CopyBufferRegion(gpu.CommandList, wqBuf.Resource, 0, upQ.Resource, 0, qBytes);
            CopyBufferRegion(gpu.CommandList, wdBuf.Resource, 0, upD.Resource, 0, dBytes);
            CopyBufferRegion(gpu.CommandList, biasBuf.Resource, 0, upB.Resource, 0, bBytes);
            gpu.Transition(wqBuf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(wdBuf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(biasBuf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.ExecuteAndWait();
        }

        public float[] Linear(float[] x, int m, GpuEpilogue epilogue = GpuEpilogue.Store, float[]? initialY = null, (ulong SrvStart, uint ShaderByteOffset) inputOffset = default)
        {
            int n = weight.Rows, k = weight.K;
            ulong nanPrefixBytes = inputOffset.SrvStart + inputOffset.ShaderByteOffset;
            ulong xBytes = nanPrefixBytes + (ulong)(m * k) * 4, yBytes = (ulong)(m * n) * 4;
            using var xGpu = gpu.CreateDefaultBuffer(xBytes, allowUav: false);
            using var yGpu = gpu.CreateDefaultBuffer(yBytes, allowUav: true);
            using var upX = gpu.CreateUploadBuffer(xBytes);
            using var upY = initialY is not null ? gpu.CreateUploadBuffer(yBytes) : null;
            using var rbY = gpu.CreateReadbackBuffer(yBytes);

            var pX = (float*)upX.Map();
            new Span<float>(pX, (int)(nanPrefixBytes / 4)).Fill(float.NaN);
            x.AsSpan(0, m * k).CopyTo(new Span<float>(pX + nanPrefixBytes / 4, m * k)); upX.Unmap();
            if (initialY is not null && upY is not null)
            {
                initialY.AsSpan(0, m * n).CopyTo(new Span<float>((float*)upY.Map(), m * n)); upY.Unmap();
            }

            gpu.BeginCommands();
            gpu.Transition(xGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            CopyBufferRegion(gpu.CommandList, xGpu.Resource, 0, upX.Resource, 0, xBytes);
            gpu.Transition(xGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (upY is not null)
            {
                gpu.Transition(yGpu, D3D12_RESOURCE_STATE_COPY_DEST);
                CopyBufferRegion(gpu.CommandList, yGpu.Resource, 0, upY.Resource, 0, yBytes);
            }
            gpu.Transition(yGpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            SetPipelineState(gpu.CommandList, pso);
            Span<uint> constants = [(uint)m, (uint)n, (uint)k, bias is not null ? 1u : 0u, (uint)epilogue, inputOffset.ShaderByteOffset, 0u, 0u];
            gpu.SetRootConstants(constants);
            gpu.SetSrv(0, xGpu, inputOffset.SrvStart); gpu.SetSrv(1, wqBuf); gpu.SetSrv(2, wdBuf); gpu.SetSrv(3, biasBuf);
            gpu.SetUav(0, yGpu);
            Dispatch(gpu.CommandList, (uint)((n + tile - 1) / tile), (uint)((m + tile - 1) / tile), 1);

            gpu.Transition(yGpu, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CopyBufferRegion(gpu.CommandList, rbY.Resource, 0, yGpu.Resource, 0, yBytes);
            gpu.ExecuteAndWait();

            var result = new float[m * n];
            new ReadOnlySpan<float>((float*)rbY.Map(forRead: true), result.Length).CopyTo(result);
            rbY.Unmap(wasWritten: false);
            return result;
        }

        public void Dispose() { SafeRelease(ref pso); biasBuf.Dispose(); wdBuf.Dispose(); wqBuf.Dispose(); }
    }

    private sealed class EncoderOpsGpuProbe : IDisposable
    {
        private readonly D3D12Device gpu;
        private nint psoLn, psoAtt, psoGlu, psoDw;

        public EncoderOpsGpuProbe(D3D12Device gpu)
        {
            this.gpu = gpu;
            psoLn = gpu.EmbeddedPipeline("EncoderOps.hlsl", "LayerNormMain");
            psoAtt = gpu.EmbeddedPipeline("EncoderOps.hlsl", "RelPosAttentionMain");
            psoGlu = gpu.EmbeddedPipeline("EncoderOps.hlsl", "GluMain");
            psoDw = gpu.EmbeddedPipeline("EncoderOps.hlsl", "DepthwiseConvBnSiluMain");
        }

        public float[] LayerNorm(float[] x, int rows, int d, float[] g, float[] b) => Run(psoLn, [(uint)d], (uint)rows, 1, rows * d, x, g, b);
        public float[] RelPosAttention(float[] qkv, float[] posP, float[] u, float[] v, int t, int d, int heads)
        {
            int dk = d / heads;
            return Run(psoAtt, [(uint)t, (uint)d, (uint)dk, BitConverter.SingleToUInt32Bits(1f / MathF.Sqrt(dk))], (uint)((t + 31) / 32), (uint)heads, t * d, qkv, posP, u, v);
        }
        public float[] Glu(float[] c, int t, int d) => Run(psoGlu, [(uint)(t * d), (uint)d], (uint)((t * d + 255) / 256), 1, t * d, c);
        public float[] DepthwiseConvBnSilu(float[] glu, float[] dw, float[] bnScale, float[] bnShift, int t, int d, int kernel) =>
            Run(psoDw, [(uint)t, (uint)d, (uint)kernel], (uint)((t * d + 255) / 256), 1, t * d, glu, dw, bnScale, bnShift);

        private float[] Run(nint pso, ReadOnlySpan<uint> constants, uint gx, uint gy, int outLen, params float[][] inputs)
        {
            var gpuIns = new GpuBuffer[inputs.Length];
            var upIns = new GpuBuffer[inputs.Length];
            ulong outBytes = (ulong)outLen * 4;
            using var outGpu = gpu.CreateDefaultBuffer(outBytes, allowUav: true);
            using var rbOut = gpu.CreateReadbackBuffer(outBytes);
            try
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    ulong bytes = (ulong)inputs[i].Length * 4;
                    gpuIns[i] = gpu.CreateDefaultBuffer(bytes, allowUav: false);
                    upIns[i] = gpu.CreateUploadBuffer(bytes);
                    inputs[i].AsSpan().CopyTo(new Span<float>((float*)upIns[i].Map(), inputs[i].Length));
                    upIns[i].Unmap();
                }

                gpu.BeginCommands();
                for (int i = 0; i < inputs.Length; i++)
                {
                    gpu.Transition(gpuIns[i], D3D12_RESOURCE_STATE_COPY_DEST);
                    CopyBufferRegion(gpu.CommandList, gpuIns[i].Resource, 0, upIns[i].Resource, 0, (ulong)inputs[i].Length * 4);
                    gpu.Transition(gpuIns[i], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                }
                gpu.Transition(outGpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

                SetPipelineState(gpu.CommandList, pso);
                gpu.SetRootConstants(constants);
                for (int i = 0; i < inputs.Length; i++) gpu.SetSrv(i, gpuIns[i]);
                gpu.SetUav(0, outGpu);
                Dispatch(gpu.CommandList, gx, gy, 1);

                gpu.Transition(outGpu, D3D12_RESOURCE_STATE_COPY_SOURCE);
                CopyBufferRegion(gpu.CommandList, rbOut.Resource, 0, outGpu.Resource, 0, outBytes);
                gpu.ExecuteAndWait();

                var result = new float[outLen];
                new ReadOnlySpan<float>((float*)rbOut.Map(forRead: true), outLen).CopyTo(result);
                rbOut.Unmap(wasWritten: false);
                return result;
            }
            finally
            {
                foreach (var b in upIns) b?.Dispose();
                foreach (var b in gpuIns) b?.Dispose();
            }
        }

        public void Dispose() { SafeRelease(ref psoLn); SafeRelease(ref psoAtt); SafeRelease(ref psoGlu); SafeRelease(ref psoDw); }
    }

    private sealed class DecoderOpsGpuProbe : IDisposable
    {
        private readonly D3D12Device gpu;
        private nint psoLstm, psoGemv, psoArgMax;

        public DecoderOpsGpuProbe(D3D12Device gpu)
        {
            this.gpu = gpu;
            psoLstm = gpu.EmbeddedPipeline("DecoderOps.hlsl", "LstmLayerCondMain");
            psoGemv = gpu.EmbeddedPipeline("DecoderOps.hlsl", "Q8GemvCondMain");
            psoArgMax = gpu.EmbeddedPipeline("DecoderOps.hlsl", "ArgMaxAndStepMain");
        }

        public float[] Q8Gemv(float[] x, Q8Matrix w, float[]? bias = null)
        {
            int n = w.Rows, k = w.K;
            ulong xBytes = (ulong)k * 4, qBytes = (ulong)w.Q.Length, dBytes = (ulong)((w.D.Length + 1) / 2 * 4), bBytes = (ulong)((bias?.Length ?? 1) * 4), yBytes = (ulong)n * 4;
            using var xGpu = gpu.CreateDefaultBuffer(xBytes, allowUav: false);
            using var qGpu = gpu.CreateDefaultBuffer(qBytes, allowUav: false);
            using var dGpu = gpu.CreateDefaultBuffer(dBytes, allowUav: false);
            using var bGpu = gpu.CreateDefaultBuffer(bBytes, allowUav: false);
            using var yGpu = gpu.CreateDefaultBuffer(yBytes, allowUav: true);
            using var upX = gpu.CreateUploadBuffer(xBytes);
            using var upQ = gpu.CreateUploadBuffer(qBytes);
            using var upD = gpu.CreateUploadBuffer(dBytes);
            using var upB = gpu.CreateUploadBuffer(bBytes);
            using var rbY = gpu.CreateReadbackBuffer(yBytes);

            x.AsSpan(0, k).CopyTo(new Span<float>((float*)upX.Map(), k)); upX.Unmap();
            w.Q.AsSpan().CopyTo(new Span<sbyte>((sbyte*)upQ.Map(), w.Q.Length)); upQ.Unmap();
            var pD = (ushort*)upD.Map();
            for (int i = 0; i < w.D.Length; i++) pD[i] = BitConverter.HalfToUInt16Bits((Half)w.D[i]);
            if ((w.D.Length & 1) != 0) pD[w.D.Length] = 0;
            upD.Unmap();
            var pB = (float*)upB.Map();
            if (bias is not null) bias.AsSpan().CopyTo(new Span<float>(pB, bias.Length)); else pB[0] = 0f;
            upB.Unmap();

            gpu.BeginCommands();
            gpu.Transition(xGpu, D3D12_RESOURCE_STATE_COPY_DEST); gpu.Transition(qGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.Transition(dGpu, D3D12_RESOURCE_STATE_COPY_DEST); gpu.Transition(bGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            CopyBufferRegion(gpu.CommandList, xGpu.Resource, 0, upX.Resource, 0, xBytes);
            CopyBufferRegion(gpu.CommandList, qGpu.Resource, 0, upQ.Resource, 0, qBytes);
            CopyBufferRegion(gpu.CommandList, dGpu.Resource, 0, upD.Resource, 0, dBytes);
            CopyBufferRegion(gpu.CommandList, bGpu.Resource, 0, upB.Resource, 0, bBytes);
            gpu.Transition(xGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu.Transition(qGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(dGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu.Transition(bGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(yGpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            SetPipelineState(gpu.CommandList, psoGemv);
            Span<uint> c = [(uint)n, (uint)k, bias is not null ? 1u : 0u, 1u, 0u];
            gpu.SetRootConstants(c);
            gpu.SetSrv(0, xGpu); gpu.SetSrv(1, qGpu); gpu.SetSrv(2, dGpu); gpu.SetSrv(3, bGpu); gpu.SetSrv(4, bGpu);
            gpu.SetUav(0, yGpu);
            Dispatch(gpu.CommandList, (uint)((n + 15) / 16), 1, 1);

            gpu.Transition(yGpu, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CopyBufferRegion(gpu.CommandList, rbY.Resource, 0, yGpu.Resource, 0, yBytes);
            gpu.ExecuteAndWait();

            var result = new float[n];
            new ReadOnlySpan<float>((float*)rbY.Map(forRead: true), result.Length).CopyTo(result);
            rbY.Unmap(wasWritten: false);
            return result;
        }

        public (float[] hNew, float[] cNew) FusedLstmEmbed(float[] embed, float[] hPrev, float[] cPrev, Q8Matrix ih, Q8Matrix hh, float[] bias, int label, uint srcSlot)
        {
            int h = hPrev.Length;
            ulong qSingle = D3D12Device.Align256((ulong)ih.Q.Length);
            ulong dSingle = D3D12Device.Align256((ulong)((ih.D.Length + 1) / 2 * 4));
            ulong eBytes = (ulong)embed.Length * 4, qBytes = qSingle * 2, dBytes = dSingle * 2, bBytes = (ulong)bias.Length * 4;
            ulong hBytes = (ulong)(2 * h) * 4, cBytes = (ulong)h * 4;

            using var eGpu = gpu.CreateDefaultBuffer(eBytes, allowUav: false);
            using var qGpu = gpu.CreateDefaultBuffer(qBytes, allowUav: false);
            using var dGpu = gpu.CreateDefaultBuffer(dBytes, allowUav: false);
            using var bGpu = gpu.CreateDefaultBuffer(bBytes, allowUav: false);
            using var stGpu = gpu.CreateDefaultBuffer(256, allowUav: false);
            using var hGpu = gpu.CreateDefaultBuffer(hBytes, allowUav: true);
            using var cGpu = gpu.CreateDefaultBuffer(cBytes, allowUav: true);
            using var upE = gpu.CreateUploadBuffer(eBytes);
            using var upQ = gpu.CreateUploadBuffer(qBytes);
            using var upD = gpu.CreateUploadBuffer(dBytes);
            using var upB = gpu.CreateUploadBuffer(bBytes);
            using var upSt = gpu.CreateUploadBuffer(256);
            using var upH = gpu.CreateUploadBuffer(hBytes);
            using var upC = gpu.CreateUploadBuffer(cBytes);
            using var rbH = gpu.CreateReadbackBuffer(hBytes);
            using var rbC = gpu.CreateReadbackBuffer(cBytes);

            embed.AsSpan().CopyTo(new Span<float>((float*)upE.Map(), embed.Length)); upE.Unmap();
            var pQ = (byte*)upQ.Map();
            ih.Q.AsSpan().CopyTo(new Span<sbyte>((sbyte*)pQ, ih.Q.Length));
            hh.Q.AsSpan().CopyTo(new Span<sbyte>((sbyte*)(pQ + qSingle), hh.Q.Length));
            upQ.Unmap();
            var pD = (byte*)upD.Map();
            var pDi = (ushort*)pD;
            for (int i = 0; i < ih.D.Length; i++) pDi[i] = BitConverter.HalfToUInt16Bits((Half)ih.D[i]);
            var pDh = (ushort*)(pD + dSingle);
            for (int i = 0; i < hh.D.Length; i++) pDh[i] = BitConverter.HalfToUInt16Bits((Half)hh.D[i]);
            upD.Unmap();
            bias.AsSpan().CopyTo(new Span<float>((float*)upB.Map(), bias.Length)); upB.Unmap();
            var st = (uint*)upSt.Map();
            st[0] = 0; st[1] = 0; st[2] = 0; st[3] = (uint)label; st[4] = 1u; st[5] = srcSlot;
            upSt.Unmap();
            var pH = (float*)upH.Map();
            new Span<float>(pH, 2 * h).Clear();
            hPrev.AsSpan().CopyTo(new Span<float>(pH + (int)srcSlot * h, h));
            upH.Unmap();
            cPrev.AsSpan().CopyTo(new Span<float>((float*)upC.Map(), h)); upC.Unmap();

            gpu.BeginCommands();
            gpu.Transition(eGpu, D3D12_RESOURCE_STATE_COPY_DEST); gpu.Transition(qGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.Transition(dGpu, D3D12_RESOURCE_STATE_COPY_DEST); gpu.Transition(bGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.Transition(stGpu, D3D12_RESOURCE_STATE_COPY_DEST); gpu.Transition(hGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.Transition(cGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            CopyBufferRegion(gpu.CommandList, eGpu.Resource, 0, upE.Resource, 0, eBytes);
            CopyBufferRegion(gpu.CommandList, qGpu.Resource, 0, upQ.Resource, 0, qBytes);
            CopyBufferRegion(gpu.CommandList, dGpu.Resource, 0, upD.Resource, 0, dBytes);
            CopyBufferRegion(gpu.CommandList, bGpu.Resource, 0, upB.Resource, 0, bBytes);
            CopyBufferRegion(gpu.CommandList, stGpu.Resource, 0, upSt.Resource, 0, 32);
            CopyBufferRegion(gpu.CommandList, hGpu.Resource, 0, upH.Resource, 0, hBytes);
            CopyBufferRegion(gpu.CommandList, cGpu.Resource, 0, upC.Resource, 0, cBytes);
            gpu.Transition(eGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu.Transition(qGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(dGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu.Transition(bGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(stGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(hGpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); gpu.Transition(cGpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            SetPipelineState(gpu.CommandList, psoLstm);
            Span<uint> c = [(uint)h, 4u, 1u, (uint)qSingle, (uint)dSingle];
            gpu.SetRootConstants(c);
            gpu.SetSrv(0, eGpu); gpu.SetSrv(1, qGpu); gpu.SetSrv(2, dGpu); gpu.SetSrv(3, bGpu); gpu.SetSrv(4, stGpu);
            gpu.SetUav(0, hGpu); gpu.SetUav(1, cGpu);
            Dispatch(gpu.CommandList, (uint)((h + 7) / 8), 1, 1);

            gpu.Transition(hGpu, D3D12_RESOURCE_STATE_COPY_SOURCE); gpu.Transition(cGpu, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CopyBufferRegion(gpu.CommandList, rbH.Resource, 0, hGpu.Resource, 0, hBytes);
            CopyBufferRegion(gpu.CommandList, rbC.Resource, 0, cGpu.Resource, 0, cBytes);
            gpu.ExecuteAndWait();

            uint dstSlot = 1u - srcSlot;
            var hOut = new float[h];
            new ReadOnlySpan<float>((float*)rbH.Map(forRead: true) + (int)dstSlot * h, h).CopyTo(hOut);
            rbH.Unmap(wasWritten: false);
            var cOut = new float[h];
            new ReadOnlySpan<float>((float*)rbC.Map(forRead: true), h).CopyTo(cOut);
            rbC.Unmap(wasWritten: false);
            return (hOut, cOut);
        }

        public float[] Q8GemvFusedJointRelu(float[] encProj, float[] predProj, Q8Matrix w, float[] bias, int tFrame, int totalT)
        {
            int n = w.Rows, k = w.K;
            ulong eBytes = (ulong)encProj.Length * 4, pBytes = (ulong)predProj.Length * 4;
            ulong qBytes = (ulong)w.Q.Length, dBytes = (ulong)((w.D.Length + 1) / 2 * 4), bBytes = (ulong)bias.Length * 4, yBytes = (ulong)n * 4;
            using var eGpu = gpu.CreateDefaultBuffer(eBytes, allowUav: false);
            using var pGpu = gpu.CreateDefaultBuffer(pBytes, allowUav: false);
            using var qGpu = gpu.CreateDefaultBuffer(qBytes, allowUav: false);
            using var dGpu = gpu.CreateDefaultBuffer(dBytes, allowUav: false);
            using var bGpu = gpu.CreateDefaultBuffer(bBytes, allowUav: false);
            using var stGpu = gpu.CreateDefaultBuffer(256, allowUav: false);
            using var yGpu = gpu.CreateDefaultBuffer(yBytes, allowUav: true);
            using var upE = gpu.CreateUploadBuffer(eBytes);
            using var upP = gpu.CreateUploadBuffer(pBytes);
            using var upQ = gpu.CreateUploadBuffer(qBytes);
            using var upD = gpu.CreateUploadBuffer(dBytes);
            using var upB = gpu.CreateUploadBuffer(bBytes);
            using var upSt = gpu.CreateUploadBuffer(256);
            using var rbY = gpu.CreateReadbackBuffer(yBytes);

            encProj.AsSpan().CopyTo(new Span<float>((float*)upE.Map(), encProj.Length)); upE.Unmap();
            predProj.AsSpan().CopyTo(new Span<float>((float*)upP.Map(), predProj.Length)); upP.Unmap();
            w.Q.AsSpan().CopyTo(new Span<sbyte>((sbyte*)upQ.Map(), w.Q.Length)); upQ.Unmap();
            var pD = (ushort*)upD.Map();
            for (int i = 0; i < w.D.Length; i++) pD[i] = BitConverter.HalfToUInt16Bits((Half)w.D[i]);
            if ((w.D.Length & 1) != 0) pD[w.D.Length] = 0;
            upD.Unmap();
            bias.AsSpan().CopyTo(new Span<float>((float*)upB.Map(), bias.Length)); upB.Unmap();
            var st = (uint*)upSt.Map();
            st[0] = (uint)tFrame; st[1] = 0; st[2] = 0; st[3] = 0; st[4] = 0; st[5] = 0;
            upSt.Unmap();

            gpu.BeginCommands();
            gpu.Transition(eGpu, D3D12_RESOURCE_STATE_COPY_DEST); gpu.Transition(pGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.Transition(qGpu, D3D12_RESOURCE_STATE_COPY_DEST); gpu.Transition(dGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.Transition(bGpu, D3D12_RESOURCE_STATE_COPY_DEST); gpu.Transition(stGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            CopyBufferRegion(gpu.CommandList, eGpu.Resource, 0, upE.Resource, 0, eBytes);
            CopyBufferRegion(gpu.CommandList, pGpu.Resource, 0, upP.Resource, 0, pBytes);
            CopyBufferRegion(gpu.CommandList, qGpu.Resource, 0, upQ.Resource, 0, qBytes);
            CopyBufferRegion(gpu.CommandList, dGpu.Resource, 0, upD.Resource, 0, dBytes);
            CopyBufferRegion(gpu.CommandList, bGpu.Resource, 0, upB.Resource, 0, bBytes);
            CopyBufferRegion(gpu.CommandList, stGpu.Resource, 0, upSt.Resource, 0, 32);
            gpu.Transition(eGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu.Transition(pGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(qGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu.Transition(dGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(bGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu.Transition(stGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(yGpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            SetPipelineState(gpu.CommandList, psoGemv);
            Span<uint> c = [(uint)n, (uint)k, 1u, (uint)totalT, 4u];
            gpu.SetRootConstants(c);
            gpu.SetSrv(0, eGpu); gpu.SetSrv(1, qGpu); gpu.SetSrv(2, dGpu); gpu.SetSrv(3, bGpu); gpu.SetSrv(4, stGpu); gpu.SetSrv(5, pGpu);
            gpu.SetUav(0, yGpu);
            Dispatch(gpu.CommandList, (uint)((n + 15) / 16), 1, 1);

            gpu.Transition(yGpu, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CopyBufferRegion(gpu.CommandList, rbY.Resource, 0, yGpu.Resource, 0, yBytes);
            gpu.ExecuteAndWait();

            var result = new float[n];
            new ReadOnlySpan<float>((float*)rbY.Map(forRead: true), result.Length).CopyTo(result);
            rbY.Unmap(wasWritten: false);
            return result;
        }

        public (uint tok, uint tNew, uint syms, uint needPred, uint slot) ArgMaxStep(
            float[] logits, int vocab, int blank, int[] durations, int maxSymbols, uint initialNeedPred = 0u, uint initialSlot = 0u)
        {
            ulong inBytes = (ulong)logits.Length * 4;
            using var inGpu = gpu.CreateDefaultBuffer(inBytes, allowUav: false);
            using var stateGpu = gpu.CreateDefaultBuffer(256, allowUav: true);
            using var tokGpu = gpu.CreateDefaultBuffer(256, allowUav: true);
            using var upIn = gpu.CreateUploadBuffer(inBytes);
            using var upSt = gpu.CreateUploadBuffer(256);
            using var rbSt = gpu.CreateReadbackBuffer(256);
            using var rbTok = gpu.CreateReadbackBuffer(256);

            logits.AsSpan().CopyTo(new Span<float>((float*)upIn.Map(), logits.Length)); upIn.Unmap();
            var st = (uint*)upSt.Map();
            st[0] = 0; st[1] = 0; st[2] = 0; st[3] = 0xFFFFFFFFu; st[4] = initialNeedPred; st[5] = initialSlot;
            upSt.Unmap();

            uint durLo = 0, durHi = 0;
            for (int i = 0; i < durations.Length && i < 8; i++)
            {
                if (i < 4) durLo |= ((uint)durations[i] & 0xFFu) << (i * 8);
                else durHi |= ((uint)durations[i] & 0xFFu) << ((i - 4) * 8);
            }

            gpu.BeginCommands();
            gpu.Transition(inGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.Transition(stateGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            CopyBufferRegion(gpu.CommandList, inGpu.Resource, 0, upIn.Resource, 0, inBytes);
            CopyBufferRegion(gpu.CommandList, stateGpu.Resource, 0, upSt.Resource, 0, 32);
            gpu.Transition(inGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(stateGpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            gpu.Transition(tokGpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            SetPipelineState(gpu.CommandList, psoArgMax);
            Span<uint> c = [(uint)vocab, (uint)blank, (uint)durations.Length, 10u, (uint)maxSymbols, 64u, durLo, durHi];
            gpu.SetRootConstants(c);
            gpu.SetSrv(0, inGpu); gpu.SetUav(0, stateGpu); gpu.SetUav(1, tokGpu);
            Dispatch(gpu.CommandList, 1, 1, 1);

            gpu.Transition(stateGpu, D3D12_RESOURCE_STATE_COPY_SOURCE);
            gpu.Transition(tokGpu, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CopyBufferRegion(gpu.CommandList, rbSt.Resource, 0, stateGpu.Resource, 0, 32);
            CopyBufferRegion(gpu.CommandList, rbTok.Resource, 0, tokGpu.Resource, 0, 16);
            gpu.ExecuteAndWait();

            var rSt = (uint*)rbSt.Map(forRead: true);
            uint tNew = rSt[0], syms = rSt[1], count = rSt[2], needPred = rSt[4], slot = rSt[5];
            rbSt.Unmap(wasWritten: false);
            var rTok = (uint*)rbTok.Map(forRead: true);
            uint tok = count > 0 ? rTok[0] : (uint)blank;
            rbTok.Unmap(wasWritten: false);
            return (tok, tNew, syms, needPred, slot);
        }

        public void Dispose() { SafeRelease(ref psoLstm); SafeRelease(ref psoGemv); SafeRelease(ref psoArgMax); }
    }

    private sealed class MelOpsGpuProbe : IDisposable
    {
        private readonly D3D12Device gpu;
        private nint psoFftLog, psoNorm;

        public MelOpsGpuProbe(D3D12Device gpu)
        {
            this.gpu = gpu;
            psoFftLog = gpu.EmbeddedPipeline("MelOps.hlsl", "MelFftLogMain");
            psoNorm = gpu.EmbeddedPipeline("MelOps.hlsl", "MelNormalizeMain");
        }

        public float[] Compute(float[] audio, Mel mel, out int frames)
        {
            int n = audio.Length;
            frames = n / mel.Hop + 1;
            int valid = Math.Min(n / mel.Hop, frames);
            int nMels = mel.Mels;
            var twid = mel.BuildTwiddlesF32();

            ulong aBytes = (ulong)n * 4, wBytes = (ulong)mel.Window.Length * 4, fbBytes = (ulong)mel.Filterbank.Length * 4, tBytes = (ulong)twid.Length * 4;
            ulong fBytes = (ulong)(frames * nMels) * 4;
            using var aGpu = gpu.CreateDefaultBuffer(aBytes, allowUav: false);
            using var wGpu = gpu.CreateDefaultBuffer(wBytes, allowUav: false);
            using var fbGpu = gpu.CreateDefaultBuffer(fbBytes, allowUav: false);
            using var tGpu = gpu.CreateDefaultBuffer(tBytes, allowUav: false);
            using var fGpu = gpu.CreateDefaultBuffer(fBytes, allowUav: true);
            using var upA = gpu.CreateUploadBuffer(aBytes);
            using var upW = gpu.CreateUploadBuffer(wBytes);
            using var upFb = gpu.CreateUploadBuffer(fbBytes);
            using var upT = gpu.CreateUploadBuffer(tBytes);
            using var rbF = gpu.CreateReadbackBuffer(fBytes);

            audio.AsSpan().CopyTo(new Span<float>((float*)upA.Map(), n)); upA.Unmap();
            mel.Window.AsSpan().CopyTo(new Span<float>((float*)upW.Map(), mel.Window.Length)); upW.Unmap();
            mel.Filterbank.AsSpan().CopyTo(new Span<float>((float*)upFb.Map(), mel.Filterbank.Length)); upFb.Unmap();
            twid.AsSpan().CopyTo(new Span<float>((float*)upT.Map(), twid.Length)); upT.Unmap();

            gpu.BeginCommands();
            gpu.Transition(aGpu, D3D12_RESOURCE_STATE_COPY_DEST); gpu.Transition(wGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.Transition(fbGpu, D3D12_RESOURCE_STATE_COPY_DEST); gpu.Transition(tGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            CopyBufferRegion(gpu.CommandList, aGpu.Resource, 0, upA.Resource, 0, aBytes);
            CopyBufferRegion(gpu.CommandList, wGpu.Resource, 0, upW.Resource, 0, wBytes);
            CopyBufferRegion(gpu.CommandList, fbGpu.Resource, 0, upFb.Resource, 0, fbBytes);
            CopyBufferRegion(gpu.CommandList, tGpu.Resource, 0, upT.Resource, 0, tBytes);
            gpu.Transition(aGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu.Transition(wGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(fbGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu.Transition(tGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(fGpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            if (valid > 0)
            {
                SetPipelineState(gpu.CommandList, psoFftLog);
                int woff = (mel.NFft - mel.Win) / 2;
                Span<uint> cm = [(uint)n, (uint)mel.Hop, (uint)mel.Win, (uint)woff, (uint)nMels, BitConverter.SingleToUInt32Bits(mel.Preemph), (uint)valid];
                gpu.SetRootConstants(cm);
                gpu.SetSrv(0, aGpu); gpu.SetSrv(1, wGpu); gpu.SetSrv(2, fbGpu); gpu.SetSrv(3, tGpu);
                gpu.SetUav(0, fGpu);
                Dispatch(gpu.CommandList, (uint)valid, 1, 1);
                gpu.UavBarrier(fGpu);
            }
            SetPipelineState(gpu.CommandList, psoNorm);
            Span<uint> cn = [(uint)frames, (uint)valid, (uint)nMels];
            gpu.SetRootConstants(cn);
            gpu.SetUav(0, fGpu);
            Dispatch(gpu.CommandList, (uint)nMels, 1, 1);

            gpu.Transition(fGpu, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CopyBufferRegion(gpu.CommandList, rbF.Resource, 0, fGpu.Resource, 0, fBytes);
            gpu.ExecuteAndWait();

            var result = new float[frames * nMels];
            new ReadOnlySpan<float>((float*)rbF.Map(forRead: true), result.Length).CopyTo(result);
            rbF.Unmap(wasWritten: false);
            return result;
        }

        public void Dispose() { SafeRelease(ref psoFftLog); SafeRelease(ref psoNorm); }
    }

    private sealed class SubsampleOpsGpuProbe : IDisposable
    {
        private readonly D3D12Device gpu;
        private nint psoF32GemmRelu;

        public SubsampleOpsGpuProbe(D3D12Device gpu)
        {
            this.gpu = gpu;
            psoF32GemmRelu = gpu.EmbeddedPipeline("SubsampleOps.hlsl", "F32GemmReluMain");
        }

        public float[] F32GemmRelu(float[] x, float[] w, float[] bias, int m, int n, int k)
        {
            ulong xBytes = (ulong)(m * k) * 4, wBytes = (ulong)(n * k) * 4, bBytes = (ulong)n * 4, yBytes = (ulong)(m * n) * 4;
            using var xGpu = gpu.CreateDefaultBuffer(xBytes, allowUav: false);
            using var wGpu = gpu.CreateDefaultBuffer(wBytes, allowUav: false);
            using var bGpu = gpu.CreateDefaultBuffer(bBytes, allowUav: false);
            using var yGpu = gpu.CreateDefaultBuffer(yBytes, allowUav: true);
            using var upX = gpu.CreateUploadBuffer(xBytes);
            using var upW = gpu.CreateUploadBuffer(wBytes);
            using var upB = gpu.CreateUploadBuffer(bBytes);
            using var rbY = gpu.CreateReadbackBuffer(yBytes);

            x.AsSpan().CopyTo(new Span<float>((float*)upX.Map(), x.Length)); upX.Unmap();
            w.AsSpan().CopyTo(new Span<float>((float*)upW.Map(), w.Length)); upW.Unmap();
            bias.AsSpan().CopyTo(new Span<float>((float*)upB.Map(), bias.Length)); upB.Unmap();

            gpu.BeginCommands();
            gpu.Transition(xGpu, D3D12_RESOURCE_STATE_COPY_DEST); gpu.Transition(wGpu, D3D12_RESOURCE_STATE_COPY_DEST); gpu.Transition(bGpu, D3D12_RESOURCE_STATE_COPY_DEST);
            CopyBufferRegion(gpu.CommandList, xGpu.Resource, 0, upX.Resource, 0, xBytes);
            CopyBufferRegion(gpu.CommandList, wGpu.Resource, 0, upW.Resource, 0, wBytes);
            CopyBufferRegion(gpu.CommandList, bGpu.Resource, 0, upB.Resource, 0, bBytes);
            gpu.Transition(xGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu.Transition(wGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu.Transition(bGpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(yGpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            SetPipelineState(gpu.CommandList, psoF32GemmRelu);
            Span<uint> c = [(uint)m, (uint)n, (uint)k];
            gpu.SetRootConstants(c);
            gpu.SetSrv(0, xGpu); gpu.SetSrv(1, wGpu); gpu.SetSrv(2, bGpu);
            gpu.SetUav(0, yGpu);
            Dispatch(gpu.CommandList, (uint)((n + 63) / 64), (uint)((m + 31) / 32), 1);

            gpu.Transition(yGpu, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CopyBufferRegion(gpu.CommandList, rbY.Resource, 0, yGpu.Resource, 0, yBytes);
            gpu.ExecuteAndWait();

            var result = new float[m * n];
            new ReadOnlySpan<float>((float*)rbY.Map(forRead: true), result.Length).CopyTo(result);
            rbY.Unmap(wasWritten: false);
            return result;
        }

        public void Dispose() => SafeRelease(ref psoF32GemmRelu);
    }
}
