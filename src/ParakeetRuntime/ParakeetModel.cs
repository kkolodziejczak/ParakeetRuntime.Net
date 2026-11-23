// Part of ParakeetRuntime; a C# port of NeMo-Speech.cpp (Apache-2.0), see THIRD-PARTY-NOTICES.md.
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace ParakeetRuntime;

internal readonly record struct TranscribeProfile(double MelMs, double SubsampleMs, double EncoderMs, double DecodeMs)
{
    public double TotalMs => MelMs + SubsampleMs + EncoderMs + DecodeMs;
}

internal sealed unsafe class ParakeetModel : IDisposable
{
    readonly string modelPath;
    readonly Mel mel;
    readonly Int8Path int8Path;
    readonly int d, heads, dk, ff, kernel, subCh, layersN, predLayers;
    float[] pe = []; int peRows;
    float[] c0wByTap = [], c0b = [], c2wByTap = [], c2b = [], c3wT = [], c3b = [], c5wByTap = [], c5b = [], c6wT = [], c6b = [];
    Q8Weights? subOut;
    Layer[] layers = [];
    ParakeetGpu? gpu;
    bool cpuLoaded;
    float[] embed = []; readonly int predH, vocab, blank; readonly int[] durations; readonly int maxSymbols;
    Q8Weights[] ih = [], hh = []; Q8Weights? jointEnc, jointPred, jointOut;
    readonly string[] pieces;

    sealed class Layer
    {
        public float[] nFf1g = null!, nFf1b = null!, nConvg = null!, nConvb = null!, nAttg = null!, nAttb = null!, nFf2g = null!, nFf2b = null!, nOutg = null!, nOutb = null!;
        public Q8Weights ff1a = null!, ff1b = null!, ff2a = null!, ff2b = null!, pw1 = null!, pw2 = null!, qkv = null!, outp = null!, pos = null!;
        public float[] projectedPositionsUpTo20s = null!; public int projectedPositionsUpTo20sStride;
        public float[] dwByTap = null!, bnScale = null!, bnShift = null!, u = null!, v = null!;
    }

    public ParakeetModel(string path, bool allowGpu = true, string? adapterNameFilter = null)
        : this(path, Kernels.BestPath, allowGpu, adapterNameFilter) { }

    internal ParakeetModel(string path, Int8Path int8Path)
        : this(path, int8Path, allowGpu: false, adapterNameFilter: null) { }

    private ParakeetModel(string path, Int8Path int8Path, bool allowGpu, string? adapterNameFilter)
    {
        modelPath = path;
        this.int8Path = int8Path;
        using var g = new Gguf(path);
        mel = new Mel(g);
        d = g.Int("asr.encoder.d_model"); heads = g.Int("asr.encoder.n_heads"); dk = d / heads;
        ff = g.Int("asr.encoder.d_ff"); kernel = g.Int("asr.encoder.conv_kernel_size");
        subCh = g.Int("asr.encoder.subsampling_conv_channels"); layersN = g.Int("asr.encoder.n_layers");
        predH = g.Int("asr.rnnt.pred_hidden"); vocab = g.Int("asr.rnnt.vocab_size"); blank = g.Int("asr.rnnt.blank_id");
        maxSymbols = g.Int("asr.rnnt.max_symbols_per_step");
        durations = g.Ints("asr.tdt.durations");
        predLayers = g.Int("asr.rnnt.pred_num_layers");
        pieces = g.Strings("asr.tokenizer.vocab");

        gpu = allowGpu
            ? ParakeetGpu.TryCreate(g, mel, layersN, d, heads, ff, kernel, subCh, predH, predLayers, vocab, blank, maxSymbols, durations, adapterNameFilter)
            : null;
        if (gpu is not null)
        {
            peRows = gpu.PeRows;
        }
        else
        {
            LoadCpuWeights(g);
        }
    }

    private void LoadCpuWeights(Gguf g)
    {
        if (cpuLoaded)
            return;
        pe = g.Floats("encoder.pos_enc.pe"); peRows = pe.Length / d;
        c0wByTap = Transpose(g.Floats("encoder.pre_encode.conv.0.weight"), subCh); c0b = g.Floats("encoder.pre_encode.conv.0.bias");
        c2wByTap = Transpose(g.Floats("encoder.pre_encode.conv.2.weight"), subCh); c2b = g.Floats("encoder.pre_encode.conv.2.bias");
        c3wT = Transpose(g.Floats("encoder.pre_encode.conv.3.weight"), subCh); c3b = g.Floats("encoder.pre_encode.conv.3.bias");
        c5wByTap = Transpose(g.Floats("encoder.pre_encode.conv.5.weight"), subCh); c5b = g.Floats("encoder.pre_encode.conv.5.bias");
        c6wT = Transpose(g.Floats("encoder.pre_encode.conv.6.weight"), subCh); c6b = g.Floats("encoder.pre_encode.conv.6.bias");
        subOut = WithBias(g, "encoder.pre_encode.out");
        jointEnc = WithBias(g, "joint.enc"); jointPred = WithBias(g, "joint.pred"); jointOut = WithBias(g, "joint.joint_net.2");
        layers = new Layer[layersN];
        for (int i = 0; i < layersN; i++)
        {
            string p = $"encoder.layers.{i}.";
            var l = layers[i] = new Layer
            {
                nFf1g = g.Floats(p + "norm_feed_forward1.weight"), nFf1b = g.Floats(p + "norm_feed_forward1.bias"),
                nConvg = g.Floats(p + "norm_conv.weight"), nConvb = g.Floats(p + "norm_conv.bias"),
                nAttg = g.Floats(p + "norm_self_att.weight"), nAttb = g.Floats(p + "norm_self_att.bias"),
                nFf2g = g.Floats(p + "norm_feed_forward2.weight"), nFf2b = g.Floats(p + "norm_feed_forward2.bias"),
                nOutg = g.Floats(p + "norm_out.weight"), nOutb = g.Floats(p + "norm_out.bias"),
                ff1a = Packed(g, p + "feed_forward1.linear1.weight"), ff1b = Packed(g, p + "feed_forward1.linear2.weight"),
                ff2a = Packed(g, p + "feed_forward2.linear1.weight"), ff2b = Packed(g, p + "feed_forward2.linear2.weight"),
                pw1 = Packed(g, p + "conv.pointwise_conv1.weight"), pw2 = Packed(g, p + "conv.pointwise_conv2.weight"),
                qkv = Kernels.Pack(g.Q8(p + "self_attn.linear_q.weight", p + "self_attn.linear_k.weight", p + "self_attn.linear_v.weight"), null, int8Path),
                outp = Packed(g, p + "self_attn.linear_out.weight"), pos = Packed(g, p + "self_attn.linear_pos.weight"),
                u = g.Floats(p + "self_attn.pos_bias_u"), v = g.Floats(p + "self_attn.pos_bias_v"),
            };
            l.dwByTap = Transpose(g.Floats(p + "conv.depthwise_conv.weight"), d);
            var w = g.Floats(p + "conv.batch_norm.weight"); var b = g.Floats(p + "conv.batch_norm.bias");
            var mean = g.Floats(p + "conv.batch_norm.running_mean"); var var = g.Floats(p + "conv.batch_norm.running_var");
            l.bnScale = new float[d]; l.bnShift = new float[d];
            for (int c = 0; c < d; c++) { l.bnScale[c] = w[c] / MathF.Sqrt(var[c] + 1e-5f); l.bnShift[c] = b[c] - mean[c] * l.bnScale[c]; }
            (l.projectedPositionsUpTo20s, l.projectedPositionsUpTo20sStride) = ProjectPositions(l, TwentySecondsOfEncoderFrames);
        }
        embed = g.Floats("decoder.prediction.embed.weight");
        ih = new Q8Weights[predLayers]; hh = new Q8Weights[predLayers];
        for (int i = 0; i < predLayers; i++)
        {
            ih[i] = WithBias(g, $"decoder.prediction.dec_rnn.lstm.ih_l{i}");
            hh[i] = WithBias(g, $"decoder.prediction.dec_rnn.lstm.hh_l{i}");
        }
        cpuLoaded = true;
    }

    private void RetireGpuAndContinueOnCpu()
    {
        gpu?.Dispose();
        gpu = null;
        if (!cpuLoaded)
        {
            using var g = new Gguf(modelPath);
            LoadCpuWeights(g);
        }
    }

    static float[] Transpose(float[] a, int rows) => Kernels.Transpose(a, rows, a.Length / rows, rows);

    Q8Weights Packed(Gguf g, string name) => Kernels.Pack(g.Q8(name), null, int8Path);

    Q8Weights WithBias(Gguf g, string name) => Kernels.Pack(g.Q8(name + ".weight"), g.Floats(name + ".bias"), int8Path);

    const int SubsamplingFactor = 8;
    const double MelFramesPerSecond = 100.0;

    public TimeSpan MaxAudio => TimeSpan.FromSeconds((peRows + 1) / 2 * SubsamplingFactor / MelFramesPerSecond - 1);

    public bool IsUsingGpu => gpu is not null;

    public string BackendName => gpu is not null ? $"GPU: {gpu.AdapterName} (D3D12)" : $"CPU ({int8Path})";

    public long GpuMemoryUsedBytes => gpu?.GpuMemoryUsedBytes ?? 0;

    internal StageClock? BenchmarkClock { get; set; }

    public void Dispose()
    {
        gpu?.Dispose();
        gpu = null;
    }

    public string Transcribe(float[] audio16KhzMono) => Transcribe(audio16KhzMono, out _);

    public string Transcribe(float[] audio16KhzMono, out TranscribeProfile profile)
    {
        if (audio16KhzMono.Length == 0)
        {
            profile = default;
            return string.Empty;
        }
        if (audio16KhzMono.Length > MaxAudio.TotalSeconds * 16000)
            throw new ArgumentException($"Audio longer than {MaxAudio.TotalSeconds:F0} s cannot be transcribed in one piece.", nameof(audio16KhzMono));
        var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        double toMs = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        if (gpu is not null)
        {
            try
            {
                if (gpu.CanRunGpuMel)
                {
                    var gpuTokens = gpu.TranscribeAudio(audio16KhzMono, out double encMs, out double decMs);
                    var gpuText = Detokenize(gpuTokens);
                    profile = new TranscribeProfile(0, 0, encMs, decMs);
                    return gpuText;
                }
                var gpuFeats = mel.Compute(audio16KhzMono, out int gpuFrames);
                var tMel = System.Diagnostics.Stopwatch.GetTimestamp();
                var fallbackMelTokens = gpu.TranscribeFromMel(gpuFeats, gpuFrames, out double fEncMs, out double fDecMs);
                var fallbackMelText = Detokenize(fallbackMelTokens);
                profile = new TranscribeProfile((tMel - t0) * toMs, 0, fEncMs, fDecMs);
                return fallbackMelText;
            }
            catch (Exception ex) when (IsGpuDeviceLost(ex))
            {
                RetireGpuAndContinueOnCpu();
                t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            }
        }
        BenchmarkClock?.Start();
        var feats = mel.Compute(audio16KhzMono, out int frames);
        BenchmarkClock?.Lap("mel");
        var t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        var x = SubsampleDwStriding(feats, frames, out int T);
        BenchmarkClock?.Lap("subsampling");
        var t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        using (var scratch = new EncoderScratch())
        {
            foreach (var l in layers)
            {
                scratch.Rewind();
                var next = ConformerLayer(l, x, T);
                Array.Copy(next, x, x.Length);
            }
        }
        var t3 = System.Diagnostics.Stopwatch.GetTimestamp();
        var text = Detokenize(DecodeTdtGreedy(x, T));
        BenchmarkClock?.Lap("decoder");
        var t4 = System.Diagnostics.Stopwatch.GetTimestamp();
        profile = new TranscribeProfile((t1 - t0) * toMs, (t2 - t1) * toMs, (t3 - t2) * toMs, (t4 - t3) * toMs);
        return text;
    }

    static bool IsGpuDeviceLost(Exception ex) => ex is COMException or InvalidOperationException or OutOfMemoryException;

    float[] SubsampleDwStriding(float[] feats, int frames, out int T)
    {
        int W0 = mel.Mels, C = subCh;
        int H1 = (frames - 1) / 2 + 1, W1 = (W0 - 1) / 2 + 1, H2 = (H1 - 1) / 2 + 1, W2 = (W1 - 1) / 2 + 1;
        var y = new float[H2 * W2 * C];
        Kernels.ForRanges(H2, (start, end) =>
        {
            var conv0RowsOneDepthwiseRowReads = new float[3 * W1 * C];
            float** input = stackalloc float*[3];
            fixed (float* fp = feats, rp = conv0RowsOneDepthwiseRowReads, yp = y, w0 = c0wByTap, b0 = c0b, w2 = c2wByTap, b2 = c2b)
                for (int h = start; h < end; h++)
                {
                    for (int kh = 0; kh < 3; kh++)
                    {
                        int r = 2 * h - 1 + kh;
                        input[kh] = r >= 0 && r < H1 ? rp + kh * W1 * C : null;
                        if (input[kh] != null) Conv0Row(fp, frames, W0, r, w0, b0, input[kh], W1);
                    }
                    DepthwiseRow(input, W1, W2, w2, b2, yp + h * W2 * C);
                }
        });
        var z = Kernels.GemmRelu(y, H2 * W2, c3wT, C, C, c3b);
        int H3 = (H2 - 1) / 2 + 1, W3 = (W2 - 1) / 2 + 1;
        var y2 = new float[H3 * W3 * C];
        Kernels.For(H3, h =>
        {
            float** input = stackalloc float*[3];
            fixed (float* zp = z, yp = y2, w5 = c5wByTap, b5 = c5b)
            {
                for (int kh = 0; kh < 3; kh++) { int r = 2 * h - 1 + kh; input[kh] = r >= 0 && r < H2 ? zp + r * W2 * C : null; }
                DepthwiseRow(input, W2, W3, w5, b5, yp + h * W3 * C);
            }
        });
        var a = Kernels.GemmRelu(y2, H3 * W3, c6wT, C, C, c6b);
        T = H3;
        var channelMajorFrames = new float[T * C * W3];
        for (int t = 0; t < T; t++) for (int w = 0; w < W3; w++) for (int c = 0; c < C; c++) channelMajorFrames[t * C * W3 + c * W3 + w] = a[(t * W3 + w) * C + c];
        return Kernels.Linear(channelMajorFrames, T, subOut!);
    }

    void Conv0Row(float* feats, int H, int W, int r, float* taps, float* bias, float* output, int W1)
    {
        int C = subCh;
        for (int w = 0; w < W1; w++)
            for (int c = 0; c < C; c += Vector<float>.Count)
            {
                var s = Vector.Load(bias + c);
                for (int kh = 0; kh < 3; kh++)
                {
                    int ih = 2 * r - 1 + kh; if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < 3; kw++)
                    {
                        int iw = 2 * w - 1 + kw; if (iw < 0 || iw >= W) continue;
                        s += new Vector<float>(feats[ih * W + iw]) * Vector.Load(taps + (kh * 3 + kw) * C + c);
                    }
                }
                Vector.Max(s, Vector<float>.Zero).Store(output + w * C + c);
            }
    }

    void DepthwiseRow(float** inputRowsNullForPadding, int W, int Wout, float* taps, float* bias, float* output)
    {
        int C = subCh;
        for (int w = 0; w < Wout; w++)
            for (int c = 0; c < C; c += Vector<float>.Count)
            {
                var s = Vector.Load(bias + c);
                for (int kh = 0; kh < 3; kh++)
                {
                    if (inputRowsNullForPadding[kh] == null) continue;
                    for (int kw = 0; kw < 3; kw++)
                    {
                        int iw = 2 * w - 1 + kw; if (iw < 0 || iw >= W) continue;
                        s += Vector.Load(inputRowsNullForPadding[kh] + iw * C + c) * Vector.Load(taps + (kh * 3 + kw) * C + c);
                    }
                }
                s.Store(output + w * C + c);
            }
    }

    const float HalfStepFeedForward = 0.5f;

    float[] ConformerLayer(Layer l, float[] x, int T)
    {
        var h = Kernels.Linear(Kernels.LayerNorm(x, T, d, l.nFf1g, l.nFf1b), T, l.ff1a, Activation.Silu);
        Kernels.LinearAdd(h, T, l.ff1b, HalfStepFeedForward, x);
        BenchmarkClock?.Lap("feed-forward");

        AddRelativePositionAttention(l, Kernels.LayerNorm(x, T, d, l.nAttg, l.nAttb), T, x);
        BenchmarkClock?.Lap("attention");

        var c = Kernels.Linear(Kernels.LayerNorm(x, T, d, l.nConvg, l.nConvb), T, l.pw1);
        var dwo = DepthwiseConvolutionBatchNormSilu(l, Glu(c, T), T);
        Kernels.LinearAdd(dwo, T, l.pw2, 1f, x);
        BenchmarkClock?.Lap("convolution");

        h = Kernels.Linear(Kernels.LayerNorm(x, T, d, l.nFf2g, l.nFf2b), T, l.ff2a, Activation.Silu);
        Kernels.LinearAdd(h, T, l.ff2b, HalfStepFeedForward, x);
        BenchmarkClock?.Lap("feed-forward");
        var o = Kernels.LayerNorm(x, T, d, l.nOutg, l.nOutb);
        BenchmarkClock?.Lap("output norm");
        return o;
    }

    float[] Glu(float[] c, int T)
    {
        var glu = EncoderScratch.Rent(T * d);
        Kernels.For(T, t =>
        {
            fixed (float* cp = c, gp = glu)
                for (int ch = 0; ch < d; ch += Vector<float>.Count)
                {
                    var a = Vector.Load(cp + t * 2 * d + ch); var gate = Vector.Load(cp + t * 2 * d + d + ch);
                    (a / (Vector<float>.One + Vector.Exp(-gate))).Store(gp + t * d + ch);
                }
        });
        return glu;
    }

    float[] DepthwiseConvolutionBatchNormSilu(Layer l, float[] glu, int T)
    {
        var dwo = EncoderScratch.Rent(T * d); int pad = (kernel - 1) / 2;
        Kernels.For(T, t =>
        {
            fixed (float* gp = glu, op = dwo, wp = l.dwByTap, sp = l.bnScale, hp = l.bnShift)
                for (int ch = 0; ch < d; ch += Vector<float>.Count)
                {
                    var s = Vector<float>.Zero;
                    for (int k = Math.Max(0, pad - t); k < Math.Min(kernel, T + pad - t); k++)
                        s += Vector.Load(gp + (t + k - pad) * d + ch) * Vector.Load(wp + k * d + ch);
                    var y = s * Vector.Load(sp + ch) + Vector.Load(hp + ch);
                    (y / (Vector<float>.One + Vector.Exp(-y))).Store(op + t * d + ch);
                }
        });
        return dwo;
    }

    const int TwentySecondsOfEncoderFrames = 251;

    (float[] Table, int Stride) ProjectPositions(Layer l, int T)
    {
        int start = (peRows + 1) / 2 - T, P = 2 * T - 1;
        var pos = EncoderScratch.Rent(P * d);
        Array.Copy(pe, (long)start * d, pos, 0, (long)P * d);
        return Kernels.TransposePositions(Kernels.Linear(pos, P, l.pos), P, d);
    }

    void AddRelativePositionAttention(Layer l, float[] x, int T, float[] residual)
    {
        var qkv = Kernels.Linear(x, T, l.qkv);
        var ((positions, stride), columnOfPositionTMinus1) = T <= TwentySecondsOfEncoderFrames
            ? ((l.projectedPositionsUpTo20s, l.projectedPositionsUpTo20sStride), TwentySecondsOfEncoderFrames - T)
            : (ProjectPositions(l, T), 0);
        var ctx = EncoderScratch.Rent(T * d);
        Kernels.RelativeAttention(qkv, T, d, heads, l.u, l.v, positions, stride, columnOfPositionTMinus1, ctx);
        Kernels.LinearAdd(ctx, T, l.outp, 1f, residual);
    }

    List<int> DecodeTdtGreedy(float[] enc, int T)
    {
        var encProj = Kernels.Linear(enc, T, jointEnc!);
        int J = jointEnc!.Rows;
        int nl = ih.Length;
        var h = new float[nl][]; var c = new float[nl][];
        for (int i = 0; i < nl; i++) { h[i] = new float[predH]; c[i] = new float[predH]; }
        var tokens = new List<int>();
        int last = -1;
        var (g, hN, cN) = Predict(last, h, c);
        var predProj = Kernels.Linear(g, 1, jointPred!);
        int t = 0;
        while (t < T)
        {
            int symbols = 0, skip = 0; bool loop = true;
            while (loop && symbols < maxSymbols)
            {
                var z = new float[J];
                for (int k = 0; k < J; k++) z[k] = MathF.Max(0, encProj[t * J + k] + predProj[k]);
                var logits = Kernels.Linear(z, 1, jointOut!);
                int tok = ArgMax(logits, 0, vocab), di = ArgMax(logits, vocab, durations.Length);
                skip = durations[di];
                if (tok == blank && skip == 0) skip = 1;
                if (tok != blank)
                {
                    tokens.Add(tok); last = tok; h = hN; c = cN;
                    (g, hN, cN) = Predict(last, h, c);
                    predProj = Kernels.Linear(g, 1, jointPred!);
                }
                symbols++;
                t += skip;
                loop = skip == 0;
                if (t >= T) break;
            }
            if (symbols == maxSymbols && skip == 0) t++;
        }
        return tokens;
    }

    (float[] g, float[][] h, float[][] c) Predict(int label, float[][] h, float[][] c)
    {
        var x = new float[predH];
        if (label >= 0) Array.Copy(embed, (long)label * predH, x, 0, predH);
        var nh = new float[h.Length][]; var nc = new float[h.Length][];
        for (int i = 0; i < h.Length; i++)
        {
            var gi = Kernels.Linear(x, 1, ih[i]); var gh = Kernels.Linear(h[i], 1, hh[i]);
            nh[i] = new float[predH]; nc[i] = new float[predH];
            for (int k = 0; k < predH; k++)
            {
                float ig = Kernels.Sigmoid(gi[k] + gh[k]), fg = Kernels.Sigmoid(gi[predH + k] + gh[predH + k]);
                float gg = MathF.Tanh(gi[2 * predH + k] + gh[2 * predH + k]), og = Kernels.Sigmoid(gi[3 * predH + k] + gh[3 * predH + k]);
                nc[i][k] = fg * c[i][k] + ig * gg; nh[i][k] = og * MathF.Tanh(nc[i][k]);
            }
            x = nh[i];
        }
        return (x, nh, nc);
    }

    static int ArgMax(float[] a, int from, int n) { int best = from; for (int i = from + 1; i < from + n; i++) if (a[i] > a[best]) best = i; return best - from; }

    string Detokenize(List<int> tokens)
    {
        var bytes = new List<byte>();
        foreach (var t in tokens)
        {
            var piece = pieces[t];
            if (piece.Length == 6 && piece.StartsWith("<0x") && piece.EndsWith('>')) { bytes.Add(Convert.ToByte(piece.Substring(3, 2), 16)); continue; }
            bytes.AddRange(Encoding.UTF8.GetBytes(piece.Replace('▁', ' ')));
        }
        return SanitizePunctuationSpacing(Encoding.UTF8.GetString(bytes.ToArray()).Trim());
    }

    static string SanitizePunctuationSpacing(string s)
    {
        const string left = ".,?!;:%)]}-", right = "([{-";
        var o = new StringBuilder();
        for (int i = 0; i < s.Length; i++)
        {
            char ch = s[i];
            if (left.Contains(ch)) while (o.Length > 0 && o[^1] == ' ') o.Length--;
            o.Append(ch);
            if (right.Contains(ch)) while (i + 1 < s.Length && s[i + 1] == ' ') i++;
        }
        return o.ToString();
    }
}
