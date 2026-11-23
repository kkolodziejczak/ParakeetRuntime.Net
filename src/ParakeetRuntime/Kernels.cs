// Part of the Parakeet engine; a C# port of NeMo-Speech.cpp (Apache-2.0), see THIRD-PARTY-NOTICES.md.
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace ParakeetRuntime;

internal static unsafe class Kernels
{
    // One thread per fast core the process may run on, at most 8 (see ThreadBudget).
    static int Threads = ThreadBudget.ForThisProcess();
    static ParallelOptions Opt = new() { MaxDegreeOfParallelism = Threads };

    internal static int ThreadCount => Threads;

    // For the Spike's --threads, to measure other budgets; set before any model runs.
    internal static void UseThreads(int threads)
    {
        Threads = threads;
        Opt = new() { MaxDegreeOfParallelism = threads };
    }

    // The fast paths need AVX2 and FMA. Not every Windows 11 CPU has them (for example the
    // Pentium Silver N6000), so 128-bit vectors stand in: slower, same results within rounding.
    static readonly bool Simd = Avx2.IsSupported && Fma.IsSupported;

    // The fastest int8 product this CPU has; the weights are packed for it once, at load.
    public static readonly Int8Path BestPath =
        Supports(Int8Path.AvxVnniInt8) ? Int8Path.AvxVnniInt8
        : Supports(Int8Path.AvxVnni) ? Int8Path.AvxVnni
        : Supports(Int8Path.Avx2) ? Int8Path.Avx2
        : Int8Path.Portable;

    public static bool Supports(Int8Path path) => path switch
    {
        Int8Path.Portable => true,
        Int8Path.Avx2 => Simd,
        Int8Path.AvxVnni => Simd && AvxVnni.IsSupported,
        Int8Path.AvxVnniInt8 => Simd && AvxVnniInt8.IsSupported,
        _ => false,
    };

    // Quantize X[M,K] per 32-value block exactly as NeMo-Speech.cpp does on x86 (ggml's AVX2
    // quantize_row_q8_0, arch/x86/quants.c): scale amax / 127 rounded to fp16, values times
    // 127 / amax rounded half to even. A block whose scale rounds to zero is silent: all zeros
    // (127 / amax could overflow there, and the values count for nothing anyway). A block holding
    // NaN or infinity gets a NaN or infinite scale and all zeros: what it multiplies is lost anyway.
    public static Q8Activations Quantize(float[] x, int m, int k) => Quantize(x, m, k, avx2: Simd);

    // avx2: false takes the 128-bit path of CPUs without AVX2, for tests on CPUs with it.
    internal static Q8Activations Quantize(float[] x, int m, int k, bool avx2)
    {
        if (avx2 && !Simd)
            throw new PlatformNotSupportedException("This CPU has no AVX2 and FMA.");
        // The inverse scales of a row live on the stack: 4 KB at this width, 512 bytes at the model's widest.
        ArgumentOutOfRangeException.ThrowIfGreaterThan(k, 32 * 1024);
        var q = EncoderScratch.Quant(m, k);
        Parallel.For(0, m, Opt, r =>
        {
            int blocks = q.Blocks;
            float* inverses = stackalloc float[blocks];
            fixed (float* xp = x, d = q.D) fixed (sbyte* qp = q.Q) fixed (int* sums = q.Sums)
            {
                float* row = xp + (long)r * k, scales = d + r * blocks;
                // Three passes over the row, the fp16 rounding between the vector passes: .NET's
                // Half conversion called while 256-bit registers were live made quantizing about
                // four times slower (measured 2026-11-22 on a Core Ultra 7 270K Plus).
                if (avx2) LargestMagnitudes256(row, blocks, scales); else LargestMagnitudes128(row, blocks, scales);
                for (int b = 0; b < blocks; b++)
                {
                    float amax = scales[b], scale = (float)(Half)(amax / 127f);
                    scales[b] = scale;
                    // 0 marks a block of zeros; any other block's inverse is at least 127 / float.MaxValue.
                    inverses[b] = float.IsFinite(amax) && scale != 0 ? 127f / amax : 0f;
                }
                if (avx2) Round256(row, blocks, inverses, qp + (long)r * k, sums + r * blocks);
                else Round128(row, blocks, inverses, qp + (long)r * k, sums + r * blocks);
            }
        });
        return q;
    }

    // The largest magnitude of each block of 32, compared as integers: without their sign bits,
    // floats order like their bits, infinity above every number and NaN above infinity, so a NaN
    // anywhere in the block becomes its amax. Vector256.Max would spend 9 instructions per step on
    // NaN and signed-zero rules; vmaxps would keep or drop a NaN by its position. Not inlined,
    // like Round256, so that the 256-bit registers are cleared before Quantize's Half conversions.
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void LargestMagnitudes256(float* x, int blocks, float* largest)
    {
        var magnitude = Vector256.Create(int.MaxValue);
        int* bits = (int*)x;
        for (int b = 0; b < blocks; b++, bits += 32)
        {
            var m = Avx2.Max(Avx2.Max(Avx2.And(Avx.LoadVector256(bits), magnitude), Avx2.And(Avx.LoadVector256(bits + 8), magnitude)),
                             Avx2.Max(Avx2.And(Avx.LoadVector256(bits + 16), magnitude), Avx2.And(Avx.LoadVector256(bits + 24), magnitude)));
            var half = Sse41.Max(m.GetLower(), m.GetUpper());
            half = Sse41.Max(half, Sse2.Shuffle(half, 0b_01_00_11_10));
            largest[b] = Sse41.Max(half, Sse2.Shuffle(half, 0b_10_11_00_01)).AsSingle().ToScalar();
        }
    }

    // Each block's values times its inverse scale, rounded half to even, and their sum; an inverse
    // of 0 marks a block of zeros. The packs saturate, but the values never exceed 127.
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void Round256(float* x, int blocks, float* inverses, sbyte* q, int* sums)
    {
        var order = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
        for (int b = 0; b < blocks; b++, x += 32, q += 32)
        {
            if (inverses[b] == 0)
            {
                Vector256<sbyte>.Zero.Store(q);
                sums[b] = 0;
                continue;
            }
            var inverse = Vector256.Create(inverses[b]);
            Vector256<int> ia = RoundToInt(Avx.LoadVector256(x), inverse), ib = RoundToInt(Avx.LoadVector256(x + 8), inverse);
            Vector256<int> ic = RoundToInt(Avx.LoadVector256(x + 16), inverse), ie = RoundToInt(Avx.LoadVector256(x + 24), inverse);
            // The packs work per 128-bit half, leaving the 4-value groups as a0 b0 c0 e0 | a1 b1 c1 e1.
            var packed = Avx2.PackSignedSaturate(Avx2.PackSignedSaturate(ia, ib), Avx2.PackSignedSaturate(ic, ie));
            Avx2.PermuteVar8x32(packed.AsInt32(), order).AsSByte().Store(q);
            sums[b] = Vector256.Sum(Avx2.Add(Avx2.Add(ia, ib), Avx2.Add(ic, ie)));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<int> RoundToInt(Vector256<float> x, Vector256<float> inverse) =>
        Avx.ConvertToVector256Int32WithTruncation(Avx.RoundToNearestInteger(Avx.Multiply(x, inverse)));

    static void LargestMagnitudes128(float* x, int blocks, float* largest)
    {
        for (int b = 0; b < blocks; b++, x += 32)
        {
            var m = Vector128.Max(Vector128.Max(Vector128.Abs(Vector128.Load(x)), Vector128.Abs(Vector128.Load(x + 4))),
                                  Vector128.Max(Vector128.Abs(Vector128.Load(x + 8)), Vector128.Abs(Vector128.Load(x + 12))));
            for (int i = 16; i < 32; i += 8)
                m = Vector128.Max(m, Vector128.Max(Vector128.Abs(Vector128.Load(x + i)), Vector128.Abs(Vector128.Load(x + i + 4))));
            largest[b] = MathF.Max(MathF.Max(m[0], m[1]), MathF.Max(m[2], m[3]));
        }
    }

    static void Round128(float* x, int blocks, float* inverses, sbyte* q, int* sums)
    {
        for (int block = 0; block < blocks; block++, x += 32, q += 32)
        {
            float inverse = inverses[block];
            if (inverse == 0)
            {
                new Span<sbyte>(q, 32).Clear();
                sums[block] = 0;
                continue;
            }
            var sum = Vector128<int>.Zero;
            for (int i = 0; i < 32; i += 16)
            {
                var a = RoundBlock(Vector128.Load(x + i), inverse); var b = RoundBlock(Vector128.Load(x + i + 4), inverse);
                var c = RoundBlock(Vector128.Load(x + i + 8), inverse); var e = RoundBlock(Vector128.Load(x + i + 12), inverse);
                Vector128.Narrow(Vector128.Narrow(a, b), Vector128.Narrow(c, e)).Store(q + i);
                sum += a + b + c + e;
            }
            sums[block] = Vector128.Sum(sum);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<int> RoundBlock(Vector128<float> x, float inverse)
    {
        return Vector128.ConvertToInt32(Vector128.Round(x * inverse));
    }

    // Packs a weight matrix (and its bias) for one int8 path, taking over m's arrays: m is not
    // usable afterwards. The AVX2 paths interleave 8 output rows 4 bytes at a time: one 256-bit
    // multiply-add then gives 8 dot products of 4 values each, so a block of 32 needs 8 of them
    // and a single float scaling for 8 outputs. Each group of 8 rows keeps its place in the
    // arrays, so packing needs no second copy; a last partial group is padded with zero scales.
    // The portable path keeps plain rows.
    public static Q8Weights Pack(Q8Matrix m, float[]? bias, Int8Path path)
    {
        if (!Supports(path))
            throw new PlatformNotSupportedException($"This CPU has no {path} int8 product.");
        if (path == Int8Path.Portable)
            return new Q8Weights(path, m.Rows, m.K, m.Q, m.D, bias);
        int groups = (m.Rows + 7) / 8, blocks = m.K / 32;
        var q = m.Rows % 8 == 0 ? m.Q : new sbyte[checked(groups * 8 * m.K)];
        var d = m.Rows % 8 == 0 ? m.D : new float[groups * 8 * blocks];
        var rows = new sbyte[8 * m.K]; var scales = new float[8 * blocks];
        for (int g = 0; g < groups; g++)
        {
            int count = Math.Min(8, m.Rows - g * 8);
            Array.Clear(rows); Array.Clear(scales);
            Array.Copy(m.Q, (long)g * 8 * m.K, rows, 0, count * m.K);
            Array.Copy(m.D, g * 8 * blocks, scales, 0, count * blocks);
            for (int j = 0; j < 8; j++)
                for (int b = 0; b < blocks; b++)
                {
                    d[(g * blocks + b) * 8 + j] = scales[j * blocks + b];
                    for (int c = 0; c < 8; c++)
                        for (int e = 0; e < 4; e++)
                            q[(long)g * 8 * m.K + b * 256 + c * 32 + j * 4 + e] = rows[j * m.K + b * 32 + c * 4 + e];
                }
        }
        float[]? padded = null;
        if (bias is not null) { padded = new float[groups * 8]; bias.CopyTo(padded, 0); }
        return new Q8Weights(path, m.Rows, m.K, q, d, padded);
    }

    // Y[M,N] = act(X[M,K] * W[N,K]^T + bias), X given as float and quantized here.
    public static float[] Linear(float[] x, int m, Q8Weights w, Activation activation = Activation.None)
    {
        var y = EncoderScratch.Rent(checked(m * w.Rows));
        Product(Quantize(x, m, w.K), w, y, new Epilogue(activation, Add: false, Scale: 1));
        return y;
    }

    // target[M,N] += scale * (X[M,K] * W[N,K]^T + bias): a residual step without a buffer for Y.
    public static void LinearAdd(float[] x, int m, Q8Weights w, float scale, float[] target) =>
        Product(Quantize(x, m, w.K), w, target, new Epilogue(Activation.None, Add: true, Scale: scale));

    static void Product(Q8Activations xq, Q8Weights w, float[] y, Epilogue e)
    {
        switch (w.Path)
        {
            case Int8Path.AvxVnniInt8: PackedQ8Gemm<VnniInt8Dot>(xq, w, y, e); break;
            case Int8Path.AvxVnni: PackedQ8Gemm<VnniDot>(xq, w, y, e); break;
            case Int8Path.Avx2: PackedQ8Gemm<Avx2Dot>(xq, w, y, e); break;
            default: PortableQ8Gemm(xq, w, y, e); break;
        }
    }

    // What happens to each output after the product and bias, done while it is still in a register.
    readonly record struct Epilogue(Activation Activation, bool Add, float Scale)
    {
        public float Apply(float v, float previous)
        {
            if (Activation == Activation.Silu) v = Silu(v);
            return Add ? previous + Scale * v : v;
        }
    }

    // Each thread takes a run of 8-row weight groups, which stay in L1 while all rows of X pass.
    static void PackedQ8Gemm<TDot>(Q8Activations X, Q8Weights W, float[] Y, Epilogue e) where TDot : struct, IInt8Dot
    {
        int groups = (W.Rows + 7) / 8, threads = Math.Min(Threads, groups);
        Parallel.For(0, threads, Opt, t =>
        {
            int start = (int)((long)groups * t / threads), end = (int)((long)groups * (t + 1) / threads);
            fixed (sbyte* xq = X.Q, wq = W.Q) fixed (float* xd = X.D, wd = W.D, y = Y, bias = W.Bias) fixed (int* xs = X.Sums)
                for (int g = start; g < end; g++)
                {
                    sbyte* wg = wq + (long)g * 8 * W.K; float* dg = wd + g * 8 * W.Blocks;
                    float* bg = bias == null ? null : bias + g * 8;
                    int n0 = g * 8, nc = Math.Min(8, W.Rows - n0), m0 = 0;
                    if (g + 1 < end && W.Rows - n0 >= 16)
                    {
                        // Two groups, 16 outputs, four rows at a time; the rest row by row.
                        for (; m0 + 4 <= X.Rows; m0 += 4)
                            Tile4x16<TDot>(xq + (long)m0 * X.K, xd + m0 * X.Blocks, xs + m0 * X.Blocks, X.K, X.Blocks, wg, dg, bg, y + (long)m0 * W.Rows + n0, W.Rows, e);
                        for (int m = m0; m < X.Rows; m++)
                            for (int half = 0; half < 2; half++)
                                Tile1<TDot>(xq + (long)m * X.K, xd + m * X.Blocks, xs + m * X.Blocks, X.Blocks, wg + half * 8 * W.K, dg + half * 8 * W.Blocks, bg == null ? null : bg + half * 8, y + (long)m * W.Rows + n0 + half * 8, 8, e);
                        g++;
                        continue;
                    }
                    for (; m0 + 4 <= X.Rows; m0 += 4)
                        Tile4<TDot>(xq + (long)m0 * X.K, xd + m0 * X.Blocks, xs + m0 * X.Blocks, X.K, X.Blocks, wg, dg, bg, y + (long)m0 * W.Rows + n0, W.Rows, nc, e);
                    for (; m0 < X.Rows; m0++)
                        Tile1<TDot>(xq + (long)m0 * X.K, xd + m0 * X.Blocks, xs + m0 * X.Blocks, X.Blocks, wg, dg, bg, y + (long)m0 * W.Rows + n0, nc, e);
                }
        });
    }

    // Four rows of X against two 8-row weight groups: 16 outputs per row. 5 loads feed 8
    // multiply-adds (the 4 x 8 tile needs 5 for 4); the float sums spill, once per block.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void Tile4x16<TDot>(sbyte* x, float* xd, int* xs, int K, int blocks, sbyte* w, float* wd, float* bias, float* y, int N, Epilogue e) where TDot : struct, IInt8Dot
    {
        Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default, h0 = default, h1 = default, h2 = default, h3 = default;
        sbyte* x1 = x + K, x2 = x1 + K, x3 = x2 + K, v = w + 8 * K; float* vd = wd + 8 * blocks;
        for (int b = 0; b < blocks; b++)
        {
            var i0 = TDot.Start(xs[b]); var i1 = TDot.Start(xs[blocks + b]); var i2 = TDot.Start(xs[2 * blocks + b]);
            var i3 = TDot.Start(xs[3 * blocks + b]); var j0 = i0; var j1 = i1; var j2 = i2; var j3 = i3;
            sbyte* wb = w + b * 256, vb = v + b * 256; int o = b * 32;
            for (int c = 0; c < 32; c += 4)
            {
                var wv = TDot.Weights(Avx.LoadVector256(wb + c * 8)); var vv = TDot.Weights(Avx.LoadVector256(vb + c * 8));
                var a = Vector256.Create(*(int*)(x + o + c)).AsSByte(); i0 = TDot.Add(i0, a, wv); j0 = TDot.Add(j0, a, vv);
                a = Vector256.Create(*(int*)(x1 + o + c)).AsSByte(); i1 = TDot.Add(i1, a, wv); j1 = TDot.Add(j1, a, vv);
                a = Vector256.Create(*(int*)(x2 + o + c)).AsSByte(); i2 = TDot.Add(i2, a, wv); j2 = TDot.Add(j2, a, vv);
                a = Vector256.Create(*(int*)(x3 + o + c)).AsSByte(); i3 = TDot.Add(i3, a, wv); j3 = TDot.Add(j3, a, vv);
            }
            var s = Avx.LoadVector256(wd + b * 8); var t = Avx.LoadVector256(vd + b * 8);
            var d0 = Vector256.Create(xd[b]); var d1 = Vector256.Create(xd[blocks + b]); var d2 = Vector256.Create(xd[2 * blocks + b]);
            f0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i0), s * d0, f0); h0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(j0), t * d0, h0);
            f1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i1), s * d1, f1); h1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(j1), t * d1, h1);
            f2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i2), s * d2, f2); h2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(j2), t * d2, h2);
            var d3 = Vector256.Create(xd[3 * blocks + b]);
            f3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i3), s * d3, f3); h3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(j3), t * d3, h3);
        }
        float* b2 = bias == null ? null : bias + 8;
        Store(f0, bias, y, 8, e); Store(h0, b2, y + 8, 8, e);
        Store(f1, bias, y + N, 8, e); Store(h1, b2, y + N + 8, 8, e);
        Store(f2, bias, y + 2 * N, 8, e); Store(h2, b2, y + 2 * N + 8, 8, e);
        Store(f3, bias, y + 3 * N, 8, e); Store(h3, b2, y + 3 * N + 8, 8, e);
    }

    // Four rows of X against one 8-row weight group: 8 outputs per row.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void Tile4<TDot>(sbyte* x, float* xd, int* xs, int K, int blocks, sbyte* w, float* wd, float* bias, float* y, int N, int nc, Epilogue e) where TDot : struct, IInt8Dot
    {
        Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default;
        sbyte* x1 = x + K, x2 = x1 + K, x3 = x2 + K;
        for (int b = 0; b < blocks; b++)
        {
            var i0 = TDot.Start(xs[b]); var i1 = TDot.Start(xs[blocks + b]); var i2 = TDot.Start(xs[2 * blocks + b]); var i3 = TDot.Start(xs[3 * blocks + b]);
            sbyte* wb = w + b * 256; int o = b * 32;
            for (int c = 0; c < 32; c += 4)
            {
                var wv = TDot.Weights(Avx.LoadVector256(wb + c * 8));
                i0 = TDot.Add(i0, Vector256.Create(*(int*)(x + o + c)).AsSByte(), wv);
                i1 = TDot.Add(i1, Vector256.Create(*(int*)(x1 + o + c)).AsSByte(), wv);
                i2 = TDot.Add(i2, Vector256.Create(*(int*)(x2 + o + c)).AsSByte(), wv);
                i3 = TDot.Add(i3, Vector256.Create(*(int*)(x3 + o + c)).AsSByte(), wv);
            }
            var s = Avx.LoadVector256(wd + b * 8);
            f0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i0), s * Vector256.Create(xd[b]), f0);
            f1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i1), s * Vector256.Create(xd[blocks + b]), f1);
            f2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i2), s * Vector256.Create(xd[2 * blocks + b]), f2);
            f3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i3), s * Vector256.Create(xd[3 * blocks + b]), f3);
        }
        Store(f0, bias, y, nc, e); Store(f1, bias, y + N, nc, e); Store(f2, bias, y + 2 * N, nc, e); Store(f3, bias, y + 3 * N, nc, e);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void Tile1<TDot>(sbyte* x, float* xd, int* xs, int blocks, sbyte* w, float* wd, float* bias, float* y, int nc, Epilogue e) where TDot : struct, IInt8Dot
    {
        Vector256<float> f = default;
        for (int b = 0; b < blocks; b++)
        {
            var i = TDot.Start(xs[b]);
            sbyte* wb = w + b * 256; int o = b * 32;
            for (int c = 0; c < 32; c += 4)
                i = TDot.Add(i, Vector256.Create(*(int*)(x + o + c)).AsSByte(), TDot.Weights(Avx.LoadVector256(wb + c * 8)));
            f = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i), Avx.LoadVector256(wd + b * 8) * Vector256.Create(xd[b]), f);
        }
        Store(f, bias, y, nc, e);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void Store(Vector256<float> f, float* bias, float* y, int nc, Epilogue e)
    {
        if (bias != null) f += Avx.LoadVector256(bias);
        if (e.Activation == Activation.Silu) f /= Vector256<float>.One + Vector256.Exp(-f);
        float* t = stackalloc float[8], target = nc == 8 ? y : t;
        if (e.Add)
        {
            if (nc < 8) for (int j = 0; j < nc; j++) t[j] = y[j];
            f = Fma.MultiplyAdd(f, Vector256.Create(e.Scale), Avx.LoadVector256(target));
        }
        Avx.Store(target, f);
        if (nc < 8) for (int j = 0; j < nc; j++) y[j] = t[j];
    }

    // One step of a packed int8 dot product: 8 lanes, each adding 4 products of x and w.
    interface IInt8Dot
    {
        static abstract Vector256<int> Start(int blockSum);
        static abstract Vector256<sbyte> Weights(Vector256<sbyte> w);
        static abstract Vector256<int> Add(Vector256<int> sum, Vector256<sbyte> x, Vector256<sbyte> w);
    }

    // Signed times signed (Arrow Lake, Lunar Lake and later).
    struct VnniInt8Dot : IInt8Dot
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector256<int> Start(int blockSum) => Vector256<int>.Zero;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector256<sbyte> Weights(Vector256<sbyte> w) => w;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector256<int> Add(Vector256<int> sum, Vector256<sbyte> x, Vector256<sbyte> w) => AvxVnniInt8.MultiplyWideningAndAdd(sum, x, w);
    }

    // Unsigned times signed (Alder Lake and later, Zen 5): w + 128 is unsigned, so the sum is
    // x.w + 128 * (sum of x), and starting each block at -128 * (sum of x) cancels the extra.
    struct VnniDot : IInt8Dot
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector256<int> Start(int blockSum) => Vector256.Create(-128 * blockSum);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector256<sbyte> Weights(Vector256<sbyte> w) => w ^ Vector256.Create(sbyte.MinValue);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector256<int> Add(Vector256<int> sum, Vector256<sbyte> x, Vector256<sbyte> w) => AvxVnni.MultiplyWideningAndAdd(sum, w.AsByte(), x);
    }

    // AVX2 only: |w| times x with the sign of w, in pairs to int16 (fits: 2 * 128 * 127), then
    // to int32. The absolute value goes to w, as in ggml: |-128| is 128 unsigned, while x, being
    // quantized to -127..127, can always be negated.
    struct Avx2Dot : IInt8Dot
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector256<int> Start(int blockSum) => Vector256<int>.Zero;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector256<sbyte> Weights(Vector256<sbyte> w) => w;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Add(Vector256<int> sum, Vector256<sbyte> x, Vector256<sbyte> w) =>
            sum + Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx2.Abs(w), Avx2.Sign(x, w)), Vector256.Create((short)1));
    }

    // 128-bit vectors work on every x64 CPU (SSE2) and on ARM (NEON); slower than the AVX2 tiles.
    static void PortableQ8Gemm(Q8Activations X, Q8Weights W, float[] Y, Epilogue e)
    {
        Parallel.For(0, W.Rows, Opt, n =>
        {
            fixed (sbyte* xq = X.Q, wq = W.Q)
                for (int m = 0; m < X.Rows; m++)
                {
                    float sum = 0;
                    for (int b = 0; b < X.Blocks; b++)
                        sum += BlockDot(xq + m * X.K + b * 32, wq + n * W.K + b * 32) * X.D[m * X.Blocks + b] * W.D[n * W.Blocks + b];
                    long i = (long)m * W.Rows + n;
                    Y[i] = e.Apply(sum + (W.Bias?[n] ?? 0f), Y[i]);
                }
        });
    }

    // The integer dot product of two 32-value int8 blocks. Products of int8 fit in int16.
    static int BlockDot(sbyte* x, sbyte* w)
    {
        var acc = Vector128<int>.Zero;
        for (int half = 0; half < 32; half += 16)
        {
            var (xl, xh) = Vector128.Widen(Vector128.Load(x + half));
            var (wl, wh) = Vector128.Widen(Vector128.Load(w + half));
            var (p0, p1) = Vector128.Widen(xl * wl);
            var (p2, p3) = Vector128.Widen(xh * wh);
            acc += p0 + p1 + p2 + p3;
        }
        return Vector128.Sum(acc);
    }

    // Y[M,N] = ReLU(X[M,K] * Wt[K,N] + bias) in float, Wt being the weights transposed; N a
    // multiple of two vectors.
    public static float[] GemmRelu(float[] x, int m, float[] wt, int n, int k, float[] bias)
    {
        RequireVectorPairs(n, nameof(n));
        var y = new float[(long)m * n];
        int tiles = (m + 3) / 4;
        ForRanges(tiles, (start, end) =>
        {
            fixed (float* xp = x, wp = wt, yp = y, bp = bias)
                for (int tile = start; tile < end; tile++)
                    FloatRows(xp + (long)tile * 4 * k, k, wp, n, yp + (long)tile * 4 * n, n, Math.Min(4, m - tile * 4), n, k, bp, relu: true);
        });
        return y;
    }

    static void RequireVectorPairs(int n, string name)
    {
        if (n % (2 * Vector<float>.Count) != 0)
            throw new ArgumentException($"{name} = {n} is not a multiple of {2 * Vector<float>.Count} floats.", name);
    }

    // Up to 4 rows of Y[., n] = X[., k] * W[k][n] (+ bias, ReLU), with row strides ldx, ldw, ldy;
    // n a multiple of two vectors. Tiles of 4 rows by 2 vectors keep 8 sums in registers.
    static void FloatRows(float* x, int ldx, float* w, int ldw, float* y, int ldy, int rows, int n, int k, float* bias, bool relu)
    {
        for (int n0 = 0; n0 < n; n0 += 2 * Vector<float>.Count)
            if (rows == 4) FloatTile4(x, ldx, w + n0, ldw, y + n0, ldy, k, bias == null ? null : bias + n0, relu);
            else for (int r = 0; r < rows; r++) FloatTile1(x + (long)r * ldx, w + n0, ldw, y + (long)r * ldy + n0, k, bias == null ? null : bias + n0, relu);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void FloatTile4(float* x, int ldx, float* w, int ldw, float* y, int ldy, int k, float* bias, bool relu)
    {
        int v = Vector<float>.Count;
        Vector<float> b0 = bias == null ? default : Vector.Load(bias), b1 = bias == null ? default : Vector.Load(bias + v);
        Vector<float> a00 = b0, a01 = b1, a10 = b0, a11 = b1, a20 = b0, a21 = b1, a30 = b0, a31 = b1;
        float* x1 = x + ldx, x2 = x1 + ldx, x3 = x2 + ldx;
        for (int i = 0; i < k; i++, w += ldw)
        {
            var w0 = Vector.Load(w); var w1 = Vector.Load(w + v);
            var s = new Vector<float>(x[i]); a00 = MultiplyAdd(s, w0, a00); a01 = MultiplyAdd(s, w1, a01);
            s = new Vector<float>(x1[i]); a10 = MultiplyAdd(s, w0, a10); a11 = MultiplyAdd(s, w1, a11);
            s = new Vector<float>(x2[i]); a20 = MultiplyAdd(s, w0, a20); a21 = MultiplyAdd(s, w1, a21);
            s = new Vector<float>(x3[i]); a30 = MultiplyAdd(s, w0, a30); a31 = MultiplyAdd(s, w1, a31);
        }
        StorePair(a00, a01, y, relu); StorePair(a10, a11, y + ldy, relu);
        StorePair(a20, a21, y + 2 * ldy, relu); StorePair(a30, a31, y + 3 * ldy, relu);
    }

    static void FloatTile1(float* x, float* w, int ldw, float* y, int k, float* bias, bool relu)
    {
        int v = Vector<float>.Count;
        Vector<float> a0 = bias == null ? default : Vector.Load(bias), a1 = bias == null ? default : Vector.Load(bias + v);
        for (int i = 0; i < k; i++, w += ldw)
        {
            var s = new Vector<float>(x[i]);
            a0 = MultiplyAdd(s, Vector.Load(w), a0); a1 = MultiplyAdd(s, Vector.Load(w + v), a1);
        }
        StorePair(a0, a1, y, relu);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void StorePair(Vector<float> a, Vector<float> b, float* y, bool relu)
    {
        if (relu) { a = Vector.Max(a, Vector<float>.Zero); b = Vector.Max(b, Vector<float>.Zero); }
        a.Store(y); b.Store(y + Vector<float>.Count);
    }

    // a * b + c, fused where the CPU can (software FMA would be far slower than the two steps).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector<float> MultiplyAdd(Vector<float> a, Vector<float> b, Vector<float> c) =>
        Simd ? Vector.FusedMultiplyAdd(a, b, c) : a * b + c;

    // [rows][columns] -> [columns][stride], stride >= rows; the padding is zero.
    public static float[] Transpose(float[] a, int rows, int columns, int stride)
    {
        var t = new float[(long)columns * stride];
        for (int r = 0; r < rows; r++) for (int c = 0; c < columns; c++) t[(long)c * stride + r] = a[(long)r * columns + c];
        return t;
    }

    // Positional projections p[P][d] as [d][stride], each head's dk rows together. The padding
    // lets score tiles read whole vectors past the last position; those columns are ignored.
    // Utterances over 20 s make a table per layer (13 MB at 2 min), so it comes from the scratch
    // and is filled by all threads, 16 destination rows at a time from whole source cache lines.
    public static (float[] Table, int Stride) TransposePositions(float[] p, int P, int d)
    {
        int stride = P + 4 * Vector<float>.Count;
        var t = EncoderScratch.Rent(d * stride);
        ForRanges((d + 15) / 16, (start, end) =>
        {
            for (int block = start; block < end; block++)
            {
                int c0 = block * 16, c1 = Math.Min(d, c0 + 16);
                for (int r = 0; r < P; r++)
                    for (int c = c0; c < c1; c++)
                        t[(long)c * stride + r] = p[(long)r * d + c];
                for (int c = c0; c < c1; c++)
                    Array.Clear(t, c * stride + P, stride - P);
            }
        });
        return (t, stride);
    }

    // Query rows per attention task. A task streams its head's keys, positions and values once,
    // so 32 rows read them 8 times less often than one 4-row tile at a time: at 2 minutes a head
    // is 2.5 MB, more L2 than a core has to itself (2 MB per E-core, 3 MB per P-core), so each
    // read came from L3. The block's scores (32 x T floats, 210 KB at 2 minutes) stay in L2.
    // A multiple of 4, so the row tiles, and with them every sum, are the ones a 4-row walk makes.
    const int QueryBlock = 32;

    // Relative-position self-attention (Transformer-XL, as FastConformer uses it), all heads:
    // score[i][j] = ((q_i + u).k_j + (q_i + v).p[first + T-1-i+j]) / sqrt(dk), ctx_i = softmax(score_i).V.
    // qkv is [T][3d] (q | k | v), positions come from TransposePositions, ctx is [T][d].
    public static void RelativeAttention(float[] qkv, int T, int d, int heads, float[] u, float[] v, float[] positions, int stride, int first, float[] ctx)
    {
        int dk = d / heads, pair = 2 * Vector<float>.Count, width = (T + pair - 1) / pair * pair, blocks = (T + QueryBlock - 1) / QueryBlock;
        RequireVectorPairs(dk, nameof(dk));
        float scale = 1f / MathF.Sqrt(dk);
        // Per head: q + u and q + v as [h][i][e]; keys as [h][pair of key columns][e][pair], so that
        // a score tile walks its dk rows of keys in one run (8 KB) instead of dk rows a whole width
        // apart, each on its own page (scores 11% faster at 2 minutes); values as
        // [h][pair of columns][j][pair] so that each pair of output columns reads one run.
        var qu = EncoderScratch.Rent(T * d);
        var qv = EncoderScratch.Rent(T * d);
        var keys = EncoderScratch.Rent(d * width);
        var values = EncoderScratch.Rent(T * d);
        // Head by head, so that no element needs a division by dk, a runtime value (dropping the
        // two divisions per element made attention 10% faster for 20 s, 5% for 2 minutes).
        For(T, i =>
        {
            fixed (float* row = &qkv[i * 3 * d], up = u, vp = v, qup = qu, qvp = qv, kp = keys, valuesp = values)
                for (int h = 0, c = 0; h < heads; h++)
                {
                    float* quRow = qup + ((long)h * T + i) * dk, qvRow = qvp + ((long)h * T + i) * dk;
                    float* keyColumn = kp + (long)h * dk * width + i / pair * dk * pair + i % pair;
                    float* valueRow = valuesp + (long)h * dk * T + i * pair;
                    for (int e = 0; e < dk; e += Vector<float>.Count)
                    {
                        var q = Vector.Load(row + c + e);
                        (q + Vector.Load(up + c + e)).Store(quRow + e);
                        (q + Vector.Load(vp + c + e)).Store(qvRow + e);
                    }
                    for (int e = 0; e < dk; e++)
                    {
                        keyColumn[e * pair] = row[d + c + e];
                        valueRow[(e - e % pair) * T + e % pair] = row[2 * d + c + e];
                    }
                    c += dk;
                }
        });
        For(heads * blocks, task =>
        {
            int h = task / blocks, i0 = task % blocks * QueryBlock, rows = Math.Min(QueryBlock, T - i0);
            var scores = ArrayPool<float>.Shared.Rent(QueryBlock * width);
            fixed (float* s = scores, qup = qu, qvp = qv, kp = keys, pp = positions, vp = values, cp = ctx)
            {
                float* qur = qup + (h * T + i0) * dk, qvr = qvp + (h * T + i0) * dk;
                float* kh = kp + h * dk * width, ph = pp + (long)h * dk * stride + first + T - 1 - i0;
                // A pair of key columns, with the positions it meets, is loaded once for every row tile.
                for (int j0 = 0; j0 < width; j0 += pair)
                    for (int r0 = 0; r0 < rows; r0 += 4)
                        if (rows - r0 >= 4) ScoreTile4(qur + r0 * dk, qvr + r0 * dk, dk, kh + j0 * dk, pair, ph + j0 - r0, stride, s + r0 * width + j0, width, scale);
                        else for (int r = r0; r < rows; r++) ScoreTile1(qur + r * dk, qvr + r * dk, dk, kh + j0 * dk, pair, ph + j0 - r, stride, s + r * width + j0, scale);
                for (int r = 0; r < rows; r++)
                    Softmax(s + r * width, T, width);
                // Likewise a pair of value columns for every row tile.
                float* vh = vp + (long)h * dk * T;
                for (int n0 = 0; n0 < dk; n0 += pair)
                    for (int r0 = 0; r0 < rows; r0 += 4)
                        FloatRows(s + r0 * width, width, vh + n0 * T, pair, cp + (i0 + r0) * d + h * dk + n0, d, Math.Min(4, rows - r0), pair, T, null, relu: false);
            }
            ArrayPool<float>.Shared.Return(scores);
        });
    }

    // Scores of 4 query rows against 2 vectors of key columns. Row r's position window starts
    // one column further left than row r-1's: that is the relative shift.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void ScoreTile4(float* qu, float* qv, int dk, float* k, int ldk, float* p, int ldp, float* s, int lds, float scale)
    {
        int v = Vector<float>.Count;
        Vector<float> a00 = default, a01 = default, a10 = default, a11 = default, a20 = default, a21 = default, a30 = default, a31 = default;
        for (int e = 0; e < dk; e++)
        {
            float* kr = k + e * ldk; var k0 = Vector.Load(kr); var k1 = Vector.Load(kr + v);
            var q = new Vector<float>(qu[e]); a00 = MultiplyAdd(q, k0, a00); a01 = MultiplyAdd(q, k1, a01);
            q = new Vector<float>(qu[dk + e]); a10 = MultiplyAdd(q, k0, a10); a11 = MultiplyAdd(q, k1, a11);
            q = new Vector<float>(qu[2 * dk + e]); a20 = MultiplyAdd(q, k0, a20); a21 = MultiplyAdd(q, k1, a21);
            q = new Vector<float>(qu[3 * dk + e]); a30 = MultiplyAdd(q, k0, a30); a31 = MultiplyAdd(q, k1, a31);
        }
        for (int e = 0; e < dk; e++)
        {
            float* pr = p + (long)e * ldp;
            var q = new Vector<float>(qv[e]); a00 = MultiplyAdd(q, Vector.Load(pr), a00); a01 = MultiplyAdd(q, Vector.Load(pr + v), a01);
            q = new Vector<float>(qv[dk + e]); a10 = MultiplyAdd(q, Vector.Load(pr - 1), a10); a11 = MultiplyAdd(q, Vector.Load(pr - 1 + v), a11);
            q = new Vector<float>(qv[2 * dk + e]); a20 = MultiplyAdd(q, Vector.Load(pr - 2), a20); a21 = MultiplyAdd(q, Vector.Load(pr - 2 + v), a21);
            q = new Vector<float>(qv[3 * dk + e]); a30 = MultiplyAdd(q, Vector.Load(pr - 3), a30); a31 = MultiplyAdd(q, Vector.Load(pr - 3 + v), a31);
        }
        var sc = new Vector<float>(scale);
        (a00 * sc).Store(s); (a01 * sc).Store(s + v); s += lds;
        (a10 * sc).Store(s); (a11 * sc).Store(s + v); s += lds;
        (a20 * sc).Store(s); (a21 * sc).Store(s + v); s += lds;
        (a30 * sc).Store(s); (a31 * sc).Store(s + v);
    }

    static void ScoreTile1(float* qu, float* qv, int dk, float* k, int ldk, float* p, int ldp, float* s, float scale)
    {
        int v = Vector<float>.Count;
        Vector<float> a0 = default, a1 = default;
        for (int e = 0; e < dk; e++)
        {
            float* kr = k + e * ldk, pr = p + (long)e * ldp;
            var q = new Vector<float>(qu[e]); a0 = MultiplyAdd(q, Vector.Load(kr), a0); a1 = MultiplyAdd(q, Vector.Load(kr + v), a1);
            q = new Vector<float>(qv[e]); a0 = MultiplyAdd(q, Vector.Load(pr), a0); a1 = MultiplyAdd(q, Vector.Load(pr + v), a1);
        }
        (a0 * scale).Store(s); (a1 * scale).Store(s + v);
    }

    // Softmax over s[0..n); s[n..width) is padding and comes out zero.
    static void Softmax(float* s, int n, int width)
    {
        int v = Vector<float>.Count;
        for (int j = n; j < width; j++) s[j] = float.NegativeInfinity;
        var max = new Vector<float>(float.NegativeInfinity);
        for (int j = 0; j < width; j += v) max = LargerOf(max, Vector.Load(s + j));
        float m = float.NegativeInfinity;
        for (int l = 0; l < v; l++) m = MathF.Max(m, max[l]);
        var sum = Vector<float>.Zero;
        for (int j = 0; j < width; j += v) { var e = Vector.Exp(Vector.Load(s + j) - new Vector<float>(m)); e.Store(s + j); sum += e; }
        var inverse = new Vector<float>(1f / Vector.Sum(sum));
        for (int j = 0; j < width; j += v) (Vector.Load(s + j) * inverse).Store(s + j);
    }

    // The larger of two vectors of numbers (no NaN): the plain vmaxps where vectors are 256-bit AVX,
    // as Vector.Max spends 9 instructions per step on NaN and signed-zero rules. Same Softmax output:
    // a tie of 0 and -0 may keep either, and exp gives both the same bits. (A row holding NaN comes out
    // all NaN either way, though not always the same NaN.)
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector<float> LargerOf(Vector<float> a, Vector<float> b) =>
        Vector<float>.Count == 8 && Avx.IsSupported ? Avx.Max(a.AsVector256(), b.AsVector256()).AsVector() : Vector.Max(a, b);

    public static float[] LayerNorm(float[] x, int m, int d, float[] g, float[] b)
    {
        var o = EncoderScratch.Rent(x.Length);
        Parallel.For(0, m, Opt, r =>
        {
            fixed (float* xp = x, op = o, gp = g, bp = b)
                NormRow(xp + (long)r * d, op + (long)r * d, gp, bp, d);
        });
        return o;
    }

    // d is a multiple of the vector width (1024 and 640 here, 4 or 8 lanes).
    static void NormRow(float* x, float* o, float* g, float* b, int d)
    {
        int n = Vector<float>.Count;
        var sum = Vector<float>.Zero;
        for (int i = 0; i < d; i += n) sum += Vector.Load(x + i);
        var mean = Vector.Sum(sum) / d;
        var squares = Vector<float>.Zero;
        for (int i = 0; i < d; i += n) { var t = Vector.Load(x + i) - new Vector<float>(mean); squares += t * t; }
        var inverse = new Vector<float>(1f / MathF.Sqrt(Vector.Sum(squares) / d + 1e-5f));
        for (int i = 0; i < d; i += n)
            ((Vector.Load(x + i) - new Vector<float>(mean)) * inverse * Vector.Load(g + i) + Vector.Load(b + i)).Store(o + i);
    }

    public static void For(int n, Action<int> body) => Parallel.For(0, n, Opt, body);

    // One contiguous run of [0, n) per thread, for work that keeps state along the run.
    public static void ForRanges(int n, Action<int, int> body)
    {
        int threads = Math.Max(1, Math.Min(Threads, n));
        Parallel.For(0, threads, Opt, t => body((int)((long)n * t / threads), (int)((long)n * (t + 1) / threads)));
    }
    public static float Silu(float v) => v / (1f + MathF.Exp(-v));
    public static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));
}

// The int8 matrix products, slowest first; Kernels.BestPath is the one this CPU uses.
internal enum Int8Path { Portable, Avx2, AvxVnni, AvxVnniInt8 }

// Weights packed for one int8 path by Kernels.Pack, with the bias added to every output.
internal sealed class Q8Weights(Int8Path path, int rows, int k, sbyte[] q, float[] d, float[]? bias)
{
    public Int8Path Path { get; } = path;
    public int Rows { get; } = rows;
    public int K { get; } = k;
    public int Blocks { get; } = k / 32;
    public sbyte[] Q { get; } = q;
    public float[] D { get; } = d;
    public float[]? Bias { get; } = bias;
}

// Activations quantized per 32-value block, with each block's sum of int8 values.
internal sealed class Q8Activations(int rows, int k)
{
    public int Rows { get; } = rows;
    public int K { get; } = k;
    public int Blocks { get; } = k / 32;
    public sbyte[] Q { get; } = new sbyte[checked(rows * k)];
    public float[] D { get; } = new float[checked(rows * (k / 32))];
    public int[] Sums { get; } = new int[checked(rows * (k / 32))];
}

internal enum Activation { None, Silu }
