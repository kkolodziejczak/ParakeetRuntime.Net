using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using ParakeetRuntime;

// The CPU encoder's kernels alone, per Conformer layer, for comparing a change in seconds instead of a corpus run.
//   --cpu-bench [--frames 250,1500] [--affinity C03] [--threads N] [--int8-path Portable|Avx2|AvxVnni|AvxVnniInt8]
// Prints the best of several runs per kernel in ms per layer (for a matrix product, also the part spent quantizing
// its input), their total, and a checksum of each kernel's output bits: two builds that print the same checksums
// computed the same bits. Each kernel's inputs depend only on the kernel and the frame count, so the checksums
// don't change with --frames' order. The total leaves out the layer's smaller kernels (norms, convolution).
internal static class CpuBench
{
    private const int D = 1024, Heads = 8, Runs = 7;

    // One Conformer layer's matrix products as the encoder runs them (ParakeetModel's layer): output columns, input
    // width, rows as a multiple of the frame count, times per layer, and the output step: SiLU, or added into
    // the residual (the feed-forwards' at half weight). No bias, like the encoder's layers. Positions run only
    // above 20 s (251 frames); shorter utterances read the table made at load.
    private static readonly (string Name, int N, int K, int RowsPerFrame, int Count, Output Output)[] LayerShapes =
    [
        ("ff in", 4096, 1024, 1, 2, Output.Silu),
        ("ff out", 1024, 4096, 1, 2, Output.AddHalfToResidual),
        ("qkv", 3072, 1024, 1, 1, Output.Plain),
        ("positions", 1024, 1024, 2, 1, Output.Plain),
        ("out, pw2", 1024, 1024, 1, 2, Output.AddToResidual),
        ("pw1", 2048, 1024, 1, 1, Output.Plain),
    ];

    private enum Output { Plain, Silu, AddHalfToResidual, AddToResidual }

    public static int Run(string[] args)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        string? Value(string name) => Array.IndexOf(args, name) is var i and >= 0 && i + 1 < args.Length ? args[i + 1] : null;
        if (Value("--affinity") is { } mask)
            Process.GetCurrentProcess().ProcessorAffinity = (nint)long.Parse(mask.Replace("0x", "", StringComparison.OrdinalIgnoreCase), NumberStyles.HexNumber);
        if (Value("--threads") is { } threads)
            Kernels.UseThreads(int.Parse(threads, CultureInfo.InvariantCulture));
        var path = Kernels.BestPath;
        if (Value("--int8-path") is { } name && (!Enum.TryParse(name, out path) || !Kernels.Supports(path)))
        {
            Console.Error.WriteLine($"--int8-path is one this CPU has: {string.Join(", ", Enum.GetValues<Int8Path>().Where(Kernels.Supports))}.");
            return 2;
        }
        var frameCounts = (Value("--frames") ?? "250,1500").Split(',').Select(int.Parse).ToArray();
        Console.WriteLine($"CPU kernels, {path}, {Kernels.ThreadCount} threads, affinity 0x{Process.GetCurrentProcess().ProcessorAffinity:X}, best of {Runs}");

        var random = new Random(1);
        var weights = LayerShapes.Select(shape => RandomWeights(random, shape.N, shape.K, path)).ToArray();
        foreach (var frames in frameCounts)
        {
            var total = 0.0;
            for (var s = 0; s < LayerShapes.Length; s++)
            {
                var (shape, w) = (LayerShapes[s], weights[s]);
                if (shape.Name == "positions" && frames <= 251)
                    continue;
                int rows = frames * shape.RowsPerFrame - (shape.RowsPerFrame - 1);
                var inputs = new Random(frames * 100 + s);
                var x = RandomFloats(inputs, rows * shape.K, 1f);
                // A residual collects every run's sum, as many runs in every build.
                bool residual = shape.Output is Output.AddHalfToResidual or Output.AddToResidual;
                var y = residual ? RandomFloats(inputs, rows * shape.N, 1f) : [];
                var ms = Best(() =>
                {
                    if (residual)
                        Kernels.LinearAdd(x, rows, w, shape.Output == Output.AddHalfToResidual ? 0.5f : 1f, y);
                    else
                        y = Kernels.Linear(x, rows, w, shape.Output == Output.Silu ? Activation.Silu : Activation.None);
                });
                var quantizeMs = Best(() => Kernels.Quantize(x, rows, shape.K));
                total += ms * shape.Count;
                Console.WriteLine($"  T={frames,5}  {shape.Name,-10} {rows,5} x {shape.K,4} x {shape.N,4}  {ms * shape.Count,8:0.00} ms (quantizing {quantizeMs * shape.Count:0.00})  {Checksum(y)}");
            }
            {
                var inputs = new Random(frames * 100 + LayerShapes.Length);
                var (qkv, u, v) = (RandomFloats(inputs, frames * 3 * D, 1f), RandomFloats(inputs, D, 0.1f), RandomFloats(inputs, D, 0.1f));
                int positionsRows = 2 * frames - 1, stride = positionsRows + 4 * Vector<float>.Count;
                var positions = RandomFloats(inputs, D * stride, 1f);
                var ctx = new float[frames * D];
                var ms = Best(() => Kernels.RelativeAttention(qkv, frames, D, Heads, u, v, positions, stride, 0, ctx));
                total += ms;
                Console.WriteLine($"  T={frames,5}  {"attention",-10} {frames,5} frames, {Heads} heads       {ms,8:0.00} ms  {Checksum(ctx)}");
            }
            Console.WriteLine($"  T={frames,5}  {"kernels",-10} {"",22}  {total,8:0.00} ms");
        }
        return 0;
    }

    // Buffers come from a scratch, reused run after run as the encoder reuses them layer after layer.
    private static double Best(Action run)
    {
        using var scratch = new EncoderScratch();
        var best = double.MaxValue;
        for (var i = 0; i <= Runs; i++)
        {
            scratch.Rewind();
            var clock = Stopwatch.StartNew();
            run();
            if (i > 0)
                best = Math.Min(best, clock.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    private static Q8Weights RandomWeights(Random random, int n, int k, Int8Path path)
    {
        var m = new Q8Matrix(n, k);
        for (var i = 0; i < m.Q.Length; i++)
            m.Q[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < m.D.Length; i++)
            m.D[i] = (float)(Half)(random.NextDouble() / 50);
        return Kernels.Pack(m, null, path);
    }

    private static float[] RandomFloats(Random random, int count, float scale)
    {
        var values = new float[count];
        for (var i = 0; i < count; i++)
            values[i] = (float)(random.NextDouble() * 2 - 1) * scale;
        return values;
    }

    // FNV-1a over the bits: the same in every process, unlike HashCode.
    private static string Checksum(float[] values)
    {
        var hash = 14695981039346656037UL;
        foreach (var value in values)
            hash = (hash ^ (uint)BitConverter.SingleToInt32Bits(value)) * 1099511628211UL;
        return $"bits {hash:X16}";
    }
}
