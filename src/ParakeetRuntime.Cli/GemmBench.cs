using System.Diagnostics;
using ParakeetRuntime;
using static ParakeetRuntime.D3D12Native;

internal static unsafe class GemmBench
{
    private static readonly (int N, int K, int RowsPerFrame, int Count)
        FeedForwardInTwice = (4096, 1024, 1, 2),
        FeedForwardOutTwice = (1024, 4096, 1, 2),
        QueryKeyValue = (3072, 1024, 1, 1),
        PositionalProjectionOf2TMinus1Rows = (1024, 1024, 2, 1),
        AttentionOutAndPointwiseConv2 = (1024, 1024, 1, 2),
        PointwiseConv1 = (2048, 1024, 1, 1);

    private static readonly (int N, int K, int RowsPerFrame, int Count)[] ConformerLayerQ8Products =
    [
        FeedForwardInTwice,
        FeedForwardOutTwice,
        QueryKeyValue,
        PositionalProjectionOf2TMinus1Rows,
        AttentionOutAndPointwiseConv2,
        PointwiseConv1,
    ];

    private sealed record Variant(string Entry, int Tile, string? File);

    public static int Run(string[] args)
    {
        string? Value(string name) => Array.IndexOf(args, name) is var i and >= 0 && i + 1 < args.Length ? args[i + 1] : null;
        var variants = new List<Variant>();
        for (var i = 0; i < args.Length; i++)
        {
            if (args[i] != "--variant")
                continue;
            var parts = i + 1 < args.Length ? args[i + 1].Split(':', 3) : [];
            if (parts.Length < 2 || !int.TryParse(parts[1], out var tile) || tile <= 0)
            {
                Console.Error.WriteLine("--variant is Entry:tile[:file.hlsl], for example Q8GemmMain:64.");
                return 2;
            }
            variants.Add(new Variant(parts[0], tile, parts.Length > 2 ? parts[2] : null));
        }
        if (variants.Count == 0)
        {
            variants.Add(new Variant("Q8GemmSmallMain", ParakeetGpu.Q8GemmSmallTile, null));
            variants.Add(new Variant("Q8GemmMain", ParakeetGpu.Q8GemmLargeTile, null));
        }
        var frameCounts = (Value("--frames") ?? "1500,250").Split(',').Select(int.Parse).ToArray();
        var device = Value("--device") ?? "gpu";
        if (device != "gpu" && !device.StartsWith("gpu:", StringComparison.Ordinal))
        {
            Console.Error.WriteLine($"--gemm-bench runs on GPUs only: --device gpu or gpu:<adapter>, not {device}.");
            return 2;
        }
        var adapterFilter = device == "gpu" ? null : device["gpu:".Length..];

        foreach (var adapter in D3D12Device.EnumerateAdapters().Where(a => a.MeetsMinimumMemory && (adapterFilter is null || a.Name.Contains(adapterFilter, StringComparison.OrdinalIgnoreCase))))
        {
            using var gpu = D3D12Device.TryCreate(adapterNameFilter: adapter.Name);
            if (gpu is null)
                continue;
            Console.WriteLine($"== {gpu.AdapterName}");
            var pipelines = new List<nint>();
            try
            {
                foreach (var v in variants)
                {
                    var compile = Stopwatch.StartNew();
                    pipelines.Add(v.File is null ? gpu.CompileEmbeddedPipeline("Q8Gemm.hlsl", v.Entry) : gpu.CompilePipeline(File.ReadAllText(v.File), v.File, v.Entry));
                    Console.WriteLine($"  {v.Entry} from {Path.GetFileName(v.File ?? "shipped")} compiled in {compile.Elapsed.TotalMilliseconds:0} ms");
                }
                foreach (var frames in frameCounts)
                {
                    var layerMs = new double[variants.Count];
                    var worstDifference = new double[variants.Count];
                    var unwritten = new long[variants.Count];
                    foreach (var shape in ConformerLayerQ8Products)
                    {
                        using var problem = new RandomGemmResidentOnGpu(gpu, frames * shape.RowsPerFrame - (shape.RowsPerFrame - 1), shape.N, shape.K);
                        float[]? reference = null;
                        for (var v = 0; v < variants.Count; v++)
                        {
                            var ms = problem.BestMsPerDispatch(pipelines[v], variants[v]);
                            layerMs[v] += ms * shape.Count;
                            var output = problem.Read();
                            reference ??= output;
                            for (var i = 0; i < output.Length; i++)
                            {
                                if (StillNaNSoNeverWritten(output[i]))
                                    unwritten[v]++;
                                else
                                    worstDifference[v] = Math.Max(worstDifference[v], Math.Abs(output[i] - reference[i]) / Math.Max(1, Math.Abs(reference[i])));
                            }
                        }
                    }
                    for (var v = 0; v < variants.Count; v++)
                        Console.WriteLine($"  T={frames,5}  {variants[v].Entry,-12} {variants[v].Tile,2}x{variants[v].Tile,-2} {Path.GetFileName(variants[v].File ?? "shipped"),-14}  {layerMs[v],7:0.000} ms per layer  {layerMs[0] / layerMs[v],5:0.00}x  worst relative difference {worstDifference[v]:E1}" +
                            (unwritten[v] > 0 ? $", {unwritten[v]} outputs NOT WRITTEN" : ""));
                }
            }
            finally
            {
                foreach (var pipeline in pipelines)
                {
                    var p = pipeline;
                    SafeRelease(ref p);
                }
            }
        }
        return 0;
    }

    private static bool StillNaNSoNeverWritten(float output) => float.IsNaN(output);

    private sealed class RandomGemmResidentOnGpu : IDisposable
    {
        private const int DispatchesPerTimedSubmit = 30;
        private const int OneWarmupAndThreeTimedSubmits = 4;
        private const int NaNFillingWarmupSubmit = 0;
        private readonly D3D12Device gpu;
        private readonly int m, n, k;
        private readonly GpuBuffer x, wq, wd, bias, y, readback, nan;

        public RandomGemmResidentOnGpu(D3D12Device gpu, int m, int n, int k)
        {
            this.gpu = gpu;
            (this.m, this.n, this.k) = (m, n, k);
            var random = new Random(m + n + k);
            nan = Upload(checked(m * n * 4), p => new Span<float>((void*)p, m * n).Fill(float.NaN));
            x = Upload(checked(m * k * 4), p => { var f = (float*)p; for (var i = 0; i < m * k; i++) f[i] = (float)(random.NextDouble() * 2 - 1); });
            wq = Upload(n * k, p => random.NextBytes(new Span<byte>((void*)p, n * k)));
            wd = Upload((n * k / 32 + 1) / 2 * 4, p => { var h = (ushort*)p; for (var i = 0; i < n * k / 32; i++) h[i] = BitConverter.HalfToUInt16Bits((Half)(random.NextDouble() / 50)); });
            bias = Upload(n * 4, p => { var f = (float*)p; for (var i = 0; i < n; i++) f[i] = (float)(random.NextDouble() - 0.5); });
            y = gpu.CreateDefaultBuffer((ulong)m * (ulong)n * 4, allowUav: true);
            readback = gpu.CreateReadbackBuffer((ulong)m * (ulong)n * 4);
        }

        private GpuBuffer Upload(int bytes, Action<nint> fill)
        {
            var buffer = gpu.CreateDefaultBuffer((ulong)bytes, allowUav: false);
            using var staging = gpu.CreateUploadBuffer((ulong)bytes);
            fill((nint)staging.Map());
            staging.Unmap();
            gpu.BeginCommands();
            gpu.Transition(buffer, D3D12_RESOURCE_STATE_COPY_DEST);
            CopyBufferRegion(gpu.CommandList, buffer.Resource, 0, staging.Resource, 0, (ulong)bytes);
            gpu.Transition(buffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.ExecuteAndWait();
            return buffer;
        }

        public double BestMsPerDispatch(nint pipeline, Variant variant)
        {
            var best = double.MaxValue;
            for (var attempt = 0; attempt < OneWarmupAndThreeTimedSubmits; attempt++)
            {
                gpu.BeginCommands();
                if (attempt == NaNFillingWarmupSubmit)
                {
                    gpu.Transition(y, D3D12_RESOURCE_STATE_COPY_DEST);
                    CopyBufferRegion(gpu.CommandList, y.Resource, 0, nan.Resource, 0, (ulong)m * (ulong)n * 4);
                }
                gpu.Transition(y, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                for (var r = 0; r < (attempt == NaNFillingWarmupSubmit ? 1 : DispatchesPerTimedSubmit); r++)
                {
                    SetPipelineState(gpu.CommandList, pipeline);
                    Span<uint> constants = [(uint)m, (uint)n, (uint)k, 1u, 0u, 0u, 0u, 0u];
                    gpu.SetRootConstants(constants);
                    gpu.SetSrv(0, x); gpu.SetSrv(1, wq); gpu.SetSrv(2, wd); gpu.SetSrv(3, bias);
                    gpu.SetUav(0, y);
                    Dispatch(gpu.CommandList, (uint)((n + variant.Tile - 1) / variant.Tile), (uint)((m + variant.Tile - 1) / variant.Tile), 1);
                    gpu.UavBarrier(y);
                }
                var clock = Stopwatch.StartNew();
                gpu.ExecuteAndWait();
                if (attempt > NaNFillingWarmupSubmit)
                    best = Math.Min(best, clock.Elapsed.TotalMilliseconds / DispatchesPerTimedSubmit);
            }
            return best;
        }

        public float[] Read()
        {
            gpu.BeginCommands();
            gpu.Transition(y, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CopyBufferRegion(gpu.CommandList, readback.Resource, 0, y.Resource, 0, (ulong)m * (ulong)n * 4);
            gpu.ExecuteAndWait();
            var result = new float[m * n];
            new ReadOnlySpan<float>((float*)readback.Map(forRead: true), result.Length).CopyTo(result);
            readback.Unmap(wasWritten: false);
            return result;
        }

        public void Dispose()
        {
            readback.Dispose(); y.Dispose(); bias.Dispose(); wd.Dispose(); wq.Dispose(); x.Dispose(); nan.Dispose();
        }
    }
}
