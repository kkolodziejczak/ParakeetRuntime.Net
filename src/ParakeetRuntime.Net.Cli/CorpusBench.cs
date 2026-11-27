using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Text.Json;
using ParakeetRuntime.Net;
using NAudio.Wave;

internal static class CorpusBench
{
    public static async Task<int> RunAsync(string[] args)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        string? Value(string name)
        {
            var index = Array.IndexOf(args, name);
            return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
        }

        if (args.Contains("--list-gpus"))
        {
            ListGpus();
            return 0;
        }
        if (Value("--affinity") is { } mask)
            Process.GetCurrentProcess().ProcessorAffinity = (nint)long.Parse(mask.Replace("0x", "", StringComparison.OrdinalIgnoreCase), NumberStyles.HexNumber);
        if (Value("--threads") is { } threads)
            Kernels.OverrideThreadBudgetForBenchmark(int.Parse(threads));
        var corpus = Value("--corpus") ?? throw new ArgumentException("Missing path after --corpus.");
        var reference = Value("--reference") is { } r ? Tsv.Read(r, key: 1, value: 2) : null;
        var compare = Value("--compare") is { } c ? Tsv.Read(c, key: 0, value: 1) : null;
        var maxSeconds = double.Parse(Value("--max-seconds") ?? "Infinity", CultureInfo.InvariantCulture);
        var clips = ClipPaths(corpus)
            .Where(path => compare?.ContainsKey(Path.GetFileName(path)) ?? true)
            .Where(path => reference?.ContainsKey(Path.GetFileName(path)) ?? true)
            .Select(Clip.Read)
            .Where(clip => clip.Seconds <= maxSeconds)
            .Take(int.Parse(Value("--limit") ?? int.MaxValue.ToString()))
            .ToList();
        if (clips.Count == 0)
        {
            Console.WriteLine("No WAV clips matched.");
            return 1;
        }
        var passes = int.Parse(Value("--passes") ?? "1");
        var cut = args.Contains("--cut");
        if (cut && Value("--nemo") is not null)
            throw new ArgumentException("--cut runs only on ParakeetRuntime.Net, not with --nemo.");
        var timeGpuOperations = args.Contains("--gpu-operations");
        if (timeGpuOperations && Value("--nemo") is not null)
            throw new ArgumentException("--gpu-operations runs only on ParakeetRuntime.Net, not with --nemo.");
        var modelPath = Value("--model") ?? SpeechModel.Default().Path;
        var device = Value("--device") ?? "cpu";
        var audioSeconds = clips.Sum(clip => clip.Seconds);

        var load = Stopwatch.StartNew();
        await using IBenchEngine engine = Value("--nemo") is { } nemo
            ? await NemoEngine.StartAsync(nemo, modelPath, device, Process.GetCurrentProcess().ProcessorAffinity)
            : LocalEngine.Load(modelPath, device, Value("--int8-path") is { } path ? Enum.Parse<Int8Path>(path) : Kernels.BestPath, cut, timeGpuOperations);
        load.Stop();
        var gpuMemory = new GpuMemory(engine.ProcessId);
        gpuMemory.Sample();
        Console.WriteLine($"{engine.Name}, affinity 0x{Process.GetCurrentProcess().ProcessorAffinity:X}, loaded in {load.Elapsed.TotalSeconds:0.00} s, " +
                          $"memory after load {engine.WorkingSetMegabytes:N0} MB, peak during load {engine.PeakWorkingSetMegabytes:N0} MB, GPU {gpuMemory}");
        Console.WriteLine($"{clips.Count} clips, {audioSeconds:0.0} s audio, longest {clips.Max(clip => clip.Seconds):0.0} s");

        await engine.TranscribeAsync(clips[0], clips[0].ReadSamplesAgain());
        engine.ResetClock();
        var texts = new Dictionary<string, string>();
        var best = TimeSpan.MaxValue;
        for (var pass = 1; pass <= passes; pass++)
        {
            var compute = TimeSpan.Zero;
            var (gcPause, fullCollections) = (GC.GetTotalPauseDuration(), GC.CollectionCount(2));
            foreach (var clip in clips)
            {
                var samples = clip.ReadSamplesAgain();
                var watch = Stopwatch.StartNew();
                var text = await engine.TranscribeAsync(clip, samples);
                compute += watch.Elapsed;
                texts.TryAdd(clip.Name, text);
                gpuMemory.Sample();
            }
            best = compute < best ? compute : best;
            Console.WriteLine($"pass {pass}: {compute.TotalSeconds:0.00} s compute, RTF {compute.TotalSeconds / audioSeconds:0.00000}, " +
                              $"GC pauses {(GC.GetTotalPauseDuration() - gcPause).TotalSeconds:0.00} s, {GC.CollectionCount(2) - fullCollections} full collections");
        }
        foreach (var line in engine.StageReport())
            Console.WriteLine(line);
        Console.WriteLine($"memory peak {engine.PeakWorkingSetMegabytes:N0} MB, GPU peak {gpuMemory}");

        if (Value("--out") is { } output)
            File.WriteAllLines(output, clips.Select(clip => $"{clip.Name}\t{texts[clip.Name]}"), new UTF8Encoding(false));
        var referenceWer = reference is null ? null : Wer.Measure(clips.Select(clip => (reference[clip.Name], texts[clip.Name])));
        var compareWer = compare is null ? null : Wer.Measure(clips.Select(clip => (compare[clip.Name], texts[clip.Name])));
        var identical = compare is null ? 0 : clips.Count(clip => compare[clip.Name] == texts[clip.Name]);
        if (referenceWer is not null)
            Console.WriteLine($"WER vs reference: {referenceWer}");
        if (compareWer is not null)
            Console.WriteLine($"vs {Path.GetFileName(Value("--compare"))}: {compareWer}, identical text in {identical}/{clips.Count}");

        if (Value("--results") is { } results)
        {
            if (!File.Exists(results))
                File.WriteAllText(results, "engine\tdevice\tcorpus\tclips\taudio_s\tbest_rtf\twer\tdiff_vs_compare\tidentical\tload_s\tpeak_ram_mb\tpeak_gpu_dedicated_mb\tpeak_gpu_shared_mb\n");
            File.AppendAllText(results, string.Join('\t',
                engine is NemoEngine ? "NeMo-Speech.cpp" : "ParakeetRuntime.Net", device, CorpusLabel(corpus, maxSeconds, cut), clips.Count, audioSeconds.ToString("0.0"),
                (best.TotalSeconds / audioSeconds).ToString("0.00000"), referenceWer?.Percent ?? "", compareWer?.Percent ?? "",
                compare is null ? "" : $"{identical}/{clips.Count}", load.Elapsed.TotalSeconds.ToString("0.00"),
                engine.PeakWorkingSetMegabytes, gpuMemory.PeakDedicatedMegabytes, gpuMemory.PeakSharedMegabytes) + "\n");
        }
        return 0;
    }

    private static void ListGpus()
    {
        var adapters = D3D12Device.EnumerateAdapters();
        if (adapters.Count == 0)
            Console.WriteLine("No DXGI adapters detected.");
        for (var i = 0; i < adapters.Count; i++)
        {
            var a = adapters[i];
            Console.WriteLine($"[{i}] {a.Name} | dedicated={a.DedicatedVideoMemoryBytes >> 20} MB | shared={a.SharedSystemMemoryBytes >> 20} MB | " +
                              $"budget={a.MemoryBudgetBytes >> 20} MB | software={a.IsSoftware} | eligible={a.MeetsMinimumMemory}");
        }
    }

    private static IEnumerable<string> ClipPaths(string listFileOrWavFolder) => Directory.Exists(listFileOrWavFolder)
        ? Directory.EnumerateFiles(listFileOrWavFolder, "*.wav").Order(StringComparer.Ordinal)
        : File.ReadAllLines(listFileOrWavFolder).Where(line => line.Length > 0);

    private static string CorpusLabel(string corpus, double maxSeconds, bool cut)
    {
        var folder = Directory.Exists(corpus) ? Path.GetFullPath(corpus) : Path.GetDirectoryName(Path.GetFullPath(corpus))!;
        var label = $"{Path.GetFileName(Path.GetDirectoryName(folder))}/{Path.GetFileName(folder)}";
        label = double.IsInfinity(maxSeconds) ? label : $"{label}<={maxSeconds}s";
        return cut ? $"{label} cut" : label;
    }

    private sealed record Clip(string Path, string Name, double Seconds)
    {
        public static Clip Read(string path)
        {
            using var reader = Open(path);
            return new Clip(path, System.IO.Path.GetFileName(path), reader.SampleCount / (double)Wav.SampleRate);
        }

        public float[] ReadSamplesAgain()
        {
            using var reader = Open(Path);
            var samples = new float[reader.SampleCount];
            var read = reader.ToSampleProvider().Read(samples, 0, samples.Length);
            return samples[..read];
        }

        private static WaveFileReader Open(string path)
        {
            var reader = new WaveFileReader(path);
            if (reader.WaveFormat.SampleRate == Wav.SampleRate && reader.WaveFormat.Channels == 1)
                return reader;
            reader.Dispose();
            throw new InvalidDataException($"{path} is not 16 kHz mono.");
        }
    }

    private interface IBenchEngine : IAsyncDisposable
    {
        string Name { get; }
        int ProcessId { get; }
        long WorkingSetMegabytes { get; }
        long PeakWorkingSetMegabytes { get; }
        Task<string> TranscribeAsync(Clip clip, float[] samples);
        void ResetClock();
        IEnumerable<string> StageReport();
    }

    private sealed class LocalEngine(ParakeetModel model, Int8Path int8Path, bool cutAtQuietestPoints, GpuOperationTimes? gpuOperationTimes = null) : IBenchEngine
    {
        private double gpuEncoderMs, gpuDecoderMs;

        public static LocalEngine Load(string path, string device, Int8Path int8Path, bool cutAtQuietestPoints, bool timeGpuOperations)
        {
            if (device == "cpu")
            {
                if (timeGpuOperations)
                    throw new ArgumentException("--gpu-operations needs --device gpu or gpu:<adapter>.");
                return new LocalEngine(new ParakeetModel(path, int8Path), int8Path, cutAtQuietestPoints);
            }
            if (device != "gpu" && !device.StartsWith("gpu:", StringComparison.Ordinal))
                throw new ArgumentException($"ParakeetRuntime.Net's --device is cpu, gpu or gpu:<adapter>, not {device}.");
            var model = new ParakeetModel(path, allowGpu: true, device == "gpu" ? null : device["gpu:".Length..]);
            if (!model.IsUsingGpu)
            {
                model.Dispose();
                throw new InvalidOperationException($"No usable GPU matched {device}; see --list-gpus.");
            }
            var operationTimes = timeGpuOperations ? model.TimeEveryGpuOperation() : null;
            if (timeGpuOperations && operationTimes is null)
            {
                model.Dispose();
                throw new InvalidOperationException($"{model.BackendName} has no GPU timestamps, so --gpu-operations can't time it.");
            }
            return new LocalEngine(model, int8Path, cutAtQuietestPoints, operationTimes);
        }

        public string Name => model.IsUsingGpu
            ? $"ParakeetRuntime.Net {model.BackendName}"
            : $"ParakeetRuntime.Net CPU, int8 path {int8Path} (AVX2 {Avx2.IsSupported}, AVX-VNNI {AvxVnni.IsSupported}, AVX-VNNI-INT8 {AvxVnniInt8.IsSupported}), {Kernels.ThreadCount} threads, {Environment.ProcessorCount} processors";

        public int ProcessId => Environment.ProcessId;

        public long WorkingSetMegabytes => Environment.WorkingSet >> 20;

        public long PeakWorkingSetMegabytes => Process.GetCurrentProcess().PeakWorkingSet64 >> 20;

        public Task<string> TranscribeAsync(Clip clip, float[] samples) => Task.FromResult(cutAtQuietestPoints
            ? QuietestPointCutter.TranscribeInPieces(samples, TranscribeInOneCall, CancellationToken.None)
            : TranscribeInOneCall(samples));

        private string TranscribeInOneCall(float[] samples)
        {
            var ranOnTheGpu = model.IsUsingGpu;
            var text = model.Transcribe(samples, out var profile);
            if (ranOnTheGpu && !model.IsUsingGpu)
                throw new InvalidOperationException("The GPU was lost during the run and the model went on on the CPU; the timings would mix both.");
            if (model.IsUsingGpu)
            {
                gpuEncoderMs += profile.EncoderMs;
                gpuDecoderMs += profile.DecodeMs;
            }
            return text;
        }

        public void ResetClock()
        {
            model.BenchmarkClock = new StageClock();
            gpuEncoderMs = gpuDecoderMs = 0;
            gpuOperationTimes?.Clear();
        }

        public IEnumerable<string> StageReport()
        {
            var totals = model.IsUsingGpu
                ? [("encoder", TimeSpan.FromMilliseconds(gpuEncoderMs)), ("decoder", TimeSpan.FromMilliseconds(gpuDecoderMs))]
                : model.BenchmarkClock!.Totals.ToList();
            var all = totals.Sum(stage => stage.Item2.TotalSeconds);
            var stages = totals.Select(stage => $"  {stage.Item1,-14} {stage.Item2.TotalSeconds,7:0.00} s  {100 * stage.Item2.TotalSeconds / all,5:0.0}%");
            if (gpuOperationTimes is null)
                return stages;
            var timed = gpuOperationTimes.Totals.ToList();
            var gpuMs = timed.Sum(operation => operation.Milliseconds);
            if (gpuMs <= 0)
                return stages.Append("  GPU time per operation: nothing was timed.");
            var missed = gpuOperationTimes.MissedOperations > 0 ? $", {gpuOperationTimes.MissedOperations:N0} operations after the last timestamp slot not timed" : "";
            return stages.Append($"  GPU time per operation, all passes, {gpuMs / 1000:0.00} s{missed}:")
                .Concat(timed.Select(operation => $"    {operation.Operation,-30} {operation.Milliseconds / 1000,7:0.00} s  {100 * operation.Milliseconds / gpuMs,5:0.0}%  {operation.Count,8:N0} times, {1000 * operation.Milliseconds / operation.Count,8:0.0} us each"));
        }

        public ValueTask DisposeAsync()
        {
            model.Dispose();
            return ValueTask.CompletedTask;
        }
    }

    private sealed class NemoEngine(Process process, HttpClient http, string device) : IBenchEngine
    {
        public static async Task<NemoEngine> StartAsync(string executable, string model, string device, nint affinity)
        {
            var port = FreePort();
            var process = Process.Start(new ProcessStartInfo(executable,
                ["serve", "--host", "127.0.0.1", "--port", port.ToString(), "--threads", "1", "--no-ui", "--asr-model", model, "--device", device])
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            })!;
            process.ProcessorAffinity = affinity;
            process.OutputDataReceived += (_, _) => { };
            process.ErrorDataReceived += (_, _) => { };
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            var http = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}"), Timeout = TimeSpan.FromMinutes(5) };
            var engine = new NemoEngine(process, http, device);
            for (var watch = Stopwatch.StartNew(); watch.Elapsed < TimeSpan.FromMinutes(2) && !process.HasExited; await Task.Delay(50))
            {
                try
                {
                    if ((await http.GetAsync("/ready")).IsSuccessStatusCode)
                        return engine;
                }
                catch (Exception exception) when (exception is HttpRequestException or TaskCanceledException)
                {
                }
            }
            await engine.DisposeAsync();
            throw new TimeoutException("NeMo-Speech.cpp did not get ready within 2 minutes.");
        }

        private static int FreePort()
        {
            using var listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start();
            return ((IPEndPoint)listener.LocalEndpoint).Port;
        }

        public string Name => $"NeMo-Speech.cpp serve, {device}";

        public int ProcessId => process.Id;

        public long WorkingSetMegabytes { get { process.Refresh(); return process.WorkingSet64 >> 20; } }

        public long PeakWorkingSetMegabytes { get { process.Refresh(); return process.PeakWorkingSet64 >> 20; } }

        public async Task<string> TranscribeAsync(Clip clip, float[] samples)
        {
            using var form = new MultipartFormDataContent
            {
                { new ByteArrayContent(await File.ReadAllBytesAsync(clip.Path)) { Headers = { ContentType = new("audio/wav") } }, "file", "u.wav" },
                { new StringContent("json"), "response_format" },
                { new StringContent("true"), "automatic_punctuation" },
            };
            using var response = await http.PostAsync("/v1/audio/transcriptions", form);
            response.EnsureSuccessStatusCode();
            using var json = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            return json.RootElement.GetProperty("text").GetString()!.Trim();
        }

        public void ResetClock() { }

        public IEnumerable<string> StageReport() => [];

        public ValueTask DisposeAsync()
        {
            http.Dispose();
            process.Kill(entireProcessTree: true);
            process.WaitForExit(10_000);
            process.Dispose();
            return ValueTask.CompletedTask;
        }
    }

    private static class Tsv
    {
        public static Dictionary<string, string> Read(string path, int key, int value) =>
            File.ReadLines(path, Encoding.UTF8)
                .Select(line => line.Split('\t'))
                .Where(cells => cells.Length > Math.Max(key, value))
                .GroupBy(cells => cells[key])
                .ToDictionary(group => group.Key, group => group.Last()[value]);
    }

    private sealed record Wer(int Errors, int Words)
    {
        public string Percent => $"{100.0 * Errors / Words:0.00}%";

        public override string ToString() => $"{Percent} ({Errors} of {Words} words)";

        public static Wer Measure(IEnumerable<(string Expected, string Actual)> pairs)
        {
            int errors = 0, words = 0;
            foreach (var (expected, actual) in pairs)
            {
                var e = LowercaseWordsBetweenNonAlphanumerics(expected);
                errors += Edits(e, LowercaseWordsBetweenNonAlphanumerics(actual));
                words += e.Length;
            }
            return new Wer(errors, words);
        }

        private static string[] LowercaseWordsBetweenNonAlphanumerics(string text)
        {
            var kept = new StringBuilder();
            foreach (var rune in text.ToLowerInvariant().EnumerateRunes())
                kept.Append(Rune.GetUnicodeCategory(rune) is >= UnicodeCategory.UppercaseLetter and <= UnicodeCategory.OtherLetter
                    or UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber
                    ? rune.ToString() : " ");
            return kept.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries);
        }

        private static int Edits(string[] a, string[] b)
        {
            var row = Enumerable.Range(0, b.Length + 1).ToArray();
            for (var i = 1; i <= a.Length; i++)
            {
                int previous = row[0];
                row[0] = i;
                for (var j = 1; j <= b.Length; j++)
                {
                    var substitution = previous + (a[i - 1] == b[j - 1] ? 0 : 1);
                    previous = row[j];
                    row[j] = Math.Min(Math.Min(row[j] + 1, row[j - 1] + 1), substitution);
                }
            }
            return row[b.Length];
        }
    }
}
