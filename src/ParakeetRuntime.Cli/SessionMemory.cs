using System.Diagnostics;
using System.Globalization;
using ParakeetRuntime;
using NAudio.Wave;

internal static class SessionMemory
{
    public static async Task<int> RunAsync(string[] args)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        string? Value(string name)
        {
            var index = Array.IndexOf(args, name);
            return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
        }

        var samples = ReadClip(Value("--session-memory") ?? throw new ArgumentException("Missing clip after --session-memory."));
        var device = Value("--device") ?? "gpu";
        var sessions = int.Parse(Value("--sessions") ?? "3");
        var idle = int.Parse(Value("--idle-seconds") ?? "10");
        var model = SpeechModel.Default();

        Report("before the first session");
        var finished = new List<SpeechEngine>();
        for (var session = 1; session <= sessions; session++)
        {
            var engine = device == "cpu"
                ? new SpeechEngine(new ParakeetModel(model.Path, allowGpu: false))
                : await SpeechEngine.LoadAsync(model);
            await engine.TranscribeAsync(samples);
            Report($"session {session} running");
            var dispose = Stopwatch.StartNew();
            engine.Dispose();
            Console.WriteLine($"session {session} disposed in {dispose.Elapsed.TotalMilliseconds:N0} ms");
            finished.Add(engine);
            for (var second = 0; second <= idle; second++)
            {
                if (second is 0 or 1 or 5 || second == idle)
                    Report($"session {session} ended +{second,2} s");
                await Task.Delay(1000);
            }
        }
        GC.KeepAlive(finished);
        return 0;
    }

    private static void Report(string moment)
    {
        using var process = Process.GetCurrentProcess();
        var (dedicated, shared) = GpuMemory.Read(process.Id);
        var gc = GC.GetGCMemoryInfo();
        Console.WriteLine($"{moment,-28} private {process.PrivateMemorySize64 >> 20,6:N0} MB  working set {process.WorkingSet64 >> 20,6:N0} MB  " +
                          $"GC heap {gc.HeapSizeBytes >> 20,5:N0} MB (committed {gc.TotalCommittedBytes >> 20,5:N0} MB)  " +
                          $"GPU dedicated {dedicated >> 20,5:N0} MB, shared {shared >> 20,5:N0} MB  gen2 GCs {GC.CollectionCount(2)}");
    }

    private static float[] ReadClip(string path)
    {
        using var reader = new WaveFileReader(path);
        if (reader.WaveFormat.SampleRate != Wav.SampleRate || reader.WaveFormat.Channels != 1)
            throw new InvalidDataException($"{path} is not 16 kHz mono.");
        var samples = new float[reader.SampleCount];
        var read = reader.ToSampleProvider().Read(samples, 0, samples.Length);
        return samples[..read];
    }
}
