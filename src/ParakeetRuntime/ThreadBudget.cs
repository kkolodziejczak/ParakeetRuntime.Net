using System.Diagnostics;
using System.Runtime.InteropServices;

namespace ParakeetRuntime;

// One logical processor: its processor group, its physical core, and its efficiency class
// (higher is faster; on hybrid CPUs the P-cores have the highest).
internal readonly record struct CpuCore(int Group, int Core, int EfficiencyClass);

// How many threads the CPU engine uses: one per fast core the process may run on, at most Most,
// so the remaining cores stay with the app, the call and audio capture. Fast cores are the
// highest efficiency class (P-cores on hybrid CPUs); the hyper-threads of one core count once.
// Measured 2026-11-22 on a Core Ultra 7 270K Plus: 8 threads on its 8 P-cores ran about twice as
// fast as the earlier fixed 4 threads, which Windows mostly put on E-cores.
internal static unsafe partial class ThreadBudget
{
    public const int Most = 8;

    public static int For(IEnumerable<CpuCore> logicalProcessors)
    {
        var processors = logicalProcessors.ToList();
        if (processors.Count == 0)
            return Math.Min(4, Environment.ProcessorCount);
        var fastest = processors.Max(processor => processor.EfficiencyClass);
        var fastCores = processors.Where(processor => processor.EfficiencyClass == fastest).Select(processor => (processor.Group, processor.Core)).Distinct().Count();
        return Math.Min(fastCores, Most);
    }

    // The logical processors this process may run on. Its affinity mask covers its first processor
    // group only, as .NET's ProcessorAffinity does.
    public static int ForThisProcess()
    {
        var affinity = (ulong)Process.GetCurrentProcess().ProcessorAffinity;
        return For(LogicalProcessors().Where(processor => processor.Group != 0 || (affinity >> processor.Index & 1) != 0).Select(processor => processor.Core));
    }

    private static List<(int Index, int Group, CpuCore Core)> LogicalProcessors()
    {
        var processors = new List<(int, int, CpuCore)>();
        GetSystemCpuSetInformation(null, 0, out var length, 0, 0);
        if (length == 0)
            return processors;
        var buffer = new byte[length];
        fixed (byte* start = buffer)
        {
            if (!GetSystemCpuSetInformation(start, length, out length, 0, 0))
                return processors;
            // SYSTEM_CPU_SET_INFORMATION: Size (4), Type (4), then the CpuSet fields Id (4), Group (2),
            // LogicalProcessorIndex, CoreIndex, LastLevelCacheIndex, NumaNodeIndex, EfficiencyClass (1 each).
            for (var entry = start; entry < start + length; entry += *(uint*)entry)
            {
                int group = *(ushort*)(entry + 12), index = entry[14], core = entry[15], efficiency = entry[18];
                processors.Add((index, group, new CpuCore(group, core, efficiency)));
            }
        }
        return processors;
    }

    [LibraryImport("kernel32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GetSystemCpuSetInformation(byte* information, uint bufferLength, out uint returnedLength, nint process, uint flags);
}
