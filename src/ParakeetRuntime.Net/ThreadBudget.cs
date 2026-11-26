using System.Diagnostics;
using System.Runtime.InteropServices;

namespace ParakeetRuntime.Net;

internal readonly record struct CpuCore(int Group, int Core, int EfficiencyClass);

internal static unsafe partial class ThreadBudget
{
    public const int Most = 8;
    private const int WithoutCoreInformation = 4;

    public static int For(IEnumerable<CpuCore> logicalProcessors)
    {
        var processors = logicalProcessors.ToList();
        if (processors.Count == 0)
            return Math.Min(WithoutCoreInformation, Environment.ProcessorCount);
        var fastest = processors.Max(processor => processor.EfficiencyClass);
        var fastCores = processors.Where(processor => processor.EfficiencyClass == fastest).Select(processor => (processor.Group, processor.Core)).Distinct().Count();
        return Math.Min(fastCores, Most);
    }

    public static int ForThisProcess()
    {
        var firstGroupAffinity = (ulong)Process.GetCurrentProcess().ProcessorAffinity;
        return For(LogicalProcessors().Where(processor => processor.Group != 0 || (firstGroupAffinity >> processor.Index & 1) != 0).Select(processor => processor.Core));
    }

    private const int CpuSetGroupOffset = 12;
    private const int CpuSetLogicalProcessorIndexOffset = 14;
    private const int CpuSetCoreIndexOffset = 15;
    private const int CpuSetEfficiencyClassOffset = 18;

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
            for (var entry = start; entry < start + length; entry += *(uint*)entry)
            {
                int group = *(ushort*)(entry + CpuSetGroupOffset), index = entry[CpuSetLogicalProcessorIndexOffset], core = entry[CpuSetCoreIndexOffset], efficiency = entry[CpuSetEfficiencyClassOffset];
                processors.Add((index, group, new CpuCore(group, core, efficiency)));
            }
        }
        return processors;
    }

    [LibraryImport("kernel32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GetSystemCpuSetInformation(byte* information, uint bufferLength, out uint returnedLength, nint process, uint flags);
}
