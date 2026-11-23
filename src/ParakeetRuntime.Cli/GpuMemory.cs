using System.Diagnostics;

internal sealed class GpuMemory(int processId)
{
    private const string Category = "GPU Process Memory";
    private long dedicated, shared;

    public long PeakDedicatedMegabytes => dedicated >> 20;

    public long PeakSharedMegabytes => shared >> 20;

    public void Sample()
    {
        var (nowDedicated, nowShared) = Read(processId);
        dedicated = Math.Max(dedicated, nowDedicated);
        shared = Math.Max(shared, nowShared);
    }

    public override string ToString() => $"dedicated {PeakDedicatedMegabytes:N0} MB, shared {PeakSharedMegabytes:N0} MB";

    public static (long Dedicated, long Shared) Read(int processId)
    {
        long dedicated = 0, shared = 0;
        foreach (var instance in new PerformanceCounterCategory(Category).GetInstanceNames().Where(name => name.StartsWith($"pid_{processId}_", StringComparison.Ordinal)))
        {
            try
            {
                using var d = new PerformanceCounter(Category, "Dedicated Usage", instance, readOnly: true);
                using var s = new PerformanceCounter(Category, "Shared Usage", instance, readOnly: true);
                dedicated += d.RawValue;
                shared += s.RawValue;
            }
            catch (InvalidOperationException)
            {
            }
        }
        return (dedicated, shared);
    }
}
