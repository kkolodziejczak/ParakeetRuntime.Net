namespace ParakeetRuntime.Net;

internal sealed class GpuOperationTimes
{
    private readonly Dictionary<string, (double Milliseconds, long Count)> totals = [];

    public long MissedOperations { get; set; }

    public IEnumerable<(string Operation, double Milliseconds, long Count)> Totals =>
        totals.OrderByDescending(t => t.Value.Milliseconds).Select(t => (t.Key, t.Value.Milliseconds, t.Value.Count));

    public void Add(string operation, double milliseconds)
    {
        var (sum, count) = totals.GetValueOrDefault(operation);
        totals[operation] = (sum + milliseconds, count + 1);
    }

    public void Clear()
    {
        totals.Clear();
        MissedOperations = 0;
    }
}
