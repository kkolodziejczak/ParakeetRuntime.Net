using System.Diagnostics;

namespace ParakeetRuntime;

internal sealed class StageClock
{
    private readonly Dictionary<string, long> ticks = [];
    private long last;

    public void Start() => last = Stopwatch.GetTimestamp();

    public void Lap(string stage)
    {
        var now = Stopwatch.GetTimestamp();
        ticks[stage] = ticks.GetValueOrDefault(stage) + now - last;
        last = now;
    }

    public IEnumerable<(string Stage, TimeSpan Time)> Totals =>
        ticks.Select(entry => (entry.Key, Stopwatch.GetElapsedTime(0, entry.Value)));
}
