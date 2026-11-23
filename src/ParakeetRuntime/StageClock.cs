// Where the time of one utterance goes, for the Spike's corpus benchmark. The app never sets one.
using System.Diagnostics;

namespace ParakeetRuntime;

internal sealed class StageClock
{
    private readonly Dictionary<string, long> ticks = [];
    private long last;

    public void Start() => last = Stopwatch.GetTimestamp();

    // Charges the time since the previous lap (or Start) to this stage.
    public void Lap(string stage)
    {
        var now = Stopwatch.GetTimestamp();
        ticks[stage] = ticks.GetValueOrDefault(stage) + now - last;
        last = now;
    }

    public IEnumerable<(string Stage, TimeSpan Time)> Totals =>
        ticks.Select(entry => (entry.Key, Stopwatch.GetElapsedTime(0, entry.Value)));
}
