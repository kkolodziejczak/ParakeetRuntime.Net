using System.Numerics;

namespace ParakeetRuntime;

internal sealed class ResamplerTo16Khz
{
    private const int HalfTapsAtFullBand = 32;
    private const double KaiserBeta = 8.6;
    private const double PassbandOfOutputNyquist = 0.94;
    private const int MostFilterPhases = 1024;

    private readonly int up;
    private readonly int down;
    private readonly int halfTaps;
    private readonly int taps;
    private readonly int phases;
    private readonly float[]? filters;
    private float[] output;
    private int outputCount;
    private float[] history;
    private long historyStart;
    private int historyCount;
    private long inputCount;

    public ResamplerTo16Khz(int fromRate, long? expectedInputFrames)
    {
        var divisor = (int)BigInteger.GreatestCommonDivisor(fromRate, Wav.SampleRate);
        up = Wav.SampleRate / divisor;
        down = fromRate / divisor;
        var expectedOutput = expectedInputFrames is { } frames ? frames * up / down : 0;
        output = new float[Math.Max(1024, expectedOutput)];
        if (fromRate == Wav.SampleRate)
        {
            history = [];
            return;
        }
        var cutoff = Math.Min(1.0, (double)up / down) * PassbandOfOutputNyquist;
        halfTaps = (int)Math.Ceiling(HalfTapsAtFullBand / cutoff);
        taps = 2 * halfTaps;
        phases = Math.Min(up, MostFilterPhases);
        filters = PhaseFilters(phases, taps, halfTaps, cutoff);
        history = new float[taps + 8192];
    }

    public void Push(ReadOnlySpan<float> mono)
    {
        inputCount += mono.Length;
        if (filters is null)
        {
            Append(mono);
            return;
        }
        KeepOnlyWhatTheNextOutputNeeds();
        if (historyCount + mono.Length > history.Length)
            Array.Resize(ref history, historyCount + mono.Length);
        mono.CopyTo(history.AsSpan(historyCount));
        historyCount += mono.Length;
        ProduceWhileTheWindowIsComplete();
    }

    public float[] Finish()
    {
        if (filters is not null)
        {
            var total = inputCount * up / down;
            while (outputCount < total)
                AppendOne(Convolve(outputCount, windowMayRunOutside: true));
        }
        return output.Length == outputCount ? output : output[..outputCount];
    }

    private void KeepOnlyWhatTheNextOutputNeeds()
    {
        var firstNeeded = Math.Max(0, WindowStart(outputCount));
        var drop = (int)Math.Clamp(firstNeeded - historyStart, 0, historyCount);
        if (drop == 0)
            return;
        history.AsSpan(drop, historyCount - drop).CopyTo(history);
        historyCount -= drop;
        historyStart += drop;
    }

    private void ProduceWhileTheWindowIsComplete()
    {
        while (WindowStart(outputCount) + taps <= historyStart + historyCount)
            AppendOne(Convolve(outputCount, windowMayRunOutside: WindowStart(outputCount) < 0));
    }

    private long WindowStart(long outputIndex) => outputIndex * down / up - halfTaps + 1;

    private float Convolve(long outputIndex, bool windowMayRunOutside)
    {
        var position = outputIndex * down;
        var phase = (int)(position % up * phases / up);
        var filter = filters.AsSpan(phase * taps, taps);
        var first = WindowStart(outputIndex);
        if (!windowMayRunOutside)
            return Dot(history.AsSpan((int)(first - historyStart), taps), filter);
        float sum = 0;
        for (var k = 0; k < taps; k++)
        {
            var at = first + k - historyStart;
            if (first + k >= 0 && at >= 0 && at < historyCount)
                sum += history[at] * filter[k];
        }
        return sum;
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var sums = Vector<float>.Zero;
        var k = 0;
        for (; k <= a.Length - Vector<float>.Count; k += Vector<float>.Count)
            sums += new Vector<float>(a[k..]) * new Vector<float>(b[k..]);
        var sum = Vector.Sum(sums);
        for (; k < a.Length; k++)
            sum += a[k] * b[k];
        return sum;
    }

    private void Append(ReadOnlySpan<float> samples)
    {
        if (outputCount + samples.Length > output.Length)
            Array.Resize(ref output, Math.Max(output.Length * 2, outputCount + samples.Length));
        samples.CopyTo(output.AsSpan(outputCount));
        outputCount += samples.Length;
    }

    private void AppendOne(float sample)
    {
        if (outputCount == output.Length)
            Array.Resize(ref output, output.Length * 2);
        output[outputCount++] = sample;
    }

    private static float[] PhaseFilters(int phases, int taps, int halfTaps, double cutoff)
    {
        var filters = new float[phases * taps];
        var i0Beta = BesselI0(KaiserBeta);
        for (var phase = 0; phase < phases; phase++)
            for (var k = 0; k < taps; k++)
            {
                var x = k - halfTaps + 1 - (double)phase / phases;
                var ratio = x / halfTaps;
                if (Math.Abs(ratio) >= 1)
                    continue;
                var window = BesselI0(KaiserBeta * Math.Sqrt(1 - ratio * ratio)) / i0Beta;
                var arg = Math.PI * x * cutoff;
                filters[phase * taps + k] = (float)(cutoff * (arg == 0 ? 1 : Math.Sin(arg) / arg) * window);
            }
        return filters;
    }

    private static double BesselI0(double x)
    {
        double sum = 1, term = 1, half = x / 2;
        for (var k = 1; k < 30; k++)
        {
            term *= half / k;
            sum += term * term;
        }
        return sum;
    }
}
