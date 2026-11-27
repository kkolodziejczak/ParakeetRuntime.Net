namespace ParakeetRuntime.Net;

internal static class QuietestPointCutter
{
    internal const int LongestPiece = 20 * Wav.SampleRate;
    private const int FrameSamples = Wav.SampleRate * 30 / 1000;
    private const int QuietWindowFrames = 10;
    private const int FramesInLongestPiece = LongestPiece / FrameSamples;

    public static string TranscribeInPieces(float[] samples, Func<float[], string> transcribe, CancellationToken stopBetweenPieces)
    {
        var texts = new List<string>();
        foreach (var piece in Pieces(samples))
        {
            stopBetweenPieces.ThrowIfCancellationRequested();
            var audio = piece.GetOffsetAndLength(samples.Length).Length == samples.Length ? samples : samples[piece];
            texts.Add(transcribe(audio).Trim());
        }
        return string.Join(' ', texts.Where(text => text.Length > 0));
    }

    private static List<Range> Pieces(float[] samples)
    {
        if (samples.Length <= LongestPiece)
            return [Range.All];
        var loudness = FrameLoudness(samples);
        var pieces = new List<Range>();
        var startFrame = 0;
        while (samples.Length - startFrame * FrameSamples > LongestPiece)
        {
            var cutFrame = QuietestWindowCentre(loudness, startFrame);
            pieces.Add((startFrame * FrameSamples)..(cutFrame * FrameSamples));
            startFrame = cutFrame;
        }
        pieces.Add((startFrame * FrameSamples)..);
        return pieces;
    }

    private static float[] FrameLoudness(float[] samples)
    {
        var frames = new float[samples.Length / FrameSamples];
        for (var frame = 0; frame < frames.Length; frame++)
        {
            double sum = 0;
            foreach (var sample in samples.AsSpan(frame * FrameSamples, FrameSamples))
                sum += sample * sample;
            frames[frame] = (float)Math.Sqrt(sum / FrameSamples);
        }
        return frames;
    }

    private static int QuietestWindowCentre(float[] loudness, int startFrame)
    {
        var quietest = startFrame + FramesInLongestPiece;
        var quietestLoudness = double.MaxValue;
        for (var centre = startFrame + FramesInLongestPiece / 2; centre <= startFrame + FramesInLongestPiece - QuietWindowFrames / 2; centre++)
        {
            double sum = 0;
            foreach (var frame in loudness.AsSpan(centre - QuietWindowFrames / 2, QuietWindowFrames))
                sum += frame;
            if (sum <= quietestLoudness)
            {
                quietestLoudness = sum;
                quietest = centre;
            }
        }
        return quietest;
    }
}
