// Part of ParakeetRuntime.Net; a C# port of NeMo-Speech.cpp (Apache-2.0), see THIRD-PARTY-NOTICES.md.
namespace ParakeetRuntime.Net;

internal sealed class Mel
{
    readonly int nFft, hop, win, nMels, nBins;
    readonly float preemph;
    readonly float[] window, fb;
    readonly int[] firstNonzeroBin, endNonzeroBin;
    readonly double[] stageTwiddleCos, stageTwiddleSin;
    const float LogGuard = 5.9604645e-8f;

    public Mel(Gguf g)
    {
        nFft = g.Int("asr.preprocessor.n_fft");
        int sr = g.Int("asr.preprocessor.sample_rate");
        hop = (int)Math.Round(g.Float("asr.preprocessor.window_stride") * sr);
        win = (int)Math.Round(g.Float("asr.preprocessor.window_size") * sr);
        nMels = g.Int("asr.preprocessor.features");
        preemph = g.Float("asr.preprocessor.preemph");
        nBins = nFft / 2 + 1;
        fb = g.Floats("preprocessor.fb");
        window = SymmetricHannWindow(win);
        firstNonzeroBin = new int[nMels]; endNonzeroBin = new int[nMels];
        for (int m = 0; m < nMels; m++)
        {
            int first = 0, end = nBins;
            while (first < end && fb[m * nBins + first] == 0) first++;
            while (end > first && fb[m * nBins + end - 1] == 0) end--;
            (firstNonzeroBin[m], endNonzeroBin[m]) = (first, end);
        }
        stageTwiddleCos = new double[nFft]; stageTwiddleSin = new double[nFft];
        for (int len = 2, at = 0; len <= nFft; at += len / 2, len <<= 1)
        {
            double ang = -2 * Math.PI / len;
            for (int k = 0; k < len / 2; k++) { stageTwiddleCos[at + k] = Math.Cos(ang * k); stageTwiddleSin[at + k] = Math.Sin(ang * k); }
        }
    }

    static float[] SymmetricHannWindow(int win)
    {
        var window = new float[win];
        for (int i = 0; i < win; i++) window[i] = (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / (win - 1)));
        return window;
    }

    public int Mels => nMels;
    public int NFft => nFft;
    public int Hop => hop;
    public int Win => win;
    public float Preemph => preemph;
    public float[] Window => window;
    public float[] Filterbank => fb;

    public float[] BuildTwiddlesF32()
    {
        var packed = new float[nFft * 2];
        for (int i = 0; i < nFft - 1; i++)
        {
            packed[i + 1] = (float)stageTwiddleCos[i];
            packed[nFft + i + 1] = (float)stageTwiddleSin[i];
        }
        return packed;
    }

    public float[] Compute(float[] audio, out int frames)
    {
        int n = audio.Length;
        var padded = new float[n + nFft];
        for (int i = 0; i < n; i++) padded[nFft / 2 + i] = i == 0 ? audio[0] : audio[i] - preemph * audio[i - 1];
        frames = (padded.Length - nFft) / hop + 1;
        var feats = new float[frames * nMels];
        int woff = (nFft - win) / 2, valid = Math.Min(n / hop, frames), nf = frames;
        Kernels.ForRanges(frames, (start, end) =>
        {
            var re = new double[nFft]; var im = new double[nFft]; var power = new float[nBins];
            for (int f = start; f < end; f++)
            {
                Array.Clear(re); Array.Clear(im);
                for (int i = 0; i < win; i++) re[woff + i] = padded[f * hop + woff + i] * window[i];
                Fft(re, im);
                for (int k = 0; k < nBins; k++) power[k] = (float)(re[k] * re[k] + im[k] * im[k]);
                for (int m = 0; m < nMels; m++)
                {
                    float acc = 0;
                    for (int k = firstNonzeroBin[m]; k < endNonzeroBin[m]; k++) acc += fb[m * nBins + k] * power[k];
                    feats[f * nMels + m] = f < valid ? MathF.Log(acc + LogGuard) : 0f;
                }
            }
        });
        for (int m = 0; m < nMels; m++)
        {
            double sum = 0; for (int f = 0; f < valid; f++) sum += feats[f * nMels + m];
            double mean = valid > 0 ? sum / valid : 0, var = 0;
            for (int f = 0; f < valid; f++) { double d = feats[f * nMels + m] - mean; var += d * d; }
            if (valid > 1) var /= valid - 1;
            double inv = 1.0 / (Math.Sqrt(var) + 1e-5);
            for (int f = 0; f < nf; f++) feats[f * nMels + m] = f < valid ? (float)((feats[f * nMels + m] - mean) * inv) : 0f;
        }
        return feats;
    }

    void Fft(double[] re, double[] im)
    {
        int n = re.Length;
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { (re[i], re[j]) = (re[j], re[i]); (im[i], im[j]) = (im[j], im[i]); }
        }
        for (int len = 2, at = 0; len <= n; at += len / 2, len <<= 1)
        {
            for (int i = 0; i < n; i += len)
                for (int k = 0; k < len / 2; k++)
                {
                    double wr = stageTwiddleCos[at + k], wi = stageTwiddleSin[at + k];
                    int a = i + k, b = a + len / 2;
                    double xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
                }
        }
    }
}
