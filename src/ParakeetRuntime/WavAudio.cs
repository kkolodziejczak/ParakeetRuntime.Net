using System.Buffers.Binary;

namespace ParakeetRuntime;

internal enum WavProblem { NotWav, Unsupported, Damaged, TooLong }

internal static class WavAudio
{
    public static readonly TimeSpan LongestAudio = TimeSpan.FromMinutes(10);

    private const ushort Pcm = 1;
    private const ushort IeeeFloat = 3;
    private const ushort Extensible = 0xFFFE;
    private const uint UnknownLengthMarker = 0xFFFFFFFF;
    private const int LargestFmtChunk = 1024;
    private const int FramesPerBlock = 8192;
    private static readonly byte[] KsDataFormatGuidTail = Convert.FromHexString("000000001000800000AA00389B71");

    public static float[] Read16KhzMono(Stream wav, TimeSpan longest)
    {
        var header = TryReadExactly(wav, 12);
        if (header is null || !header.AsSpan(0, 4).SequenceEqual("RIFF"u8) || !header.AsSpan(8, 4).SequenceEqual("WAVE"u8))
            throw Rejected(WavProblem.NotWav, "This is not a WAV file (no RIFF/WAVE header). Only WAV audio is supported.");
        Format? format = null;
        while (true)
        {
            var chunk = TryReadExactly(wav, 8) ?? throw Rejected(WavProblem.Damaged, "The WAV file has no audio data.");
            var size = BinaryPrimitives.ReadUInt32LittleEndian(chunk.AsSpan(4));
            if (chunk.AsSpan(0, 4).SequenceEqual("fmt "u8))
            {
                if (size is < 16 or > LargestFmtChunk)
                    throw Rejected(WavProblem.Damaged, "The WAV format chunk has an invalid size.");
                format = Format.Parse(ReadExactly(wav, (int)size, "the format chunk"));
                Skip(wav, size & 1);
            }
            else if (chunk.AsSpan(0, 4).SequenceEqual("data"u8))
            {
                var known = format ?? throw Rejected(WavProblem.Damaged, "The WAV audio data comes before its format.");
                return ReadData(wav, known, size, longest);
            }
            else
            {
                Skip(wav, (long)size + (size & 1));
            }
        }
    }

    private sealed record Format(ushort Encoding, int Channels, int SampleRate, int BitsPerSample)
    {
        public int BytesPerFrame => Channels * BitsPerSample / 8;

        public static Format Parse(byte[] fmt)
        {
            var encoding = BinaryPrimitives.ReadUInt16LittleEndian(fmt);
            var channels = BinaryPrimitives.ReadUInt16LittleEndian(fmt.AsSpan(2));
            var rate = BinaryPrimitives.ReadInt32LittleEndian(fmt.AsSpan(4));
            var blockAlign = BinaryPrimitives.ReadUInt16LittleEndian(fmt.AsSpan(12));
            var bits = BinaryPrimitives.ReadUInt16LittleEndian(fmt.AsSpan(14));
            if (encoding == Extensible)
            {
                if (fmt.Length < 40 || !fmt.AsSpan(26, 14).SequenceEqual(KsDataFormatGuidTail))
                    throw Rejected(WavProblem.Unsupported, "The WAV file's extensible format is not supported. Use PCM or float audio.");
                encoding = BinaryPrimitives.ReadUInt16LittleEndian(fmt.AsSpan(24));
            }
            if ((encoding, bits) is not ((Pcm, 8 or 16 or 24 or 32) or (IeeeFloat, 32 or 64)))
                throw Rejected(WavProblem.Unsupported, $"The WAV encoding is not supported (format {encoding}, {bits} bits). Use PCM at 8, 16, 24 or 32 bits, or float at 32 or 64 bits.");
            if (channels is < 1 or > 8 || rate is < 8_000 or > 192_000)
                throw Rejected(WavProblem.Unsupported, $"WAV audio with {channels} channels at {rate} Hz is not supported. Use 1 to 8 channels at 8,000 to 192,000 Hz.");
            if (blockAlign != channels * bits / 8)
                throw Rejected(WavProblem.Damaged, "The WAV format has an invalid block size.");
            return new Format(encoding, channels, rate, bits);
        }
    }

    private static float[] ReadData(Stream wav, Format format, uint declaredBytes, TimeSpan longest)
    {
        var maxFrames = (long)Math.Ceiling(longest.TotalSeconds * format.SampleRate);
        var lengthIsKnown = declaredBytes != UnknownLengthMarker;
        var declaredFrames = declaredBytes / format.BytesPerFrame;
        if (lengthIsKnown && declaredBytes % format.BytesPerFrame != 0)
            throw Rejected(WavProblem.Damaged, "The WAV audio ends inside a sample frame.");
        if (lengthIsKnown && declaredFrames > maxFrames)
            throw TooLong(longest);
        var resampler = new ResamplerTo16Khz(format.SampleRate, lengthIsKnown ? declaredFrames : null);
        var bytes = new byte[format.BytesPerFrame * FramesPerBlock];
        var mono = new float[FramesPerBlock];
        var bytesLeft = lengthIsKnown ? declaredFrames * format.BytesPerFrame : long.MaxValue;
        long frames = 0;
        var carry = 0;
        while (bytesLeft > 0)
        {
            var read = wav.Read(bytes, carry, (int)Math.Min(bytes.Length - carry, bytesLeft));
            if (read == 0)
                break;
            bytesLeft -= read;
            var filled = carry + read;
            var whole = filled / format.BytesPerFrame;
            frames += whole;
            if (frames > maxFrames)
                throw TooLong(longest);
            ToMono(bytes.AsSpan(0, whole * format.BytesPerFrame), format, mono);
            resampler.Push(mono.AsSpan(0, whole));
            carry = filled - whole * format.BytesPerFrame;
            bytes.AsSpan(whole * format.BytesPerFrame, carry).CopyTo(bytes);
        }
        if (lengthIsKnown && bytesLeft > 0)
            throw Rejected(WavProblem.Damaged, $"The WAV file ends after {frames / (double)format.SampleRate:0.0} of {declaredFrames / (double)format.SampleRate:0.0} seconds of audio.");
        if (carry != 0)
            throw Rejected(WavProblem.Damaged, "The WAV audio ends inside a sample frame.");
        return resampler.Finish();
    }

    private static InvalidDataException TooLong(TimeSpan longest) =>
        Rejected(WavProblem.TooLong, $"The audio is longer than {longest.TotalMinutes:0.##} minutes, the most one call transcribes.");

    public static WavProblem ProblemOf(InvalidDataException rejection) =>
        rejection.Data[nameof(WavProblem)] is WavProblem problem ? problem : WavProblem.Damaged;

    private static InvalidDataException Rejected(WavProblem problem, string message) =>
        new(message) { Data = { [nameof(WavProblem)] = problem } };

    private static void ToMono(ReadOnlySpan<byte> data, Format format, Span<float> mono)
    {
        var bytesPerSample = format.BitsPerSample / 8;
        for (int frame = 0, offset = 0; offset < data.Length; frame++, offset += format.BytesPerFrame)
        {
            float sum = 0;
            for (var channel = 0; channel < format.Channels; channel++)
                sum += Sample(data.Slice(offset + channel * bytesPerSample, bytesPerSample), format.Encoding, format.BitsPerSample);
            var average = sum / format.Channels;
            if (!float.IsFinite(average))
                throw Rejected(WavProblem.Damaged, "The WAV file holds samples that are not finite numbers.");
            mono[frame] = average;
        }
    }

    private static float Sample(ReadOnlySpan<byte> s, ushort encoding, int bits) => (encoding, bits) switch
    {
        (Pcm, 8) => (s[0] - 128) / 128f,
        (Pcm, 16) => BinaryPrimitives.ReadInt16LittleEndian(s) / 32768f,
        (Pcm, 24) => ((s[0] << 8 | s[1] << 16 | s[2] << 24) >> 8) / 8388608f,
        (Pcm, 32) => BinaryPrimitives.ReadInt32LittleEndian(s) / 2147483648f,
        (IeeeFloat, 32) => BinaryPrimitives.ReadSingleLittleEndian(s),
        _ => (float)BinaryPrimitives.ReadDoubleLittleEndian(s)
    };

    private static byte[] ReadExactly(Stream stream, int count, string what) =>
        TryReadExactly(stream, count) ?? throw Rejected(WavProblem.Damaged, $"The WAV file ends inside {what}.");

    private static byte[]? TryReadExactly(Stream stream, int count)
    {
        var buffer = new byte[count];
        return stream.ReadAtLeast(buffer, count, throwOnEndOfStream: false) == count ? buffer : null;
    }

    private static void Skip(Stream stream, long count)
    {
        if (count == 0)
            return;
        if (stream.CanSeek)
        {
            if (stream.Position + count > stream.Length)
                throw Rejected(WavProblem.Damaged, "The WAV file ends inside a chunk.");
            stream.Seek(count, SeekOrigin.Current);
            return;
        }
        var scratch = new byte[(int)Math.Min(81920, count)];
        while (count > 0)
        {
            var read = stream.Read(scratch, 0, (int)Math.Min(scratch.Length, count));
            if (read == 0)
                throw Rejected(WavProblem.Damaged, "The WAV file ends inside a chunk.");
            count -= read;
        }
    }
}
