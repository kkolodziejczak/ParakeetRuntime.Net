using System.Buffers.Binary;
using ParakeetRuntime;

namespace ParakeetRuntime.Tests;

public sealed class WavAudioTests
{
    private const ushort Pcm = 1;
    private const ushort IeeeFloat = 3;
    private static readonly TimeSpan TenMinutes = TimeSpan.FromMinutes(10);

    [Fact]
    public void Sixteen_kilohertz_mono_PCM16_comes_back_sample_for_sample()
    {
        short[] samples = [0, 16384, -16384, 32767, -32768];

        var audio = Read(Wav(Pcm, 16, 1, 16_000, PcmBytes16(samples)));

        Assert.Equal([0f, 0.5f, -0.5f, 32767 / 32768f, -1f], audio);
    }

    [Theory]
    [InlineData(Pcm, 8)]
    [InlineData(Pcm, 16)]
    [InlineData(Pcm, 24)]
    [InlineData(Pcm, 32)]
    [InlineData(IeeeFloat, 32)]
    [InlineData(IeeeFloat, 64)]
    public void Every_supported_sample_encoding_reads_as_the_same_level(ushort encoding, int bits)
    {
        var audio = Read(Wav(encoding, bits, 1, 16_000, Constant(encoding, bits, frames: 100, level: 0.5)));

        Assert.All(audio, sample => Assert.Equal(0.5f, sample, 0.01f));
    }

    [Fact]
    public void An_extensible_format_reads_like_the_plain_one()
    {
        var data = Constant(Pcm, 16, frames: 100, level: 0.25);

        var audio = Read(Wav(Pcm, 16, 1, 16_000, data, extensible: true));

        Assert.Equal(Read(Wav(Pcm, 16, 1, 16_000, data)), audio);
    }

    [Fact]
    public void Channels_are_averaged_into_mono()
    {
        short[] leftRight = [16384, 0, 16384, -16384];

        var audio = Read(Wav(Pcm, 16, 2, 16_000, PcmBytes16(leftRight)));

        Assert.Equal([0.25f, 0f], audio);
    }

    [Theory]
    [InlineData(48_000)]
    [InlineData(44_100)]
    [InlineData(8_000)]
    [InlineData(44_101)]
    [InlineData(192_000)]
    public void Other_rates_become_the_same_tone_at_16_kHz(int rate)
    {
        const double hertz = 440;
        var seconds = 2;
        var tone = Enumerable.Range(0, rate * seconds).Select(n => (float)(0.5 * Math.Sin(2 * Math.PI * hertz * n / rate))).ToArray();

        var audio = Read(Wav(IeeeFloat, 32, 1, rate, FloatBytes(tone)));

        Assert.Equal((long)tone.Length * 16_000 / rate, audio.Length);
        var worst = Enumerable.Range(1_000, audio.Length - 2_000).Max(n => Math.Abs(audio[n] - 0.5 * Math.Sin(2 * Math.PI * hertz * n / 16_000)));
        Assert.True(worst < 0.01, $"Largest difference from the expected tone: {worst}");
    }

    [Fact]
    public void Reading_in_tiny_pieces_gives_exactly_the_same_audio()
    {
        var rate = 44_100;
        var noise = new Random(7);
        var input = Enumerable.Range(0, rate).Select(_ => (float)(noise.NextDouble() - 0.5)).ToArray();
        var wav = Wav(IeeeFloat, 32, 1, rate, FloatBytes(input));

        var whole = Read(wav);
        var trickled = WavAudio.Read16KhzMono(new TrickleStream(wav, new Random(3)), TenMinutes);

        Assert.Equal(whole, trickled);
    }

    [Fact]
    public void Chunks_before_the_audio_are_skipped_on_a_stream_that_cannot_seek()
    {
        var plain = Wav(Pcm, 16, 1, 16_000, PcmBytes16([100, 200]));
        byte[] withJunk = [.. plain[..12], .. "JUNK"u8, 3, 0, 0, 0, 1, 2, 3, 0, .. plain[12..]];

        var audio = WavAudio.Read16KhzMono(new ForwardOnlyStream(withJunk), TenMinutes);

        Assert.Equal(Read(plain), audio);
    }

    [Fact]
    public void A_data_chunk_of_unknown_length_is_read_to_the_end()
    {
        var wav = Wav(Pcm, 16, 1, 16_000, PcmBytes16([100, 200, 300]));
        BinaryPrimitives.WriteUInt32LittleEndian(wav.AsSpan(40), 0xFFFFFFFF);

        Assert.Equal(3, WavAudio.Read16KhzMono(new ForwardOnlyStream(wav), TenMinutes).Length);
    }

    [Fact]
    public void A_data_chunk_of_unknown_length_that_ends_inside_a_frame_is_rejected()
    {
        var wav = Wav(Pcm, 16, 1, 16_000, PcmBytes16([100, 200, 300]));
        BinaryPrimitives.WriteUInt32LittleEndian(wav.AsSpan(40), 0xFFFFFFFF);

        Rejected(wav[..^1], "inside a sample");
    }

    [Theory]
    [InlineData("not a WAV file")]
    [InlineData("empty")]
    public void Something_that_is_not_a_WAV_file_is_rejected_as_such(string what)
    {
        byte[] bytes = what == "empty" ? [] : [0xFF, 0xFB, 0x90, 0x00, .. new byte[100]];

        Rejected(bytes, "not a WAV file");
    }

    [Fact]
    public void A_WAV_file_without_audio_data_is_rejected() =>
        Rejected(Wav(Pcm, 16, 1, 16_000, PcmBytes16([1, 2]))[..36], "no audio data");

    [Fact]
    public void Audio_data_before_its_format_is_rejected()
    {
        byte[] wav = [.. "RIFF"u8, 0, 0, 0, 0, .. "WAVE"u8, .. "data"u8, 2, 0, 0, 0, 1, 0];

        Rejected(wav, "before its format");
    }

    [Theory]
    [InlineData(2, 16, 1, 16_000, "not supported")]
    [InlineData(Pcm, 12, 1, 16_000, "not supported")]
    [InlineData(Pcm, 0, 1, 16_000, "not supported")]
    [InlineData(IeeeFloat, 16, 1, 16_000, "not supported")]
    [InlineData(Pcm, 16, 9, 16_000, "not supported")]
    [InlineData(Pcm, 16, 0, 16_000, "not supported")]
    [InlineData(Pcm, 16, 1, 0, "not supported")]
    [InlineData(Pcm, 16, 1, 7_999, "not supported")]
    [InlineData(Pcm, 16, 1, 192_001, "not supported")]
    public void Formats_outside_the_supported_range_are_rejected(int encoding, int bits, int channels, int rate, string message) =>
        Rejected(Wav((ushort)encoding, bits, channels, rate, new byte[64]), message);

    [Fact]
    public void A_format_whose_block_size_does_not_match_is_rejected()
    {
        var wav = Wav(Pcm, 16, 1, 16_000, new byte[4]);
        BinaryPrimitives.WriteUInt16LittleEndian(wav.AsSpan(32), 3);

        Rejected(wav, "block size");
    }

    [Fact]
    public void An_oversized_format_chunk_is_rejected()
    {
        var wav = Wav(Pcm, 16, 1, 16_000, new byte[4]);
        BinaryPrimitives.WriteUInt32LittleEndian(wav.AsSpan(16), 1_000_000);

        Rejected(wav, "format chunk");
    }

    [Fact]
    public void A_skipped_chunk_that_runs_past_the_end_is_rejected()
    {
        var plain = Wav(Pcm, 16, 1, 16_000, PcmBytes16([1]));
        byte[] wav = [.. plain[..12], .. "LIST"u8, 0xFF, 0xFF, 0xFF, 0xFF, 1, 2, 3];

        Rejected(wav, "ends inside a chunk");
        Rejected(new ForwardOnlyStream(wav), "ends inside a chunk");
    }

    [Fact]
    public void A_file_cut_off_inside_its_audio_is_rejected_instead_of_returning_part_of_it()
    {
        var wav = Wav(Pcm, 16, 1, 16_000, PcmBytes16(new short[16_000]));

        Rejected(wav[..(wav.Length / 2)], "ends after");
    }

    [Fact]
    public void Samples_that_are_not_finite_numbers_are_rejected() =>
        Rejected(Wav(IeeeFloat, 32, 1, 16_000, FloatBytes([0f, float.NaN])), "not finite");

    [Fact]
    public void Audio_declared_longer_than_the_limit_is_rejected_before_it_is_read()
    {
        var wav = Wav(Pcm, 16, 1, 16_000, PcmBytes16(new short[16_001]));

        var error = Assert.Throws<InvalidDataException>(() => WavAudio.Read16KhzMono(new MemoryStream(wav), TimeSpan.FromSeconds(1)));
        Assert.Contains("longer than", error.Message);
    }

    [Fact]
    public void Audio_of_unknown_length_that_runs_past_the_limit_is_rejected_while_reading()
    {
        var wav = Wav(Pcm, 16, 1, 16_000, PcmBytes16(new short[16_001]));
        BinaryPrimitives.WriteUInt32LittleEndian(wav.AsSpan(40), 0xFFFFFFFF);

        var error = Assert.Throws<InvalidDataException>(() => WavAudio.Read16KhzMono(new ForwardOnlyStream(wav), TimeSpan.FromSeconds(1)));
        Assert.Contains("longer than", error.Message);
    }

    [Fact]
    public void Ten_minutes_at_192_kHz_stereo_converts_without_holding_the_source_audio_in_memory()
    {
        var wav = new GeneratedToneWav(rate: 192_000, channels: 2, seconds: 600);
        var before = GC.GetAllocatedBytesForCurrentThread();

        var audio = WavAudio.Read16KhzMono(wav, TenMinutes);

        var allocatedMegabytes = (GC.GetAllocatedBytesForCurrentThread() - before) / 1_048_576.0;
        Assert.Equal(600 * 16_000, audio.Length);
        Assert.True(allocatedMegabytes < 80, $"Allocated {allocatedMegabytes:0} MB; the 16 kHz result alone is 37 MB, the 192 kHz source would be 880 MB.");
    }

    private static float[] Read(byte[] wav) => WavAudio.Read16KhzMono(new MemoryStream(wav), TenMinutes);

    private static void Rejected(byte[] wav, string messagePart) => Rejected(new MemoryStream(wav), messagePart);

    private static void Rejected(Stream wav, string messagePart)
    {
        var error = Assert.Throws<InvalidDataException>(() => WavAudio.Read16KhzMono(wav, TenMinutes));
        Assert.Contains(messagePart, error.Message);
    }

    private static byte[] Wav(ushort encoding, int bits, int channels, int rate, byte[] data, bool extensible = false)
    {
        var header = WavHeader(encoding, bits, channels, rate, (uint)data.Length, extensible);
        return [.. header, .. data];
    }

    internal static byte[] WavHeader(ushort encoding, int bits, int channels, int rate, uint dataBytes, bool extensible = false)
    {
        var fmtSize = extensible ? 40 : 16;
        var header = new byte[12 + 8 + fmtSize + 8];
        var s = header.AsSpan();
        "RIFF"u8.CopyTo(s);
        BinaryPrimitives.WriteUInt32LittleEndian(s[4..], unchecked((uint)(header.Length - 8) + dataBytes));
        "WAVE"u8.CopyTo(s[8..]);
        "fmt "u8.CopyTo(s[12..]);
        BinaryPrimitives.WriteInt32LittleEndian(s[16..], fmtSize);
        BinaryPrimitives.WriteUInt16LittleEndian(s[20..], extensible ? (ushort)0xFFFE : encoding);
        BinaryPrimitives.WriteUInt16LittleEndian(s[22..], (ushort)channels);
        BinaryPrimitives.WriteInt32LittleEndian(s[24..], rate);
        BinaryPrimitives.WriteInt32LittleEndian(s[28..], rate * channels * bits / 8);
        BinaryPrimitives.WriteUInt16LittleEndian(s[32..], (ushort)(channels * bits / 8));
        BinaryPrimitives.WriteUInt16LittleEndian(s[34..], (ushort)bits);
        if (extensible)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(s[36..], 22);
            BinaryPrimitives.WriteUInt16LittleEndian(s[38..], (ushort)bits);
            BinaryPrimitives.WriteUInt16LittleEndian(s[44..], encoding);
            Convert.FromHexString("000000001000800000AA00389B71").CopyTo(s[46..]);
        }
        "data"u8.CopyTo(s[(20 + fmtSize)..]);
        BinaryPrimitives.WriteUInt32LittleEndian(s[(24 + fmtSize)..], dataBytes);
        return header;
    }

    private static byte[] PcmBytes16(short[] samples)
    {
        var bytes = new byte[samples.Length * 2];
        for (var i = 0; i < samples.Length; i++)
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(i * 2), samples[i]);
        return bytes;
    }

    private static byte[] FloatBytes(float[] samples)
    {
        var bytes = new byte[samples.Length * 4];
        for (var i = 0; i < samples.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), samples[i]);
        return bytes;
    }

    private static byte[] Constant(ushort encoding, int bits, int frames, double level)
    {
        var size = bits / 8;
        var bytes = new byte[frames * size];
        for (var i = 0; i < frames; i++)
        {
            var s = bytes.AsSpan(i * size, size);
            switch ((encoding, bits))
            {
                case (Pcm, 8): s[0] = (byte)(128 + level * 128); break;
                case (Pcm, 16): BinaryPrimitives.WriteInt16LittleEndian(s, (short)(level * 32768)); break;
                case (Pcm, 24): var v = (int)(level * 8388608); s[0] = (byte)v; s[1] = (byte)(v >> 8); s[2] = (byte)(v >> 16); break;
                case (Pcm, 32): BinaryPrimitives.WriteInt32LittleEndian(s, (int)(level * 2147483648.0)); break;
                case (IeeeFloat, 32): BinaryPrimitives.WriteSingleLittleEndian(s, (float)level); break;
                default: BinaryPrimitives.WriteDoubleLittleEndian(s, level); break;
            }
        }
        return bytes;
    }

    private sealed class ForwardOnlyStream(byte[] bytes) : MemoryStream(bytes)
    {
        public override bool CanSeek => false;
        public override long Seek(long offset, SeekOrigin loc) => throw new NotSupportedException();
        public override long Position { get => base.Position; set => throw new NotSupportedException(); }
        public override long Length => throw new NotSupportedException();
    }

    private sealed class TrickleStream(byte[] bytes, Random sizes) : MemoryStream(bytes)
    {
        public override int Read(byte[] buffer, int offset, int count) => base.Read(buffer, offset, Math.Min(count, sizes.Next(1, 7)));
        public override int Read(Span<byte> buffer) => base.Read(buffer[..Math.Min(buffer.Length, sizes.Next(1, 7))]);
    }

    private sealed class GeneratedToneWav : Stream
    {
        private readonly byte[] header;
        private readonly long length;
        private readonly int rate;
        private readonly int channels;
        private long position;

        public GeneratedToneWav(int rate, int channels, int seconds)
        {
            this.rate = rate;
            this.channels = channels;
            var dataBytes = (uint)((long)rate * seconds * channels * 4);
            header = WavHeader(IeeeFloat, 32, channels, rate, dataBytes);
            length = header.Length + dataBytes;
        }

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

        public override int Read(Span<byte> buffer)
        {
            var written = 0;
            while (written < buffer.Length && position < length)
            {
                if (position < header.Length)
                {
                    buffer[written++] = header[position++];
                    continue;
                }
                var dataOffset = position - header.Length;
                if (dataOffset % 4 != 0 || buffer.Length - written < 4)
                    break;
                var frame = dataOffset / 4 / channels;
                BinaryPrimitives.WriteSingleLittleEndian(buffer[written..], (float)(0.25 * Math.Sin(2 * Math.PI * 440 * frame / rate)));
                written += 4;
                position += 4;
            }
            return written;
        }

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => position; set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
