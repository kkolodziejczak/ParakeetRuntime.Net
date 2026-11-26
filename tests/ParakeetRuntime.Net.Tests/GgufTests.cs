using System.Text;
using ParakeetRuntime.Net;

namespace ParakeetRuntime.Net.Tests;

public sealed class GgufTests : IDisposable
{
    private readonly string path = Path.Combine(Path.GetTempPath(), $"gguf-test-{Guid.NewGuid():N}.gguf");

    public void Dispose() => File.Delete(path);

    [Fact]
    public void Metadata_and_tensors_are_read_back()
    {
        var q8Values = Enumerable.Range(-16, 32).Select(v => (sbyte)v).ToArray();
        File.WriteAllBytes(path, new GgufBuilder()
            .Int("asr.encoder.n_layers", 24)
            .Float("asr.preprocessor.preemph", 0.97f)
            .Strings("asr.tokenizer.vocab", ["▁a", "b"])
            .F32("bias", [1.5f, -2f])
            .Q8("weight", rows: 1, scale: 0.5f, q8Values)
            .Build());

        using var file = new Gguf(path);

        Assert.Equal(24, file.Int("asr.encoder.n_layers"));
        Assert.Equal(0.97f, file.Float("asr.preprocessor.preemph"));
        Assert.Equal(["▁a", "b"], file.Strings("asr.tokenizer.vocab"));
        Assert.Equal([1.5f, -2f], file.Floats("bias"));
        var weight = file.Q8("weight");
        Assert.Equal((1, 32), (weight.Rows, weight.K));
        Assert.Equal(q8Values, weight.Q);
        Assert.Equal([0.5f], weight.D);
    }

    [Fact]
    public void Q8_tensors_with_the_same_width_are_read_as_one_matrix_of_stacked_rows()
    {
        var first = Enumerable.Range(-16, 32).Select(v => (sbyte)v).ToArray();
        var second = Enumerable.Range(0, 64).Select(v => (sbyte)(v - 32)).ToArray();
        File.WriteAllBytes(path, new GgufBuilder()
            .Q8("q", rows: 1, scale: 0.5f, first)
            .Q8("k", rows: 2, scale: 0.25f, second)
            .Q8("wide", rows: 1, scale: 1f, second)
            .Build());

        using var file = new Gguf(path);
        var stacked = file.Q8("q", "k");

        Assert.Equal((3, 32), (stacked.Rows, stacked.K));
        Assert.Equal([.. first, .. second], stacked.Q);
        Assert.Equal([0.5f, 0.25f, 0.25f], stacked.D);
        var error = Assert.Throws<InvalidDataException>(() => file.Q8("q", "wide"));
        Assert.Contains("wide", error.Message);
    }

    [Fact]
    public void A_file_that_is_not_GGUF_is_rejected()
    {
        File.WriteAllBytes(path, Encoding.ASCII.GetBytes("this is not a model file at all"));

        Assert.Throws<InvalidDataException>(() => new Gguf(path));
    }

    [Fact]
    public void A_cut_off_header_is_rejected()
    {
        var whole = new GgufBuilder().Int("a", 1).F32("t", [1f]).Build();
        File.WriteAllBytes(path, whole[..20]);

        Assert.Throws<InvalidDataException>(() => new Gguf(path));
    }

    [Fact]
    public void A_tensor_that_runs_past_the_end_of_the_file_is_rejected()
    {
        var whole = new GgufBuilder().F32("t", new float[64]).Build();
        File.WriteAllBytes(path, whole[..^16]);

        Assert.Throws<InvalidDataException>(() => new Gguf(path));
    }

    [Fact]
    public void A_missing_tensor_is_named_in_the_error()
    {
        File.WriteAllBytes(path, new GgufBuilder().F32("t", [1f]).Build());
        using var file = new Gguf(path);

        var error = Assert.Throws<InvalidDataException>(() => file.Floats("encoder.missing"));

        Assert.Contains("encoder.missing", error.Message);
    }

    [Fact]
    public void A_tensor_offset_so_large_that_it_wraps_around_is_rejected()
    {
        File.WriteAllBytes(path, new GgufBuilder().F32("t", [1f], offset: long.MaxValue - 3).Build());

        Assert.Throws<InvalidDataException>(() => new Gguf(path));
    }

    [Fact]
    public void A_tensor_shape_whose_size_overflows_is_rejected()
    {
        File.WriteAllBytes(path, new GgufBuilder().F32("t", [1f], dims: [int.MaxValue - 1, int.MaxValue - 1, int.MaxValue - 1]).Build());

        Assert.Throws<InvalidDataException>(() => new Gguf(path));
    }

    [Fact]
    public void A_tensor_whose_value_count_fits_a_long_but_whose_byte_size_would_wrap_to_zero_is_rejected()
    {
        File.WriteAllBytes(path, new GgufBuilder().F32("t", [1f], dims: [1L << 30, 1L << 30, 4]).Build());

        Assert.Throws<InvalidDataException>(() => new Gguf(path));
    }

    [Fact]
    public void An_alignment_of_zero_is_rejected()
    {
        File.WriteAllBytes(path, new GgufBuilder().Int("general.alignment", 0).F32("t", [1f]).Build());

        Assert.Throws<InvalidDataException>(() => new Gguf(path));
    }

    [Fact]
    public void Arrays_nested_in_arrays_are_rejected()
    {
        File.WriteAllBytes(path, new GgufBuilder().NestedArrays("deep", 5000).Build());

        Assert.Throws<InvalidDataException>(() => new Gguf(path));
    }

    [Fact]
    public void A_value_of_the_wrong_type_is_named_in_the_error()
    {
        File.WriteAllBytes(path, new GgufBuilder().Strings("asr.encoder.n_layers", ["24"]).Build());
        using var file = new Gguf(path);

        var error = Assert.Throws<InvalidDataException>(() => file.Int("asr.encoder.n_layers"));

        Assert.Contains("asr.encoder.n_layers", error.Message);
    }

    private sealed class GgufBuilder
    {
        private readonly List<Action<BinaryWriter>> keyValues = [];
        private readonly List<(string Name, long[] Dims, int Type, byte[] Data, long? Offset)> tensors = [];

        public GgufBuilder Int(string key, int value) => Add(key, 5, w => w.Write(value));

        public GgufBuilder Float(string key, float value) => Add(key, 6, w => w.Write(value));

        public GgufBuilder Strings(string key, string[] values) => Add(key, 9, w =>
        {
            w.Write(8);
            w.Write((long)values.Length);
            foreach (var value in values)
                WriteString(w, value);
        });

        public GgufBuilder F32(string name, float[] values, long? offset = null, long[]? dims = null)
        {
            tensors.Add((name, dims ?? [values.Length], 0, values.SelectMany(BitConverter.GetBytes).ToArray(), offset));
            return this;
        }

        public GgufBuilder NestedArrays(string key, int depth) => Add(key, 9, w =>
        {
            for (var i = 0; i < depth; i++)
            {
                w.Write(9);
                w.Write(1L);
            }
            w.Write(5);
            w.Write(0L);
        });

        public GgufBuilder Q8(string name, int rows, float scale, sbyte[] values)
        {
            var data = new List<byte>();
            foreach (var block in values.Chunk(32))
            {
                data.AddRange(BitConverter.GetBytes((Half)scale));
                data.AddRange(block.Select(v => (byte)v));
            }
            tensors.Add((name, [values.Length / rows, rows], 8, [.. data], null));
            return this;
        }

        private GgufBuilder Add(string key, int type, Action<BinaryWriter> value)
        {
            keyValues.Add(w =>
            {
                WriteString(w, key);
                w.Write(type);
                value(w);
            });
            return this;
        }

        public byte[] Build()
        {
            using var stream = new MemoryStream();
            using var w = new BinaryWriter(stream);
            w.Write(0x46554747u);
            w.Write(3u);
            w.Write((long)tensors.Count);
            w.Write((long)keyValues.Count);
            foreach (var keyValue in keyValues)
                keyValue(w);
            long offset = 0;
            foreach (var (name, dims, type, data, fixedOffset) in tensors)
            {
                WriteString(w, name);
                w.Write(dims.Length);
                foreach (var dim in dims)
                    w.Write(dim);
                w.Write(type);
                w.Write(fixedOffset ?? offset);
                offset += Align(data.Length);
            }
            w.Write(new byte[Align(stream.Length) - stream.Length]);
            foreach (var (_, _, _, data, _) in tensors)
            {
                w.Write(data);
                w.Write(new byte[Align(data.Length) - data.Length]);
            }
            return stream.ToArray();
        }

        private static long Align(long length) => (length + 31) / 32 * 32;

        private static void WriteString(BinaryWriter w, string value)
        {
            var bytes = Encoding.UTF8.GetBytes(value);
            w.Write((long)bytes.Length);
            w.Write(bytes);
        }
    }
}
