// The Parakeet engine is a C# port of the parts of NeMo-Speech.cpp (Apache-2.0) that LocalBuddy
// uses; see THIRD-PARTY-NOTICES.md. Written for LocalBuddy, starting from the council prototype in
// %USERPROFILE%\.councils\local-parakeet-engine.
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace ParakeetRuntime;

// A GGUF v3 model file: its metadata and its F32, F16 and Q8_0 tensors. Everything about the file
// is checked here, where it enters, so the model code can trust shapes and offsets. Tensors are
// read one at a time rather than mapped, so the file's pages never count towards the process's
// memory on top of the weights copied out of it.
internal sealed class Gguf : IDisposable
{
    private const uint Magic = 0x46554747; // "GGUF"
    private const int F32 = 0, F16 = 1, Q8_0 = 8;
    private const int Q8BlockBytes = 34; // fp16 scale + 32 int8 values

    private sealed record Tensor(string Name, long[] Dims, int Type, long Offset)
    {
        // Each dimension is below int.MaxValue, so the product only overflows past four of them;
        // checked anyway so a crafted shape fails here instead of wrapping.
        public long Count => Dims.Aggregate(1L, (n, d) => checked(n * d));
    }

    private readonly Dictionary<string, object> metadata = [];
    private readonly Dictionary<string, Tensor> tensors = [];
    private readonly SafeFileHandle file;
    private readonly long dataStart;
    private byte[] buffer = [];

    public Gguf(string path)
    {
        long length;
        using (var stream = File.OpenRead(path))
        using (var reader = new BinaryReader(stream, Encoding.UTF8))
        {
            length = stream.Length;
            try
            {
                ReadHeader(reader);
            }
            catch (EndOfStreamException)
            {
                throw new InvalidDataException($"The model file {path} ends in the middle of its header.");
            }
            var alignment = metadata.ContainsKey("general.alignment") ? Int("general.alignment") : 32;
            if (alignment is < 1 or > 1 << 20)
                throw new InvalidDataException($"The model file has an alignment of {alignment}.");
            dataStart = (stream.Position + alignment - 1) / alignment * alignment;
        }
        try
        {
            foreach (var tensor in tensors.Values)
                CheckFits(tensor, length);
        }
        catch (OverflowException)
        {
            throw new InvalidDataException("The model file has a tensor too large to address.");
        }
        file = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.RandomAccess);
    }

    private void ReadHeader(BinaryReader reader)
    {
        if (reader.ReadUInt32() != Magic)
            throw new InvalidDataException("The file is not a GGUF model file.");
        if (reader.ReadUInt32() is var version and not 3)
            throw new InvalidDataException($"GGUF version {version} is not supported; expected 3.");
        var tensorCount = Count(reader, "tensor count");
        var keyCount = Count(reader, "metadata count");
        for (long i = 0; i < keyCount; i++)
        {
            var key = ReadString(reader);
            metadata[key] = ReadValue(reader, reader.ReadInt32(), insideArray: false);
        }
        for (long i = 0; i < tensorCount; i++)
        {
            var name = ReadString(reader);
            var dimensions = reader.ReadInt32();
            if (dimensions is < 1 or > 4)
                throw new InvalidDataException($"Tensor {name} has {dimensions} dimensions.");
            var dims = new long[dimensions];
            for (var d = 0; d < dimensions; d++)
                dims[d] = reader.ReadInt64() is var size and > 0 and < int.MaxValue ? size : throw new InvalidDataException($"Tensor {name} has a bad shape.");
            tensors[name] = new Tensor(name, dims, reader.ReadInt32(), reader.ReadInt64());
        }
    }

    private static long Count(BinaryReader reader, string what) =>
        reader.ReadInt64() is var count and >= 0 and <= 1_000_000 ? count : throw new InvalidDataException($"The model file has a bad {what}.");

    private static string ReadString(BinaryReader reader)
    {
        var length = reader.ReadInt64();
        if (length < 0 || length > reader.BaseStream.Length - reader.BaseStream.Position)
            throw new InvalidDataException("The model file has a string longer than the file.");
        return Encoding.UTF8.GetString(reader.ReadBytes((int)length));
    }

    private static object ReadValue(BinaryReader reader, int type, bool insideArray) => type switch
    {
        0 => reader.ReadByte(),
        1 => reader.ReadSByte(),
        2 => reader.ReadUInt16(),
        3 => reader.ReadInt16(),
        4 => reader.ReadUInt32(),
        5 => reader.ReadInt32(),
        6 => reader.ReadSingle(),
        7 => reader.ReadBoolean(),
        8 => ReadString(reader),
        9 when insideArray => throw new InvalidDataException("The model file nests arrays inside arrays."),
        9 => ReadArray(reader),
        10 => reader.ReadUInt64(),
        11 => reader.ReadInt64(),
        12 => reader.ReadDouble(),
        _ => throw new InvalidDataException($"GGUF value type {type} is not supported.")
    };

    private static object[] ReadArray(BinaryReader reader)
    {
        var type = reader.ReadInt32();
        var count = reader.ReadInt64();
        if (count < 0 || count > reader.BaseStream.Length - reader.BaseStream.Position)
            throw new InvalidDataException("The model file has an array longer than the file.");
        var values = new object[count];
        for (long i = 0; i < count; i++)
            values[i] = ReadValue(reader, type, insideArray: true);
        return values;
    }

    private void CheckFits(Tensor tensor, long fileLength)
    {
        var bytes = tensor.Type switch
        {
            F32 => checked(tensor.Count * 4),
            F16 => checked(tensor.Count * 2),
            Q8_0 when tensor.Dims[0] % 32 == 0 => checked(tensor.Count / 32 * Q8BlockBytes),
            Q8_0 => throw new InvalidDataException($"Q8_0 tensor {tensor.Name} has rows that are not a multiple of 32."),
            _ => 0 // Not used by LocalBuddy; reading it fails below.
        };
        if (tensor.Offset < 0 || tensor.Offset > fileLength - dataStart || bytes > fileLength - dataStart - tensor.Offset)
            throw new InvalidDataException($"Tensor {tensor.Name} runs past the end of the model file; the file is damaged or cut off.");
    }

    public int Int(string key) => Metadata(key, Convert.ToInt32);

    public float Float(string key) => Metadata(key, Convert.ToSingle);

    public string[] Strings(string key) => Metadata(key, value => ((object[])value).Cast<string>().ToArray());

    public int[] Ints(string key) => Metadata(key, value => ((object[])value).Select(Convert.ToInt32).ToArray());

    private T Metadata<T>(string key, Func<object, T> convert)
    {
        if (!metadata.TryGetValue(key, out var value))
            throw new InvalidDataException($"The model file has no {key}.");
        try
        {
            return convert(value);
        }
        catch (Exception exception) when (exception is InvalidCastException or FormatException or OverflowException)
        {
            throw new InvalidDataException($"The model file's {key} has the wrong type.", exception);
        }
    }

    private Tensor Find(string name) =>
        tensors.TryGetValue(name, out var tensor) ? tensor : throw new InvalidDataException($"The model file has no tensor {name}.");

    // Fills the span with the file's bytes from offset on; CheckFits made sure they exist.
    private void Read(Span<byte> destination, long offset)
    {
        while (destination.Length > 0)
        {
            var read = RandomAccess.Read(file, destination, offset);
            if (read == 0)
                throw new InvalidDataException("The model file ended early; it is damaged or was changed while loading.");
            destination = destination[read..];
            offset += read;
        }
    }

    // A tensor's bytes, in a buffer reused for the next tensor.
    private ReadOnlySpan<byte> Bytes(Tensor tensor, long length)
    {
        if (buffer.Length < length)
            buffer = new byte[checked((int)length)];
        var bytes = buffer.AsSpan(0, (int)length);
        Read(bytes, dataStart + tensor.Offset);
        return bytes;
    }

    // Any F32, F16 or Q8_0 tensor as floats, row-major with the first dimension fastest.
    public float[] Floats(string name)
    {
        var tensor = Find(name);
        var values = new float[tensor.Count];
        switch (tensor.Type)
        {
            case F32:
                Read(MemoryMarshal.AsBytes(values.AsSpan()), dataStart + tensor.Offset);
                break;
            case F16:
                var halves = MemoryMarshal.Cast<byte, Half>(Bytes(tensor, values.Length * 2L));
                for (var i = 0; i < values.Length; i++)
                    values[i] = (float)halves[i];
                break;
            case Q8_0:
                var q = Q8(name);
                for (var i = 0; i < values.Length; i++)
                    values[i] = q.Q[i] * q.D[i / 32];
                break;
            default:
                throw new InvalidDataException($"Tensor {name} has type {tensor.Type}; expected F32, F16 or Q8_0.");
        }
        return values;
    }

    // Q8_0 matrices of [rows][K] with the same K, stacked in the order given, unpacked into int8
    // values and one float scale per 32 values.
    public Q8Matrix Q8(params string[] names)
    {
        var parts = names.Select(Find).ToArray();
        foreach (var tensor in parts)
        {
            if (tensor.Type != Q8_0)
                throw new InvalidDataException($"Tensor {tensor.Name} has type {tensor.Type}; expected Q8_0.");
            if (tensor.Dims[0] != parts[0].Dims[0])
                throw new InvalidDataException($"Tensor {tensor.Name} has rows of {tensor.Dims[0]}, not {parts[0].Dims[0]} like {parts[0].Name}.");
        }
        var k = (int)parts[0].Dims[0];
        var matrix = new Q8Matrix(checked((int)parts.Sum(tensor => tensor.Count / k)), k);
        long block = 0;
        foreach (var tensor in parts)
        {
            var blocks = tensor.Count / 32;
            var data = Bytes(tensor, blocks * Q8BlockBytes);
            for (long b = 0; b < blocks; b++, block++)
            {
                var blockData = data.Slice((int)(b * Q8BlockBytes), Q8BlockBytes);
                matrix.D[block] = (float)MemoryMarshal.Read<Half>(blockData);
                MemoryMarshal.Cast<byte, sbyte>(blockData[2..]).CopyTo(matrix.Q.AsSpan((int)(block * 32), 32));
            }
        }
        return matrix;
    }

    public (int Rows, int K) Q8Shape(string name)
    {
        var tensor = Find(name);
        if (tensor.Type != Q8_0)
            throw new InvalidDataException($"Tensor {name} has type {tensor.Type}; expected Q8_0.");
        var k = (int)tensor.Dims[0];
        return (checked((int)(tensor.Count / k)), k);
    }

    // Streams raw Q8_0 int8 weights and fp16 scale bits via RandomAccess into GPU staging buffers
    // without allocating per-tensor managed weight arrays.
    public unsafe (int Rows, int K) CopyQ8Raw(string name, byte* qDest, ushort* dDest)
    {
        var tensor = Find(name);
        if (tensor.Type != Q8_0)
            throw new InvalidDataException($"Tensor {name} has type {tensor.Type}; expected Q8_0.");
        var k = (int)tensor.Dims[0];
        var rows = checked((int)(tensor.Count / k));
        long blocks = (long)rows * (k / 32);
        var data = Bytes(tensor, checked(blocks * Q8BlockBytes));
        fixed (byte* pData = data)
        {
            for (long block = 0; block < blocks; block++)
            {
                var blockStart = pData + block * Q8BlockBytes;
                dDest[block] = *(ushort*)blockStart;
                Buffer.MemoryCopy(blockStart + 2, qDest + block * 32, 32, 32);
            }
        }
        return (rows, k);
    }

    public void Dispose() => file.Dispose();
}

// Int8 weights in 32-value blocks, each block with its own scale, as the file stores them.
internal sealed class Q8Matrix(int rows, int k)
{
    public int Rows { get; } = rows;
    public int K { get; } = k;
    public int Blocks { get; } = k / 32;
    public sbyte[] Q { get; } = new sbyte[checked(rows * k)];
    public float[] D { get; } = new float[checked(rows * (k / 32))];
}
