// Part of ParakeetRuntime.Net; a C# port of NeMo-Speech.cpp (Apache-2.0), see THIRD-PARTY-NOTICES.md.
namespace ParakeetRuntime.Net;

internal sealed class EncoderScratch
{
    [ThreadStatic] static EncoderScratch? current;
    readonly List<float[]> buffers = new();
    readonly List<int> lengthRentedThisUtterance = new();
    readonly Dictionary<int, Q8Activations> quantizedByColumns = new();
    int cursor;

    public Utterance BeginUtterance()
    {
        if (current != null) throw new InvalidOperationException();
        current = this;
        cursor = 0;
        for (int slot = 0; slot < lengthRentedThisUtterance.Count; slot++)
            lengthRentedThisUtterance[slot] = NotRentedYet;
        return new Utterance();
    }

    public readonly struct Utterance : IDisposable
    {
        public void Dispose() => current = null;
    }

    public void Rewind() => cursor = 0;

    public void Release()
    {
        buffers.Clear();
        lengthRentedThisUtterance.Clear();
        quantizedByColumns.Clear();
        cursor = 0;
    }

    const int NotRentedYet = -1;

    public static Q8Activations Quant(int rows, int columns)
    {
        if (current is not { } scope) return new Q8Activations(rows, columns);
        if (!scope.quantizedByColumns.TryGetValue(columns, out var m) || m.CapacityRows < rows)
            scope.quantizedByColumns[columns] = m = new Q8Activations(rows, columns);
        m.UseRows(rows);
        return m;
    }

    public static float[] Rent(int length)
    {
        if (current is not { } scope) return new float[length];
        int slot = scope.cursor++;
        if (slot == scope.buffers.Count)
        {
            scope.buffers.Add(new float[length]);
            scope.lengthRentedThisUtterance.Add(length);
            return scope.buffers[slot];
        }
        int rentedBefore = scope.lengthRentedThisUtterance[slot];
        if (rentedBefore == NotRentedYet)
        {
            if (scope.buffers[slot].Length < length)
                scope.buffers[slot] = new float[length];
            else
                Array.Clear(scope.buffers[slot], 0, length);
            scope.lengthRentedThisUtterance[slot] = length;
        }
        else if (rentedBefore != length)
            throw new InvalidOperationException();
        return scope.buffers[slot];
    }
}
