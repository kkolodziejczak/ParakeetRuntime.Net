// Part of ParakeetRuntime.Net; a C# port of NeMo-Speech.cpp (Apache-2.0), see THIRD-PARTY-NOTICES.md.
namespace ParakeetRuntime.Net;

internal sealed class EncoderScratch : IDisposable
{
    [ThreadStatic] static EncoderScratch? current;
    readonly List<float[]> buffers = new();
    readonly Dictionary<(int,int),Q8Activations> quantized = new();
    int cursor;
    public EncoderScratch() { if(current!=null) throw new InvalidOperationException(); current=this; }
    public void Rewind() { cursor=0; }
    public void Dispose() { current=null; }
    public static Q8Activations Quant(int rows,int columns)
    {
        if(current is not {} scope) return new Q8Activations(rows,columns);
        if(!scope.quantized.TryGetValue((rows,columns), out var m))
            scope.quantized[(rows,columns)]=m=new Q8Activations(rows,columns);
        return m;
    }
    public static float[] Rent(int length)
    {
        if(current is not {} scope) return new float[length];
        int slot=scope.cursor++;
        if(slot==scope.buffers.Count) scope.buffers.Add(new float[length]);
        var result=scope.buffers[slot];
        if(result.Length!=length) throw new InvalidOperationException();
        return result;
    }
}
