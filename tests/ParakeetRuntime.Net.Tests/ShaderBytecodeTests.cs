using ParakeetRuntime.Net;

namespace ParakeetRuntime.Net.Tests;

public sealed class ShaderBytecodeTests
{
    public static TheoryData<string, string> PipelinesTheEngineCreates()
    {
        var pipelines = new TheoryData<string, string>();
        foreach (var (file, entry) in ParakeetGpu.Pipelines.Append(ParakeetGpu.LargeGemmPipeline))
            pipelines.Add(file, entry);
        return pipelines;
    }

    [Theory]
    [MemberData(nameof(PipelinesTheEngineCreates))]
    public void Every_pipeline_the_engine_creates_is_embedded_as_the_bytecode_its_current_source_compiles_to(string file, string entryPoint)
    {
        using var embedded = typeof(D3D12Device).Assembly.GetManifestResourceStream(D3D12Device.EmbeddedBytecodeName(file, entryPoint));
        Assert.NotNull(embedded);
        var bytes = new byte[embedded.Length];
        embedded.ReadExactly(bytes);

        Assert.Equal(D3D12Device.CompileToBytecode(D3D12Device.EmbeddedSource(file), file, entryPoint), bytes);
    }
}
