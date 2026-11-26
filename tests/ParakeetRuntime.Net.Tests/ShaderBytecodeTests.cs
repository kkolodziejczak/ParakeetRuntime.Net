using System.Text.RegularExpressions;
using ParakeetRuntime.Net;

namespace ParakeetRuntime.Net.Tests;

public sealed class ShaderBytecodeTests
{
    private static readonly string[] ShaderFiles = ["DecoderOps.hlsl", "EncoderOps.hlsl", "MelOps.hlsl", "Q8Gemm.hlsl", "SubsampleOps.hlsl"];

    public static TheoryData<string, string> EntryPoints()
    {
        var entries = new TheoryData<string, string>();
        foreach (var file in ShaderFiles)
            foreach (Match entry in Regex.Matches(D3D12Device.EmbeddedSource(file), @"\[numthreads\([^)]*\)\]\s*void\s+(\w+)"))
                entries.Add(file, entry.Groups[1].Value);
        return entries;
    }

    [Fact]
    public void The_shaders_have_the_seventeen_entry_points_the_engine_uses() =>
        Assert.Equal(17, EntryPoints().Count);

    [Theory]
    [MemberData(nameof(EntryPoints))]
    public void Each_entry_point_is_embedded_as_the_bytecode_its_current_source_compiles_to(string file, string entryPoint)
    {
        using var embedded = typeof(D3D12Device).Assembly.GetManifestResourceStream(D3D12Device.EmbeddedBytecodeName(file, entryPoint));
        Assert.NotNull(embedded);
        var bytes = new byte[embedded.Length];
        embedded.ReadExactly(bytes);

        Assert.Equal(D3D12Device.CompileToBytecode(D3D12Device.EmbeddedSource(file), file, entryPoint), bytes);
    }
}
