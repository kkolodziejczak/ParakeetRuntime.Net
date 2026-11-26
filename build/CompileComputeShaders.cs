using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;

public sealed class CompileComputeShaders : Microsoft.Build.Utilities.Task
{
    private const uint OptimizationLevel3 = 1 << 15;
    private const string Target = "cs_5_1";

    [Required] public ITaskItem[] Sources { get; set; } = Array.Empty<ITaskItem>();

    [Required] public string OutputDirectory { get; set; } = "";

    public override bool Execute()
    {
        Directory.CreateDirectory(OutputDirectory);
        var entries = Sources.Select(source => source.GetMetadata("FullPath"))
            .SelectMany(path => EntryPoints(path).Select(entry => (Path: path, Entry: entry))).ToList();
        if (entries.Count == 0)
        {
            Log.LogError("No [numthreads] entry points were found in the shader sources.");
            return false;
        }
        var failures = new ConcurrentBag<string>();
        System.Threading.Tasks.Parallel.ForEach(entries, shader =>
        {
            try
            {
                var fileName = System.IO.Path.GetFileName(shader.Path);
                var bytecode = Compile(File.ReadAllText(shader.Path), fileName, shader.Entry);
                File.WriteAllBytes(System.IO.Path.Combine(OutputDirectory, $"{fileName}.{shader.Entry}.dxbc"), bytecode);
            }
            catch (Exception failure)
            {
                failures.Add(failure.Message);
            }
        });
        foreach (var failure in failures)
            Log.LogError(failure);
        return failures.IsEmpty;
    }

    private static IEnumerable<string> EntryPoints(string path) =>
        Regex.Matches(File.ReadAllText(path), @"\[numthreads\([^)]*\)\]\s*void\s+(\w+)").Cast<Match>().Select(match => match.Groups[1].Value);

    private static byte[] Compile(string source, string sourceName, string entry)
    {
        var sourceBytes = Encoding.UTF8.GetBytes(source);
        var code = IntPtr.Zero;
        var errors = IntPtr.Zero;
        try
        {
            int hr = D3DCompile(sourceBytes, (UIntPtr)sourceBytes.Length, sourceName, IntPtr.Zero, IntPtr.Zero, entry, Target, OptimizationLevel3, 0, out code, out errors);
            if (hr < 0)
                throw new InvalidOperationException($"HLSL compile failed for {sourceName}:{entry}: {(errors != IntPtr.Zero ? Marshal.PtrToStringAnsi(BufferPointer(errors), (int)BufferSize(errors)) : $"HRESULT 0x{hr:X8}")}");
            var bytes = new byte[(int)BufferSize(code)];
            Marshal.Copy(BufferPointer(code), bytes, 0, bytes.Length);
            return bytes;
        }
        finally
        {
            Release(errors);
            Release(code);
        }
    }

    [DllImport("d3dcompiler_47.dll", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
    private static extern int D3DCompile(byte[] source, UIntPtr sourceSize, string sourceName, IntPtr defines, IntPtr include,
        string entryPoint, string target, uint flags1, uint flags2, out IntPtr code, out IntPtr errors);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate IntPtr GetBufferPointerMethod(IntPtr blob);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate UIntPtr GetBufferSizeMethod(IntPtr blob);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate uint ReleaseMethod(IntPtr blob);

    private static T Method<T>(IntPtr blob, int slot) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(Marshal.ReadIntPtr(Marshal.ReadIntPtr(blob), slot * IntPtr.Size));

    private static IntPtr BufferPointer(IntPtr blob) => Method<GetBufferPointerMethod>(blob, 3)(blob);

    private static ulong BufferSize(IntPtr blob) => Method<GetBufferSizeMethod>(blob, 4)(blob).ToUInt64();

    private static void Release(IntPtr blob)
    {
        if (blob != IntPtr.Zero)
            Method<ReleaseMethod>(blob, 2)(blob);
    }
}
