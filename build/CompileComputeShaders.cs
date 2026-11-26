using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;

public sealed class CompileComputeShaders : Microsoft.Build.Utilities.Task
{
    private const uint OptimizationLevel3 = 1 << 15;
    private const string Target = "cs_5_1";
    private const string FingerprintFile = "sources.sha256";

    [Required] public ITaskItem[] Sources { get; set; } = Array.Empty<ITaskItem>();

    [Required] public string OutputDirectory { get; set; } = "";

    [Required] public string TaskSource { get; set; } = "";

    [Output] public ITaskItem[] Bytecode { get; set; } = Array.Empty<ITaskItem>();

    public override bool Execute()
    {
        var sources = Sources.Select(source => source.GetMetadata("FullPath")).OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .Select(path => (Path: path, Text: File.ReadAllText(path))).ToList();
        var shaders = sources.SelectMany(source => EntryPoints(source.Text).Select(entry => (source.Path, source.Text, Entry: entry))).ToList();
        if (shaders.Count == 0)
        {
            Log.LogError("No [numthreads] entry points were found in the shader sources.");
            return false;
        }
        var outputs = shaders.Select(shader => System.IO.Path.Combine(OutputDirectory, $"{System.IO.Path.GetFileName(shader.Path)}.{shader.Entry}.dxbc")).ToList();
        var fingerprint = Fingerprint(sources);
        var fingerprintPath = System.IO.Path.Combine(OutputDirectory, FingerprintFile);
        if (!File.Exists(fingerprintPath) || File.ReadAllText(fingerprintPath) != fingerprint || !outputs.All(File.Exists))
        {
            if (Directory.Exists(OutputDirectory))
                Directory.Delete(OutputDirectory, recursive: true);
            Directory.CreateDirectory(OutputDirectory);
            var failures = new ConcurrentBag<string>();
            System.Threading.Tasks.Parallel.For(0, shaders.Count, i =>
            {
                try
                {
                    File.WriteAllBytes(outputs[i], Compile(shaders[i].Text, System.IO.Path.GetFileName(shaders[i].Path), shaders[i].Entry));
                }
                catch (Exception failure)
                {
                    failures.Add(failure.Message);
                }
            });
            foreach (var failure in failures)
                Log.LogError(failure);
            if (!failures.IsEmpty)
                return false;
            File.WriteAllText(fingerprintPath, fingerprint);
        }
        Bytecode = outputs.Select(output => (ITaskItem)new TaskItem(output)).ToArray();
        return true;
    }

    private string Fingerprint(IEnumerable<(string Path, string Text)> sources)
    {
        var all = new StringBuilder(File.ReadAllText(TaskSource));
        foreach (var source in sources)
            all.Append('\0').Append(System.IO.Path.GetFileName(source.Path)).Append('\0').Append(source.Text);
        using (var sha = SHA256.Create())
            return BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(all.ToString()))).Replace("-", "");
    }

    private static IEnumerable<string> EntryPoints(string source) =>
        Regex.Matches(Regex.Replace(source, @"//[^\n]*|/\*.*?\*/", " ", RegexOptions.Singleline), @"\[\s*numthreads\s*\([^)]*\)\s*\]\s*void\s+(\w+)")
            .Cast<Match>().Select(match => match.Groups[1].Value);

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
