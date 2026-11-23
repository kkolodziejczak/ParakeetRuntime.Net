using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using static ParakeetRuntime.D3D12Native;

namespace ParakeetRuntime;

internal sealed unsafe class GpuBuffer : IDisposable
{
    public nint Resource { get; private set; }
    public ulong GpuAddress { get; }
    public ulong SizeBytes { get; }
    public uint CurrentState { get; set; }

    public GpuBuffer(nint resource, ulong sizeBytes, uint initialState)
    {
        Resource = resource;
        SizeBytes = sizeBytes;
        CurrentState = initialState;
        GpuAddress = GetGPUVirtualAddress(resource);
    }

    public void* Map(bool forRead = false)
    {
        void* ptr = null;
        D3D12_RANGE empty = default;
        var hr = D3D12Native.Map(Resource, 0, forRead ? null : &empty, &ptr);
        if (hr < 0)
            Marshal.ThrowExceptionForHR(hr);
        return ptr;
    }

    public void Unmap(bool wasWritten = true)
    {
        D3D12_RANGE empty = default;
        D3D12Native.Unmap(Resource, 0, wasWritten ? null : &empty);
    }

    public void Dispose()
    {
        if (Resource != 0)
        {
            Release(Resource);
            Resource = 0;
        }
        GC.SuppressFinalize(this);
    }

    ~GpuBuffer()
    {
        if (Resource != 0)
        {
            Release(Resource);
            Resource = 0;
        }
    }
}

internal readonly record struct GpuAdapterInfo(
    string Name,
    long DedicatedVideoMemoryBytes,
    long SharedSystemMemoryBytes,
    long MemoryBudgetBytes,
    bool IsSoftware,
    bool MeetsMinimumMemory);

internal sealed unsafe class D3D12Device : IDisposable
{
    public const long DefaultMinimumMemoryBytes = 900L * 1024 * 1024;
    public const int RootConstantsIndex = 0;
    public const int RootSrvBase = 1;
    public const int RootUavBase = 7;
    private const int RootConstantCount = 16;
    private const int RootSrvCount = 6;
    private const int RootUavCount = 2;
    private const int RootParameterCount = 9;

    private nint adapter;
    private nint adapter3;
    private nint device;
    private nint queue;
    private nint allocator;
    private nint commandList;
    private nint fence;
    private nint fenceEvent;
    private nint rootSignature;
    private nint timestampHeap;
    private ulong fenceValue;
    private bool listOpen;
    private bool disposed;

    public string AdapterName { get; }
    public long DedicatedVideoMemoryBytes { get; }
    public long SharedSystemMemoryBytes { get; }
    public ulong TimestampFrequency { get; }
    public bool HasTimestamps => timestampHeap != 0 && TimestampFrequency > 0;
    public nint CommandList => commandList;
    public nint RootSignature => rootSignature;

    private D3D12Device(
        string adapterName, long dedicatedVram, long sharedMem,
        nint adapter, nint adapter3, nint device, nint queue,
        nint allocator, nint commandList, nint fence, nint fenceEvent, nint rootSignature,
        nint timestampHeap, ulong timestampFrequency)
    {
        AdapterName = adapterName;
        DedicatedVideoMemoryBytes = dedicatedVram;
        SharedSystemMemoryBytes = sharedMem;
        TimestampFrequency = timestampFrequency;
        this.adapter = adapter;
        this.adapter3 = adapter3;
        this.device = device;
        this.queue = queue;
        this.allocator = allocator;
        this.commandList = commandList;
        this.fence = fence;
        this.fenceEvent = fenceEvent;
        this.rootSignature = rootSignature;
        this.timestampHeap = timestampHeap;
    }

    public static List<GpuAdapterInfo> EnumerateAdapters(long minimumMemoryBytes = DefaultMinimumMemoryBytes)
    {
        var result = new List<GpuAdapterInfo>();
        if (!OperatingSystem.IsWindows())
            return result;
        var factoryIid = IID_IDXGIFactory1;
        nint factory = 0;
        if (CreateDXGIFactory1(&factoryIid, &factory) < 0 || factory == 0)
            return result;
        nint factory6 = 0;
        var factory6Iid = IID_IDXGIFactory6;
        QueryInterface(factory, &factory6Iid, &factory6);
        try
        {
            for (uint index = 0; ; index++)
            {
                nint candidate = 0;
                int hr;
                if (factory6 != 0)
                {
                    var adapter1Iid = IID_IDXGIAdapter1;
                    hr = EnumAdapterByGpuPreference(factory6, index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, &adapter1Iid, &candidate);
                }
                else
                {
                    hr = EnumAdapters1(factory, index, &candidate);
                }
                if (hr == DXGI_ERROR_NOT_FOUND || hr < 0 || candidate == 0)
                    break;
                try
                {
                    DXGI_ADAPTER_DESC1 desc;
                    if (GetDesc1(candidate, &desc) < 0)
                        continue;
                    bool isSoftware = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
                    long dedicated = (long)desc.DedicatedVideoMemory;
                    long shared = (long)desc.SharedSystemMemory;
                    long budget = Math.Max(dedicated, shared);
                    nint a3 = 0;
                    var a3Iid = IID_IDXGIAdapter3;
                    if (QueryInterface(candidate, &a3Iid, &a3) >= 0 && a3 != 0)
                    {
                        try
                        {
                            DXGI_QUERY_VIDEO_MEMORY_INFO memInfo;
                            if (QueryVideoMemoryInfo(a3, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memInfo) >= 0 && memInfo.Budget > 0)
                                budget = (long)memInfo.Budget;
                        }
                        finally
                        {
                            SafeRelease(ref a3);
                        }
                    }
                    result.Add(new GpuAdapterInfo(
                        desc.GetDescription().Trim(), dedicated, shared, budget,
                        isSoftware, !isSoftware && budget >= minimumMemoryBytes));
                }
                finally
                {
                    Release(candidate);
                }
            }
        }
        finally
        {
            SafeRelease(ref factory6);
            SafeRelease(ref factory);
        }
        return result;
    }

    public static D3D12Device? TryCreate(long minimumMemoryBytes = DefaultMinimumMemoryBytes, string? adapterNameFilter = null)
    {
        if (!OperatingSystem.IsWindows())
            return null;
        try
        {
            return CreateCore(minimumMemoryBytes, adapterNameFilter);
        }
        catch
        {
            return null;
        }
    }

    private static D3D12Device? CreateCore(long minimumMemoryBytes, string? adapterNameFilter)
    {
        var factoryIid = IID_IDXGIFactory1;
        nint factory = 0;
        if (CreateDXGIFactory1(&factoryIid, &factory) < 0 || factory == 0)
            return null;

        nint factory6 = 0;
        var factory6Iid = IID_IDXGIFactory6;
        QueryInterface(factory, &factory6Iid, &factory6);

        try
        {
            for (uint index = 0; ; index++)
            {
                nint candidate = 0;
                int hr;
                if (factory6 != 0)
                {
                    var adapter1Iid = IID_IDXGIAdapter1;
                    hr = EnumAdapterByGpuPreference(factory6, index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, &adapter1Iid, &candidate);
                }
                else
                {
                    hr = EnumAdapters1(factory, index, &candidate);
                }
                if (hr == DXGI_ERROR_NOT_FOUND || hr < 0 || candidate == 0)
                    break;

                var created = TryCreateOnAdapter(candidate, minimumMemoryBytes, adapterNameFilter);
                if (created is not null)
                    return created;
                Release(candidate);
            }
            return null;
        }
        finally
        {
            SafeRelease(ref factory6);
            SafeRelease(ref factory);
        }
    }

    private static D3D12Device? TryCreateOnAdapter(nint candidate, long minimumMemoryBytes, string? adapterNameFilter)
    {
        DXGI_ADAPTER_DESC1 desc;
        if (GetDesc1(candidate, &desc) < 0)
            return null;
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0)
            return null;
        string adapterName = desc.GetDescription().Trim();
        if (!string.IsNullOrWhiteSpace(adapterNameFilter) &&
            !adapterName.Contains(adapterNameFilter, StringComparison.OrdinalIgnoreCase))
            return null;

        long dedicated = (long)desc.DedicatedVideoMemory;
        long shared = (long)desc.SharedSystemMemory;
        long availableEstimate = Math.Max(dedicated, shared);

        nint adapter3 = 0;
        var adapter3Iid = IID_IDXGIAdapter3;
        if (QueryInterface(candidate, &adapter3Iid, &adapter3) >= 0 && adapter3 != 0)
        {
            DXGI_QUERY_VIDEO_MEMORY_INFO memInfo;
            if (QueryVideoMemoryInfo(adapter3, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memInfo) >= 0 && memInfo.Budget > 0)
                availableEstimate = (long)memInfo.Budget;
        }

        if (availableEstimate < minimumMemoryBytes)
        {
            SafeRelease(ref adapter3);
            return null;
        }

        nint dev = 0, q = 0, alloc = 0, list = 0, f = 0, rs = 0, tsHeap = 0;
        nint ev = 0;
        try
        {
            var devIid = IID_ID3D12Device;
            if (D3D12CreateDevice(candidate, D3D_FEATURE_LEVEL_11_0, &devIid, &dev) < 0 || dev == 0)
            {
                SafeRelease(ref adapter3);
                return null;
            }

            var qDesc = new D3D12_COMMAND_QUEUE_DESC { Type = D3D12_COMMAND_LIST_TYPE_DIRECT };
            var qIid = IID_ID3D12CommandQueue;
            CheckHr(CreateCommandQueue(dev, &qDesc, &qIid, &q));

            var allocIid = IID_ID3D12CommandAllocator;
            CheckHr(CreateCommandAllocator(dev, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocIid, &alloc));

            var listIid = IID_ID3D12GraphicsCommandList;
            CheckHr(CreateCommandList(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, 0, &listIid, &list));
            CheckHr(ListClose(list));

            var fenceIid = IID_ID3D12Fence;
            CheckHr(CreateFence(dev, 0, 0, &fenceIid, &f));

            ev = CreateEventW(0, 0, 0, 0);
            if (ev == 0)
                throw new InvalidOperationException("CreateEventW failed.");

            rs = CreateUniversalRootSignature(dev);

            ulong tsFreq = 0;
            var qhDesc = new D3D12_QUERY_HEAP_DESC { Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count = 4, NodeMask = 0 };
            var qhIid = IID_ID3D12QueryHeap;
            if (CreateQueryHeap(dev, &qhDesc, &qhIid, &tsHeap) >= 0 && tsHeap != 0)
            {
                if (GetTimestampFrequency(q, &tsFreq) < 0 || tsFreq == 0)
                {
                    SafeRelease(ref tsHeap);
                    tsFreq = 0;
                }
            }

            var result = new D3D12Device(
                adapterName, dedicated, shared,
                candidate, adapter3, dev, q, alloc, list, f, ev, rs, tsHeap, tsFreq);
            adapter3 = dev = q = alloc = list = f = rs = tsHeap = ev = 0;
            return result;
        }
        catch
        {
            if (ev != 0) CloseHandle(ev);
            SafeRelease(ref tsHeap);
            SafeRelease(ref rs);
            SafeRelease(ref f);
            SafeRelease(ref list);
            SafeRelease(ref alloc);
            SafeRelease(ref q);
            SafeRelease(ref dev);
            SafeRelease(ref adapter3);
            return null;
        }
    }

    private static nint CreateUniversalRootSignature(nint dev)
    {
        var parameters = stackalloc D3D12_ROOT_PARAMETER[RootParameterCount];
        parameters[RootConstantsIndex] = new D3D12_ROOT_PARAMETER
        {
            ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            ShaderRegister = 0,
            RegisterSpace = 0,
            Num32BitValues = RootConstantCount,
            ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
        };
        for (uint i = 0; i < RootSrvCount; i++)
        {
            parameters[RootSrvBase + i] = new D3D12_ROOT_PARAMETER
            {
                ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV,
                ShaderRegister = i,
                RegisterSpace = 0,
                ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
            };
        }
        for (uint i = 0; i < RootUavCount; i++)
        {
            parameters[RootUavBase + i] = new D3D12_ROOT_PARAMETER
            {
                ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV,
                ShaderRegister = i,
                RegisterSpace = 0,
                ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
            };
        }

        var desc = new D3D12_ROOT_SIGNATURE_DESC
        {
            NumParameters = RootParameterCount,
            pParameters = parameters,
            NumStaticSamplers = 0,
            pStaticSamplers = 0,
            Flags = 0,
        };

        nint blob = 0, errBlob = 0;
        try
        {
            CheckHr(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, &errBlob));
            var rsIid = IID_ID3D12RootSignature;
            nint rs = 0;
            CheckHr(CreateRootSignature(dev, 0, GetBufferPointer(blob), GetBufferSize(blob), &rsIid, &rs));
            return rs;
        }
        finally
        {
            SafeRelease(ref errBlob);
            SafeRelease(ref blob);
        }
    }

    public GpuBuffer CreateBuffer(ulong sizeBytes, uint heapType, uint initialState, uint flags = D3D12_RESOURCE_FLAG_NONE)
    {
        ulong aligned = Math.Max(256UL, (sizeBytes + 255UL) & ~255UL);
        var heap = new D3D12_HEAP_PROPERTIES { Type = heapType };
        var desc = new D3D12_RESOURCE_DESC
        {
            Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment = 0,
            Width = aligned,
            Height = 1,
            DepthOrArraySize = 1,
            MipLevels = 1,
            Format = 0,
            SampleDesc = new DXGI_SAMPLE_DESC { Count = 1, Quality = 0 },
            Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags = flags,
        };
        var resIid = IID_ID3D12Resource;
        nint res = 0;
        CheckHr(CreateCommittedResource(device, &heap, 0, &desc, initialState, 0, &resIid, &res));
        return new GpuBuffer(res, aligned, initialState);
    }

    public GpuBuffer CreateDefaultBuffer(ulong sizeBytes, bool allowUav = true) =>
        CreateBuffer(sizeBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
            allowUav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE);

    public GpuBuffer CreateUploadBuffer(ulong sizeBytes) =>
        CreateBuffer(sizeBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);

    public GpuBuffer CreateReadbackBuffer(ulong sizeBytes) =>
        CreateBuffer(sizeBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    public nint CompileEmbeddedPipeline(string resourceSuffix, string entryPoint)
    {
        var asm = Assembly.GetExecutingAssembly();
        var name = asm.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith(resourceSuffix, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidOperationException($"Embedded HLSL shader resource ending with '{resourceSuffix}' was not found.");
        using var stream = asm.GetManifestResourceStream(name)!;
        using var reader = new StreamReader(stream, Encoding.UTF8);
        return CompilePipeline(reader.ReadToEnd(), resourceSuffix, entryPoint);
    }

    public nint CompilePipeline(string hlslSource, string sourceName, string entryPoint)
    {
        var srcBytes = Encoding.UTF8.GetBytes(hlslSource);
        var nameBytes = Encoding.ASCII.GetBytes(sourceName + '\0');
        var entryBytes = Encoding.ASCII.GetBytes(entryPoint + '\0');
        var targetBytes = Encoding.ASCII.GetBytes("cs_5_1\0");

        nint codeBlob = 0, errBlob = 0;
        try
        {
            fixed (byte* pSrc = srcBytes, pName = nameBytes, pEntry = entryBytes, pTarget = targetBytes)
            {
                int hr = D3DCompile(
                    pSrc, (nuint)srcBytes.Length, pName, 0, 0, pEntry, pTarget,
                    D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &codeBlob, &errBlob);
                if (hr < 0)
                {
                    var msg = errBlob != 0
                        ? Marshal.PtrToStringAnsi(GetBufferPointer(errBlob), (int)GetBufferSize(errBlob))
                        : $"HRESULT 0x{hr:X8}";
                    throw new InvalidOperationException($"HLSL compile failed for {sourceName}:{entryPoint}: {msg}");
                }
            }

            var psoDesc = new D3D12_COMPUTE_PIPELINE_STATE_DESC
            {
                pRootSignature = rootSignature,
                CS = new D3D12_SHADER_BYTECODE
                {
                    pShaderBytecode = GetBufferPointer(codeBlob),
                    BytecodeLength = GetBufferSize(codeBlob),
                },
            };
            var psoIid = IID_ID3D12PipelineState;
            nint pso = 0;
            CheckHr(CreateComputePipelineState(device, &psoDesc, &psoIid, &pso));
            return pso;
        }
        finally
        {
            SafeRelease(ref errBlob);
            SafeRelease(ref codeBlob);
        }
    }

    public void BeginCommands()
    {
        if (listOpen)
            throw new InvalidOperationException("Command list is already open.");
        CheckHr(AllocatorReset(allocator));
        CheckHr(ListReset(commandList, allocator, 0));
        SetComputeRootSignature(commandList, rootSignature);
        listOpen = true;
    }

    public void Transition(GpuBuffer buffer, uint newState)
    {
        if (buffer.CurrentState == newState)
            return;
        var barrier = D3D12_RESOURCE_BARRIER.Transition(buffer.Resource, buffer.CurrentState, newState);
        ResourceBarrier(commandList, 1, &barrier);
        buffer.CurrentState = newState;
    }

    public void UavBarrier(GpuBuffer? buffer = null)
    {
        var barrier = D3D12_RESOURCE_BARRIER.Uav(buffer?.Resource ?? 0);
        ResourceBarrier(commandList, 1, &barrier);
    }

    public void Barriers(ReadOnlySpan<D3D12_RESOURCE_BARRIER> barriers)
    {
        if (barriers.IsEmpty)
            return;
        fixed (D3D12_RESOURCE_BARRIER* p = barriers)
            ResourceBarrier(commandList, (uint)barriers.Length, p);
    }

    public static bool TryTransition(GpuBuffer buffer, uint newState, out D3D12_RESOURCE_BARRIER barrier)
    {
        if (buffer.CurrentState == newState)
        {
            barrier = default;
            return false;
        }
        barrier = D3D12_RESOURCE_BARRIER.Transition(buffer.Resource, buffer.CurrentState, newState);
        buffer.CurrentState = newState;
        return true;
    }

    public static D3D12_RESOURCE_BARRIER MakeUavBarrierOrTransition(GpuBuffer buffer)
    {
        if (buffer.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            return D3D12_RESOURCE_BARRIER.Uav(buffer.Resource);
        var barrier = D3D12_RESOURCE_BARRIER.Transition(buffer.Resource, buffer.CurrentState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        buffer.CurrentState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        return barrier;
    }

    public void WriteTimestamp(uint index)
    {
        if (timestampHeap != 0)
            EndQuery(commandList, timestampHeap, D3D12_QUERY_TYPE_TIMESTAMP, index);
    }

    public void ResolveTimestamps(uint startIndex, uint count, GpuBuffer readbackDest, ulong destByteOffset = 0)
    {
        if (timestampHeap != 0)
            ResolveQueryData(commandList, timestampHeap, D3D12_QUERY_TYPE_TIMESTAMP, startIndex, count, readbackDest.Resource, destByteOffset);
    }

    public void SetRootConstants(ReadOnlySpan<uint> values)
    {
        fixed (uint* p = values)
            SetComputeRoot32BitConstants(commandList, RootConstantsIndex, (uint)values.Length, p, 0);
    }

    public void SetSrv(int slot, GpuBuffer buffer, ulong byteOffset = 0) =>
        SetComputeRootShaderResourceView(commandList, (uint)(RootSrvBase + slot), buffer.GpuAddress + byteOffset);

    public void SetUav(int slot, GpuBuffer buffer, ulong byteOffset = 0) =>
        SetComputeRootUnorderedAccessView(commandList, (uint)(RootUavBase + slot), buffer.GpuAddress + byteOffset);

    public void ExecuteAndWait()
    {
        if (!listOpen)
            return;
        listOpen = false;
        CheckHr(ListClose(commandList));
        nint list = commandList;
        ExecuteCommandLists(queue, 1, &list);
        ulong target = ++fenceValue;
        CheckHr(Signal(queue, fence, target));
        ulong completed = GetCompletedValue(fence);
        if (completed < target)
        {
            CheckHr(SetEventOnCompletion(fence, target, fenceEvent));
            uint wait = WaitForSingleObject(fenceEvent, INFINITE);
            if (wait != 0)
                throw new InvalidOperationException($"GPU fence wait failed ({wait}).");
            completed = GetCompletedValue(fence);
        }
        ThrowIfDeviceRemoved(completed);
    }

    private void ThrowIfDeviceRemoved(ulong completedFenceValue)
    {
        int removed = GetDeviceRemovedReason(device);
        if (removed < 0)
            Marshal.ThrowExceptionForHR(removed);
        if (completedFenceValue == ulong.MaxValue)
            throw new InvalidOperationException("GPU device was removed during execution.");
    }

    public long QueryVideoMemoryUsageBytes()
    {
        if (adapter3 == 0)
            return 0;
        DXGI_QUERY_VIDEO_MEMORY_INFO info;
        return QueryVideoMemoryInfo(adapter3, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info) >= 0
            ? (long)info.CurrentUsage
            : 0;
    }

    public static ulong Align256(ulong bytes) => (bytes + 255UL) & ~255UL;

    private static void CheckHr(int hr)
    {
        if (hr < 0)
            Marshal.ThrowExceptionForHR(hr);
    }

    public void Dispose()
    {
        DisposeCore();
        GC.SuppressFinalize(this);
    }

    ~D3D12Device() => DisposeCore();

    private void DisposeCore()
    {
        if (disposed)
            return;
        disposed = true;
        if (fenceEvent != 0)
        {
            CloseHandle(fenceEvent);
            fenceEvent = 0;
        }
        SafeRelease(ref timestampHeap);
        SafeRelease(ref rootSignature);
        SafeRelease(ref fence);
        SafeRelease(ref commandList);
        SafeRelease(ref allocator);
        SafeRelease(ref queue);
        SafeRelease(ref device);
        SafeRelease(ref adapter3);
        SafeRelease(ref adapter);
    }
}
