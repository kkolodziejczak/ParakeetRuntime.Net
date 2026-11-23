using System.Runtime.InteropServices;

namespace ParakeetRuntime;

// Minimal Direct3D 12, DXGI, and D3DCompiler P/Invoke and COM vtable bindings.
// Uses only inbox Windows DLLs (d3d12.dll, dxgi.dll, d3dcompiler_47.dll, kernel32.dll).
internal static unsafe class D3D12Native
{
    public static readonly Guid IID_IDXGIFactory1 = new("770aae78-f26f-4dba-a829-253c83d1b387");
    public static readonly Guid IID_IDXGIFactory6 = new("c1b6694f-ff09-44a9-b03c-77900a0a1d17");
    public static readonly Guid IID_IDXGIAdapter1 = new("29038f61-3839-4626-91fd-086879011a05");
    public static readonly Guid IID_IDXGIAdapter3 = new("645967A4-1392-4310-A798-8053CE3E93FD");
    public static readonly Guid IID_ID3D12Device = new("189819f1-1db6-4b57-be54-1821339b85f7");
    public static readonly Guid IID_ID3D12CommandQueue = new("0ec870a6-5d7e-4c22-8cfc-5baae07616ed");
    public static readonly Guid IID_ID3D12CommandAllocator = new("6102dee4-af59-4b09-b999-b44d73f09b24");
    public static readonly Guid IID_ID3D12GraphicsCommandList = new("5b160d0f-ac1b-4185-8ba8-b3ae42a5a455");
    public static readonly Guid IID_ID3D12Resource = new("696442be-a72e-4059-bc79-5b5c98040fad");
    public static readonly Guid IID_ID3D12Fence = new("0a753dcf-c4d8-4b91-adf6-be5a60d95a76");
    public static readonly Guid IID_ID3D12RootSignature = new("c54a6b66-72df-4ee8-8be5-a946a1429214");
    public static readonly Guid IID_ID3D12PipelineState = new("765a30f3-f624-4c6f-a828-ace948622445");
    public static readonly Guid IID_ID3D12QueryHeap = new("0d9658ae-ed45-469e-a61d-970ec583cab4");

    public const uint D3D_FEATURE_LEVEL_11_0 = 0xb000;
    public const uint DXGI_ADAPTER_FLAG_SOFTWARE = 2;
    public const uint DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE = 2;
    public const int DXGI_ERROR_NOT_FOUND = unchecked((int)0x887A0002);
    public const uint DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0;

    public const uint D3D12_COMMAND_LIST_TYPE_DIRECT = 0;
    public const uint D3D12_HEAP_TYPE_DEFAULT = 1;
    public const uint D3D12_HEAP_TYPE_UPLOAD = 2;
    public const uint D3D12_HEAP_TYPE_READBACK = 3;
    public const uint D3D12_RESOURCE_DIMENSION_BUFFER = 1;
    public const uint D3D12_TEXTURE_LAYOUT_ROW_MAJOR = 1;
    public const uint D3D12_RESOURCE_FLAG_NONE = 0;
    public const uint D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS = 0x4;

    public const uint D3D12_RESOURCE_STATE_COMMON = 0;
    public const uint D3D12_RESOURCE_STATE_UNORDERED_ACCESS = 0x8;
    public const uint D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE = 0x40;
    public const uint D3D12_RESOURCE_STATE_COPY_DEST = 0x400;
    public const uint D3D12_RESOURCE_STATE_COPY_SOURCE = 0x800;
    public const uint D3D12_RESOURCE_STATE_GENERIC_READ = 0x1 | 0x2 | 0x40 | 0x80 | 0x200 | 0x800;

    public const uint D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS = 1;
    public const uint D3D12_ROOT_PARAMETER_TYPE_SRV = 3;
    public const uint D3D12_ROOT_PARAMETER_TYPE_UAV = 4;
    public const uint D3D12_SHADER_VISIBILITY_ALL = 0;
    public const uint D3D_ROOT_SIGNATURE_VERSION_1_0 = 1;

    public const uint D3D12_RESOURCE_BARRIER_TYPE_TRANSITION = 0;
    public const uint D3D12_RESOURCE_BARRIER_TYPE_UAV = 2;
    public const uint D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES = 0xFFFFFFFF;
    public const uint D3D12_QUERY_HEAP_TYPE_TIMESTAMP = 1;
    public const uint D3D12_QUERY_TYPE_TIMESTAMP = 2;

    public const uint D3DCOMPILE_OPTIMIZATION_LEVEL3 = 1 << 15;
    public const uint INFINITE = 0xFFFFFFFF;

    [DllImport("dxgi.dll")]
    public static extern int CreateDXGIFactory1(Guid* riid, nint* ppFactory);

    [DllImport("d3d12.dll")]
    public static extern int D3D12CreateDevice(nint pAdapter, uint minimumFeatureLevel, Guid* riid, nint* ppDevice);

    [DllImport("d3d12.dll")]
    public static extern int D3D12SerializeRootSignature(
        D3D12_ROOT_SIGNATURE_DESC* pRootSignature, uint version, nint* ppBlob, nint* ppErrorBlob);

    [DllImport("d3dcompiler_47.dll")]
    public static extern int D3DCompile(
        byte* pSrcData, nuint srcDataSize, byte* pSourceName,
        nint pDefines, nint pInclude,
        byte* pEntrypoint, byte* pTarget,
        uint flags1, uint flags2,
        nint* ppCode, nint* ppErrorMsgs);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern nint CreateEventW(nint lpEventAttributes, int bManualReset, int bInitialState, nint lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint WaitForSingleObject(nint hHandle, uint dwMilliseconds);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern int CloseHandle(nint hObject);

    // --- Structs matching Windows SDK 10.0.26100.0 (x64) ---

    [StructLayout(LayoutKind.Sequential)]
    public struct LUID
    {
        public uint LowPart;
        public int HighPart;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DXGI_ADAPTER_DESC1
    {
        public fixed char Description[128];
        public uint VendorId;
        public uint DeviceId;
        public uint SubSysId;
        public uint Revision;
        public nuint DedicatedVideoMemory;
        public nuint DedicatedSystemMemory;
        public nuint SharedSystemMemory;
        public LUID AdapterLuid;
        public uint Flags;

        public string GetDescription()
        {
            fixed (char* p = Description)
                return new string(p);
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DXGI_QUERY_VIDEO_MEMORY_INFO
    {
        public ulong Budget;
        public ulong CurrentUsage;
        public ulong AvailableForReservation;
        public ulong CurrentReservation;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct D3D12_COMMAND_QUEUE_DESC
    {
        public uint Type;
        public int Priority;
        public uint Flags;
        public uint NodeMask;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct D3D12_HEAP_PROPERTIES
    {
        public uint Type;
        public uint CPUPageProperty;
        public uint MemoryPoolPreference;
        public uint CreationNodeMask;
        public uint VisibleNodeMask;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DXGI_SAMPLE_DESC
    {
        public uint Count;
        public uint Quality;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct D3D12_RESOURCE_DESC
    {
        public uint Dimension;
        public ulong Alignment;
        public ulong Width;
        public uint Height;
        public ushort DepthOrArraySize;
        public ushort MipLevels;
        public uint Format;
        public DXGI_SAMPLE_DESC SampleDesc;
        public uint Layout;
        public uint Flags;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct D3D12_RANGE
    {
        public nuint Begin;
        public nuint End;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct D3D12_QUERY_HEAP_DESC
    {
        public uint Type;
        public uint Count;
        public uint NodeMask;
    }

    [StructLayout(LayoutKind.Explicit, Size = 32)]
    public struct D3D12_ROOT_PARAMETER
    {
        [FieldOffset(0)] public uint ParameterType;
        // D3D12_ROOT_CONSTANTS (offset 8)
        [FieldOffset(8)] public uint ShaderRegister;
        [FieldOffset(12)] public uint RegisterSpace;
        [FieldOffset(16)] public uint Num32BitValues;
        // ShaderVisibility (offset 24)
        [FieldOffset(24)] public uint ShaderVisibility;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct D3D12_ROOT_SIGNATURE_DESC
    {
        public uint NumParameters;
        public D3D12_ROOT_PARAMETER* pParameters;
        public uint NumStaticSamplers;
        public nint pStaticSamplers;
        public uint Flags;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct D3D12_SHADER_BYTECODE
    {
        public nint pShaderBytecode;
        public nuint BytecodeLength;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct D3D12_CACHED_PIPELINE_STATE
    {
        public nint pCachedBlob;
        public nuint CachedBlobSizeInBytes;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct D3D12_COMPUTE_PIPELINE_STATE_DESC
    {
        public nint pRootSignature;
        public D3D12_SHADER_BYTECODE CS;
        public uint NodeMask;
        public D3D12_CACHED_PIPELINE_STATE CachedPSO;
        public uint Flags;
    }

    [StructLayout(LayoutKind.Explicit, Size = 32)]
    public struct D3D12_RESOURCE_BARRIER
    {
        [FieldOffset(0)] public uint Type;
        [FieldOffset(4)] public uint Flags;
        // Transition & UAV share pResource at offset 8
        [FieldOffset(8)] public nint pResource;
        [FieldOffset(16)] public uint Subresource;
        [FieldOffset(20)] public uint StateBefore;
        [FieldOffset(24)] public uint StateAfter;

        public static D3D12_RESOURCE_BARRIER Transition(nint resource, uint before, uint after) => new()
        {
            Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags = 0,
            pResource = resource,
            Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            StateBefore = before,
            StateAfter = after,
        };

        public static D3D12_RESOURCE_BARRIER Uav(nint resource = 0) => new()
        {
            Type = D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags = 0,
            pResource = resource,
        };
    }

    // --- COM vtable call helpers ---

    private static void** Vtbl(nint obj) => *(void***)obj;

    public static int QueryInterface(nint obj, Guid* riid, nint* ppv) =>
        ((delegate* unmanaged[Stdcall]<nint, Guid*, nint*, int>)Vtbl(obj)[0])(obj, riid, ppv);

    public static uint Release(nint obj) =>
        obj == 0 ? 0 : ((delegate* unmanaged[Stdcall]<nint, uint>)Vtbl(obj)[2])(obj);

    public static void SafeRelease(ref nint obj)
    {
        if (obj != 0)
        {
            Release(obj);
            obj = 0;
        }
    }

    // IDXGIFactory1 [12] EnumAdapters1
    public static int EnumAdapters1(nint factory, uint index, nint* ppAdapter) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, nint*, int>)Vtbl(factory)[12])(factory, index, ppAdapter);

    // IDXGIFactory6 [29] EnumAdapterByGpuPreference
    public static int EnumAdapterByGpuPreference(nint factory6, uint index, uint preference, Guid* riid, nint* ppAdapter) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, uint, Guid*, nint*, int>)Vtbl(factory6)[29])(factory6, index, preference, riid, ppAdapter);

    // IDXGIAdapter1 [10] GetDesc1
    public static int GetDesc1(nint adapter, DXGI_ADAPTER_DESC1* pDesc) =>
        ((delegate* unmanaged[Stdcall]<nint, DXGI_ADAPTER_DESC1*, int>)Vtbl(adapter)[10])(adapter, pDesc);

    // IDXGIAdapter3 [14] QueryVideoMemoryInfo
    public static int QueryVideoMemoryInfo(nint adapter3, uint nodeIndex, uint memorySegmentGroup, DXGI_QUERY_VIDEO_MEMORY_INFO* pInfo) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, uint, DXGI_QUERY_VIDEO_MEMORY_INFO*, int>)Vtbl(adapter3)[14])(adapter3, nodeIndex, memorySegmentGroup, pInfo);

    // ID3D10Blob [3] GetBufferPointer, [4] GetBufferSize
    public static nint GetBufferPointer(nint blob) =>
        ((delegate* unmanaged[Stdcall]<nint, nint>)Vtbl(blob)[3])(blob);

    public static nuint GetBufferSize(nint blob) =>
        ((delegate* unmanaged[Stdcall]<nint, nuint>)Vtbl(blob)[4])(blob);

    // ID3D12Device
    public static int CreateCommandQueue(nint device, D3D12_COMMAND_QUEUE_DESC* desc, Guid* riid, nint* ppQueue) =>
        ((delegate* unmanaged[Stdcall]<nint, D3D12_COMMAND_QUEUE_DESC*, Guid*, nint*, int>)Vtbl(device)[8])(device, desc, riid, ppQueue);

    public static int CreateCommandAllocator(nint device, uint type, Guid* riid, nint* ppAllocator) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, Guid*, nint*, int>)Vtbl(device)[9])(device, type, riid, ppAllocator);

    public static int CreateComputePipelineState(nint device, D3D12_COMPUTE_PIPELINE_STATE_DESC* desc, Guid* riid, nint* ppPso) =>
        ((delegate* unmanaged[Stdcall]<nint, D3D12_COMPUTE_PIPELINE_STATE_DESC*, Guid*, nint*, int>)Vtbl(device)[11])(device, desc, riid, ppPso);

    public static int CreateCommandList(nint device, uint nodeMask, uint type, nint allocator, nint initialState, Guid* riid, nint* ppList) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, uint, nint, nint, Guid*, nint*, int>)Vtbl(device)[12])(device, nodeMask, type, allocator, initialState, riid, ppList);

    public static int CreateRootSignature(nint device, uint nodeMask, nint blobPtr, nuint blobLen, Guid* riid, nint* ppRootSig) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, nint, nuint, Guid*, nint*, int>)Vtbl(device)[16])(device, nodeMask, blobPtr, blobLen, riid, ppRootSig);

    public static int CreateCommittedResource(
        nint device, D3D12_HEAP_PROPERTIES* heapProps, uint heapFlags, D3D12_RESOURCE_DESC* desc,
        uint initialState, nint clearValue, Guid* riid, nint* ppResource) =>
        ((delegate* unmanaged[Stdcall]<nint, D3D12_HEAP_PROPERTIES*, uint, D3D12_RESOURCE_DESC*, uint, nint, Guid*, nint*, int>)Vtbl(device)[27])(
            device, heapProps, heapFlags, desc, initialState, clearValue, riid, ppResource);

    public static int CreateFence(nint device, ulong initialValue, uint flags, Guid* riid, nint* ppFence) =>
        ((delegate* unmanaged[Stdcall]<nint, ulong, uint, Guid*, nint*, int>)Vtbl(device)[36])(device, initialValue, flags, riid, ppFence);

    public static int GetDeviceRemovedReason(nint device) =>
        ((delegate* unmanaged[Stdcall]<nint, int>)Vtbl(device)[37])(device);

    public static int CreateQueryHeap(nint device, D3D12_QUERY_HEAP_DESC* desc, Guid* riid, nint* ppvHeap) =>
        ((delegate* unmanaged[Stdcall]<nint, D3D12_QUERY_HEAP_DESC*, Guid*, nint*, int>)Vtbl(device)[39])(device, desc, riid, ppvHeap);

    // ID3D12CommandQueue
    public static void ExecuteCommandLists(nint queue, uint numLists, nint* ppLists) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, nint*, void>)Vtbl(queue)[10])(queue, numLists, ppLists);

    public static int Signal(nint queue, nint fence, ulong value) =>
        ((delegate* unmanaged[Stdcall]<nint, nint, ulong, int>)Vtbl(queue)[14])(queue, fence, value);

    public static int GetTimestampFrequency(nint queue, ulong* pFrequency) =>
        ((delegate* unmanaged[Stdcall]<nint, ulong*, int>)Vtbl(queue)[16])(queue, pFrequency);

    // ID3D12CommandAllocator
    public static int AllocatorReset(nint allocator) =>
        ((delegate* unmanaged[Stdcall]<nint, int>)Vtbl(allocator)[8])(allocator);

    // ID3D12GraphicsCommandList
    public static int ListClose(nint list) =>
        ((delegate* unmanaged[Stdcall]<nint, int>)Vtbl(list)[9])(list);

    public static int ListReset(nint list, nint allocator, nint initialState) =>
        ((delegate* unmanaged[Stdcall]<nint, nint, nint, int>)Vtbl(list)[10])(list, allocator, initialState);

    public static void Dispatch(nint list, uint x, uint y, uint z) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, uint, uint, void>)Vtbl(list)[14])(list, x, y, z);

    public static void CopyBufferRegion(nint list, nint dst, ulong dstOffset, nint src, ulong srcOffset, ulong numBytes) =>
        ((delegate* unmanaged[Stdcall]<nint, nint, ulong, nint, ulong, ulong, void>)Vtbl(list)[15])(list, dst, dstOffset, src, srcOffset, numBytes);

    public static void SetPipelineState(nint list, nint pso) =>
        ((delegate* unmanaged[Stdcall]<nint, nint, void>)Vtbl(list)[25])(list, pso);

    public static void ResourceBarrier(nint list, uint numBarriers, D3D12_RESOURCE_BARRIER* barriers) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, D3D12_RESOURCE_BARRIER*, void>)Vtbl(list)[26])(list, numBarriers, barriers);

    public static void SetComputeRootSignature(nint list, nint rootSig) =>
        ((delegate* unmanaged[Stdcall]<nint, nint, void>)Vtbl(list)[29])(list, rootSig);

    public static void SetComputeRoot32BitConstants(nint list, uint rootParameterIndex, uint num32BitValues, void* srcData, uint destOffset) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, uint, void*, uint, void>)Vtbl(list)[35])(list, rootParameterIndex, num32BitValues, srcData, destOffset);

    public static void SetComputeRootShaderResourceView(nint list, uint rootParameterIndex, ulong gpuAddress) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, ulong, void>)Vtbl(list)[39])(list, rootParameterIndex, gpuAddress);

    public static void SetComputeRootUnorderedAccessView(nint list, uint rootParameterIndex, ulong gpuAddress) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, ulong, void>)Vtbl(list)[41])(list, rootParameterIndex, gpuAddress);

    public static void EndQuery(nint list, nint queryHeap, uint type, uint index) =>
        ((delegate* unmanaged[Stdcall]<nint, nint, uint, uint, void>)Vtbl(list)[53])(list, queryHeap, type, index);

    public static void ResolveQueryData(nint list, nint queryHeap, uint type, uint startIndex, uint numQueries, nint destinationBuffer, ulong alignedDestinationBufferOffset) =>
        ((delegate* unmanaged[Stdcall]<nint, nint, uint, uint, uint, nint, ulong, void>)Vtbl(list)[54])(
            list, queryHeap, type, startIndex, numQueries, destinationBuffer, alignedDestinationBufferOffset);

    // ID3D12Resource
    public static int Map(nint resource, uint subresource, D3D12_RANGE* readRange, void** ppData) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, D3D12_RANGE*, void**, int>)Vtbl(resource)[8])(resource, subresource, readRange, ppData);

    public static void Unmap(nint resource, uint subresource, D3D12_RANGE* writtenRange) =>
        ((delegate* unmanaged[Stdcall]<nint, uint, D3D12_RANGE*, void>)Vtbl(resource)[9])(resource, subresource, writtenRange);

    public static ulong GetGPUVirtualAddress(nint resource) =>
        ((delegate* unmanaged[Stdcall]<nint, ulong>)Vtbl(resource)[11])(resource);

    // ID3D12Fence
    public static ulong GetCompletedValue(nint fence) =>
        ((delegate* unmanaged[Stdcall]<nint, ulong>)Vtbl(fence)[8])(fence);

    public static int SetEventOnCompletion(nint fence, ulong value, nint hEvent) =>
        ((delegate* unmanaged[Stdcall]<nint, ulong, nint, int>)Vtbl(fence)[9])(fence, value, hEvent);
}
