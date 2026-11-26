This document explains how the Direct3D 12 path (`ParakeetGpu`, `D3D12Device`, `D3D12Native`, `Shaders/Q8Gemm.hlsl` and `GemmBench`) works and why it is built this way.

# GPU path

## Overview

`ParakeetGpu` runs the whole model on the GPU, from raw 16 kHz PCM samples to decoded token IDs:

1. Log-Mel spectrogram (`RecordLogMelSpectrogram`), into `bufFeats[frames, melBins]`.
2. `dw_striding` subsampling (`RecordSubsampling`), into `curX[T, d]`.
3. The Conformer layers (`RecordConformerLayer`), in place on `curX[T, d]`.
4. The `joint.enc` projection, into `bufEncProj[T, jointDim]`.
5. The TDT greedy decoder, as a loop of GPU dispatches (`RecordDecodeSteps`).

Everything stays in VRAM. A short utterance on a fast GPU is usually one submit.

`TranscribeFromMel` is the entry point for models whose Mel spectrogram the GPU shaders cannot
compute (`CanRunGpuMel` is false: an FFT size other than 512, or more than 256 Mel bins). The CPU
computes the features, and the GPU runs the rest.

## Device and adapter

- `D3D12Device` uses only DLLs that every supported Windows ships: `d3d12.dll`, `dxgi.dll`,
  `d3dcompiler_47.dll` and `kernel32.dll`. There is no Windows SDK projection, so any `net10.0`
  application can reference the library.
- Adapters are listed with `IDXGIFactory6::EnumAdapterByGpuPreference(HIGH_PERFORMANCE)` when it
  exists, so a discrete GPU comes before an integrated one. Older systems fall back to
  `IDXGIFactory1::EnumAdapters1`.
- Software adapters (WARP, the Basic Render Driver) are rejected.
- An adapter must have at least `DefaultMinimumMemoryBytes` of memory. The estimate starts as the
  larger of `DedicatedVideoMemory` and `SharedSystemMemory`, because an integrated GPU has little
  dedicated memory and works from shared system memory. When `IDXGIAdapter3` is available, its
  local-segment `Budget` replaces the estimate.
- The device is created at feature level 11_0 with one direct command queue, one command
  allocator, one command list, one fence and a timestamp query heap. A GPU without timestamp
  support still works; it just has no GPU-side timing.

## Native bindings

- `D3D12Native` calls COM methods through their vtable slot (`Vtbl(obj)[n]`) with unmanaged
  function pointers. The slot numbers and struct layouts follow the Windows SDK 10.0.26100.0
  headers for x64.
- `D3D12_ROOT_PARAMETER` is a union in the SDK. The binding lays out only the members this engine
  uses: `ParameterType` at offset 0, the `D3D12_ROOT_CONSTANTS` / root descriptor fields
  (`ShaderRegister`, `RegisterSpace`, `Num32BitValues`) from offset 8, and `ShaderVisibility` at
  offset 24.
- `D3D12_RESOURCE_BARRIER` is also a union: transition and UAV barriers both keep `pResource` at
  offset 8.

## Root signature

Every compute shader uses the same root signature, so any pipeline can run with any binding:

| Root parameter | Register | Contents |
| --- | --- | --- |
| 0 | `b0` | 16 32-bit root constants |
| 1..6 | `t0`..`t5` | root SRVs |
| 7..8 | `u0`..`u1` | root UAVs |

Buffers bind as root descriptors (a GPU virtual address plus a byte offset), so there are no
descriptor heaps. A shader reads its inputs as `ByteAddressBuffer` and writes as
`RWByteAddressBuffer`.

## Pipelines and compilation

- Shaders are HLSL Shader Model 5.1 (`cs_5_1`), compiled by FXC in the inbox `d3dcompiler_47.dll`
  with `D3DCOMPILE_OPTIMIZATION_LEVEL3`.
- The build compiles every `[numthreads]` entry point (`build/CompileComputeShaders.cs`, run by
  `ParakeetRuntime.Net.csproj` on Windows) and embeds the DXBC as `Shaders/<file>.<entry>.dxbc`.
  `EmbeddedPipeline` hands that bytecode to `CreateComputePipelineState`, so a load runs no HLSL
  compiler. FXC took several seconds of CPU per load before. The `.hlsl` sources stay embedded:
  without bytecode (a build on another OS) the entry point compiles at run time, and `--gemm-bench`
  compiles shader files given to it. The task keeps a SHA-256 of the sources' and its own content in
  `obj/<configuration>/<framework>/shader-bytecode/` and compiles again when it differs or an output
  is missing. `ParakeetGpu.Pipelines` and `LargeGemmPipeline` list the pipelines the engine creates;
  `ShaderBytecodeTests` checks that each one is embedded as what its current source compiles to.
- The pipelines are created in parallel (`Parallel.For`) while the weights upload, only after every
  tensor shape in the model has been checked against the model metadata and the shader limits. A
  bad model fails at once.
- The 64 x 64 Q8 GEMM (`Q8GemmMain`) pipeline is created in the background and may be ready after
  the model has loaded. Its `Task<nint>` (`largeGemmCompile`) is the only owner of that pipeline. Until the task
  has completed, `DispatchQ8Gemm` uses the 32 x 32 kernel, which gives the same bits (see below).
  If creating it fails, the 32 x 32 kernel serves throughout.
- On a failed load, the code waits for every pipeline still being created before it releases the
  pipelines (`WaitUntilNoCompileCanStoreAPipeline`), so that none can be stored into an array that
  has already been released.
- `Dispose` waits for a large-GEMM pipeline still being created (`WaitForLargeGemmPipelineOrZero`),
  because the device must outlive the call that uses it.

## Buffers

- `GpuBuffer` wraps one committed buffer resource on a DEFAULT, UPLOAD or READBACK heap. Sizes
  round up to 256 bytes. It tracks its current resource state, so `Transition` issues a barrier
  only when the state changes.
- The weights live in three large buffers: the Q8_0 quants (`wqMega`), their fp16 scales
  (`wdMega`) and every float tensor (`floatMega`). Each tensor starts at a 256-byte aligned offset.
- Activation scratch buffers (`bufX`, `bufA`, `bufB`, `bufWide`, the subsampling and Mel buffers,
  the token buffer) grow on demand to the largest audio, frame count and `T` seen so far, and are
  then reused.
- The projected positions (`pe × linear_pos`, `2T − 1` rows per layer) depend only on the
  position, never on the audio. The first utterance of 20 s or less (`T ≤ 251`) fills a table of
  501 rows per layer for all 24 layers (47 MB), with the same Q8 GEMM writing at a per-layer offset;
  every later utterance up to 20 s binds the rows it needs from the middle of that table (an SRV
  byte offset) instead of 24 positional GEMMs per utterance. Each output row of the Q8 GEMM is the
  same chain of multiply-adds whatever the row count, so a slice of the table has the bits a
  per-utterance projection would have. Longer utterances still project per utterance, and so does
  every utterance when the model's position table has fewer than 501 rows or the table could not be
  allocated; a failed allocation keeps the GPU. The table is filled in a submit of its own. This is
  8% of the encoder's GEMM work; on the laptop's MX450 transcription got about 6% faster.
- The decoder state buffers are fixed in size and small. Each LSTM layer's hidden state `decH[i]`
  holds two slots (`LstmHiddenPingPongSlots`), so `LstmLayerCondMain` reads one slot and writes the
  other in the same dispatch.

## Command lists and resource states

- The device has two command allocators and takes them in turn. `BeginCommands` waits until the
  next allocator's last submission has finished, resets it and the list, and sets the root
  signature. `Submit` closes the list, submits it and signals the fence, and returns the fence
  value to wait for; `WaitFor` waits for it, and `ExecuteAndWait` does both. The fence value
  advances only after `Signal` succeeds, so a wait never targets a value that was never queued. If
  `SetEventOnCompletion` fails, `WaitFor` polls the fence every millisecond until it is reached or
  the device is removed.
- The weight upload uses this to overlap: it fills one pair of staging buffers (16 MB + 2 MB, or
  as large as the largest staged tensor with the scales' buffer an eighth of that) from the model
  file while the GPU copies the other pair. A buffer pair is refilled only after the
  copy that read it has finished. The staging buffers are released only when the GPU is idle,
  also on a failed load. The driver frees released upload buffers only at some later submit,
  seconds after the load, so an app that loads and waits for speech holds them until then; the
  pairs are small for that reason; the default model's largest tensor (the joint output, 5.2 MB)
  fits in 16 MB.
  The GPU copy is never what the load waits for, so smaller pairs don't make it slower.
- D3D12 resource-state rules apply: a buffer a shader reads is in
  `NON_PIXEL_SHADER_RESOURCE`, a buffer it writes is in `UNORDERED_ACCESS`, and copies need
  `COPY_DEST` / `COPY_SOURCE`. A buffer written by one dispatch and read or written by the next,
  while it stays in `UNORDERED_ACCESS`, needs a UAV barrier between them.
  `MakeUavBarrierOrTransition` gives whichever of the two the buffer needs.
- After a device removal (TDR or page fault), `ID3D12Fence::GetCompletedValue` returns
  `UINT64_MAX` instead of blocking. `ThrowIfDeviceRemoved` turns that into an exception with the
  removal reason.

## Staying under the Windows GPU timeout

Windows resets a GPU whose work runs longer than its timeout detection limit (TDR, 2 s by
default). A slow or integrated GPU could exceed that on one long utterance, so the work is split
into submits sized to `SubmitBudgetMsUnderGpuTimeout`:

- Encoder: the first utterance times layer 0 alone (after flushing the Mel and subsampling work)
  to learn the time per layer and frame. Later layers go into submits of as many layers as fit
  the budget. A fast discrete GPU fits all layers into one submit.
- Every timing sample updates the estimate with `TrackSlowerAtOnceFasterGradually`: a slower
  sample counts at once, a faster one only gradually, so the estimate stays on the safe side.
- When the encoder fits in one submit, the first decoder batch shares that submit and gets the
  budget the encoder leaves. GPU timestamps then split the time between encoder and decoder.
  When the encoder is split, the decoder gets its own submits and wall-clock time splits them.
- Decoder: `DecodeBatchSteps` asks for 3/5 of the frames left plus 16 steps (a heuristic, not a
  bound: a TDT step advances 0 to 4 frames, and the next batch picks up what is left), but only as many as fit the budget at the measured time per step. Until a step has
  been timed, a batch is `UncalibratedDecodeBatchSteps`. The decoder step time comes from the
  wall time of decoder-only submits, recording and fence wait included, which is on the safe side.
- Decoding continues in batches until the frame pointer reaches `T`, and fails if it does not
  advance within a step limit.

## GPU decoder step

Each decoder step is `predLayers + 3` dispatches (five for this model, with two LSTM layers), all gated on the GPU by the current frame `t < T`, so a
batch can run more steps than needed without changing the result:

1. `RecordLstmLayers`: one fused LSTM dispatch per layer. Layer 0 reads the embedding of the last
   token; layer `i > 0` reads the slot of `decH[i - 1]` that the layer below just wrote.
2. `RecordJointPred`: `Q8GemvCondMain` in mode `GemvModeJointPredFromPingPongSlot`. It runs only
   when a new prediction is needed and reads the slot the last LSTM layer just wrote
   (`1 - lstmSlot`; `RecordArgMaxAndStep` flips the slot afterwards).
3. `RecordJointOutWithFusedRelu`: `Q8GemvCondMain` in mode `GemvModeJointOutWithFusedRelu`. It
   computes `max(0, encProj[t] + predProj)` on the fly and multiplies by the joint output weights.
4. `RecordArgMaxAndStep`: argmax over tokens and durations, the frame and symbol-count update, the
   LSTM slot flip, and appending the token.

Token IDs are read back with the decoder state when there are few of them
(`InlineTokenCopyCount`); a longer list takes one extra copy.

## Q8_0 GEMM

`Q8Gemm.hlsl` computes `Y[M, N] = X[M, K] * dequant(W[N, K])^T`, plus an optional bias and a
fused epilogue (`GpuEpilogue`: store, SiLU, add to `Y`, add half to `Y`).

- The weights use the ggml Q8_0 format: blocks of 32 signed 8-bit values along `K`, one fp16
  scale per block. `K` is a multiple of 32. The quants and the scales are in separate buffers
  (`Wq` and `Wd`).
- Each k block is 32 wide, exactly one Q8_0 block, so a block's scale is loaded once per row.
- There are two kernels, and `Q8GemmTile` picks one by the number of rows `M`:
  - `Q8GemmMain`: a 64 x 64 output tile, 16 x 8 = 128 threads, each computing 8 rows x 4
    columns (rows `ty*4..+3` and `32+ty*4..+3`, columns `tx*4..+3`). Shared memory is two
    `float4[32][16]` arrays, 16 KB.
  - `Q8GemmSmallMain`: a 32 x 32 output tile, 8 x 8 = 64 threads, 4 x 4 outputs each, 8 KB of
    shared memory. Up to `RowsUpToWhichLargeTileUnderfillsTheGpu` rows, it starts twice as many
    thread groups per axis (four times as many in all) as the 64 x 64 tile would, which keeps a large GPU busy on short utterances.
- Each thread stages 4 rows x 4 k values of `X`, and the same of `W`, transposed into one
  `float4` per k: `rowQuad = tid >> 3` is the group of four rows, `kCol = 4 * (tid & 7)` the first
  k. The shared arrays are indexed `[k][rowQuad ^ (k >> 2)]`, an XOR swizzle of the quad column.
  A `[k]` row is 16 or 8 `float4`s, a multiple of the 32 four-byte banks, so without the swizzle
  the 8 threads that share a `rowQuad` (one per `kCol`) store to the same banks: an 8-way
  conflict on every staging store. With it they store to 8 different `float4` columns. The reads
  undo it (`ty ^ (k >> 2)`, `tx ^ (k >> 2)`; `k >> 2` is a constant per unrolled `k`), so each
  thread reads the same values in the same order and the bits don't change. On the laptop's
  MX450 this made transcription about 15% faster.
- Four consecutive rows load under one bounds check while all four are inside the matrix
  (`LoadX4Rows`, `LoadW4Rows`); only the last partial group checks each row.
- Every out-of-bounds load sits under a real `[branch]`. Without it, FXC may hoist a root SRV load
  above the bounds check and read out of bounds.
- FXC's `[unroll(n)]` with `n` below the trip count runs only `n` iterations, silently. The k loops
  therefore use a plain `[unroll]` and unroll fully.

### Why both tiles give the same bits

Both kernels compute every output as the same chain of multiply-adds over k, in k order, from the
same inputs. So on the GPUs tested they give identical bits, and switching tile by row count, or
using the small kernel while the large one still compiles, does not change a transcript. The test
`Gpu_q8_linear_kernels_give_the_same_bits` checks this.

A driver is still free to contract `a * b + c` into a fused multiply-add differently in the two
programs, which would change the rounding. So identical bits are a fact on the GPUs tested, not a
guarantee from the language.

## GEMM benchmark

`parakeet-runtime --gemm-bench` times the encoder's Q8 matrix multiplications alone, on every GPU,
to compare shader variants in seconds:

```
--gemm-bench [--variant Entry:tile[:file.hlsl] ...] [--frames 1500,250] [--device gpu:RTX]
```

- A variant is an entry point, its square output tile and a shader file (by default the shipped
  `Q8Gemm.hlsl`). Without `--variant`, both shipped kernels run.
- Each variant runs one Conformer layer's products (`ConformerLayerQ8Products`) for each frame
  count and reports the time per layer and its speed relative to the first variant.
- Each product is timed as the best of three submits of `DispatchesPerTimedSubmit` dispatches,
  after one warm-up submit.
- The output starts as NaN, so outputs a variant never writes are counted and reported, and every
  other output is compared with the first variant's.

The products of one Conformer layer of Parakeet TDT 0.6B (`d = 1024`, feed-forward 4096), with `T`
frames:

| Product | Output columns | Input width | Rows | Per layer |
| --- | --- | --- | --- | --- |
| Feed-forward in | 4096 | 1024 | `T` | 2 |
| Feed-forward out | 1024 | 4096 | `T` | 2 |
| q, k, v | 3072 | 1024 | `T` | 1 |
| Positional projection | 1024 | 1024 | `2T - 1` | 1 |
| Attention out, pointwise conv 2 | 1024 | 1024 | `T` | 2 |
| Pointwise conv 1 | 2048 | 1024 | `T` | 1 |
