# Ideas not built yet

Performance ideas that were proposed and checked on paper, or measured only partly, and are not in
the engine. Each says what it would gain, what is known, and what proving it takes. Measured values
belong in [benchmarks.md](benchmarks.md) once an idea is built; the figures here come from the
2026-11-26 and 2026-11-27 research rounds on the laptop (i7-11850H, MX450, Intel UHD) and are
marked as estimates where nothing was timed. When an idea is built or rejected, remove it here.

## GPU

### Reuse compiled pipelines across app starts

- **Gain:** GPU load. Creating the 17 compute pipelines costs the driver 1.0 s of CPU on the MX450
  and 1.5 s on the Intel UHD (the large Q8 GEMM alone 0.26 s and 0.77 s); from a cached blob
  (`ID3D12PipelineState::GetCachedBlob`, passed back in `D3D12_CACHED_PIPELINE_STATE`) the same
  pipelines take 2–3 ms. Estimated 100–200 ms off a GPU load, and the large GEMM would be ready
  before the first utterance.
- **Known:** measured with a probe, not in the engine. The blobs are 0.4 MB (MX450) and 0.6 MB
  (UHD).
- **Open:** it writes a cache to disk, for example `%LOCALAPPDATA%\ParakeetRuntime\pso-cache`, keyed
  by adapter and bytecode hash; a blob from another driver version must fall back to a fresh
  compile. The driver may already cache across processes, so measure a second process start before
  and after.

### int8 dot products in the Q8 GEMM (Shader Model 6.4)

- **Gain, estimated:** the encoder's Q8 GEMMs are 84% of GPU time on the MX450 (2026-11-27
  record) and run fp32 multiply-adds at about half the card's fp32 peak. `dot4add_i8packed` does
  four int8 multiply-adds per instruction, with activations quantized per 32-block as the CPU path
  already does. If the hardware runs it at the fp32 instruction rate, the GEMMs could be up to about
  twice as fast. Of the 2026-11-27 research round, Neon expected 2–3x end to end on the MX450 and
  the UHD, and Fable expected little on NVIDIA cards outside tensor cores.
- **Known:** FXC compiles only up to `cs_5_1`; this needs DXC at build time (the
  `Microsoft.Direct3D.DXC` package), which produces signed DXIL that the inbox `d3d12.dll` loads, so
  no new runtime DLL. GPU transcripts would change towards the CPU path's int8 rounding.
- **Next:** prove the hardware rate first. Two compute shaders with the bindings of `Q8Gemm.hlsl`,
  one doing fp32 `mad` and one `dot4add_i8packed` at the same multiply-add count as a layer's GEMMs,
  compiled with DXC to `cs_6_4` and timed by `--gemm-bench` (which then has to accept compiled
  bytecode). Build the GEMM only if `dot4add_i8packed` is clearly faster.

### Large GPUs: tile threshold, split-K and the decoder

- The small-GPU tile threshold (40 rows below 4 GiB of dedicated memory) is measured on the MX450
  only. On the RTX 5070 Ti the 96-row threshold stays as tuned on 2026-11-22; re-check 48–96 rows
  there with `--gemm-bench --frames 48,64,80,96`.
- At 120 rows and 1,024 outputs the 64 × 64 tile starts 32 thread groups, under half of the RTX
  5070 Ti's 70 multiprocessors. Split-K, or the small tile, may fill it. Desktop only.
- The decoder is 8% of GPU time on the MX450 (2026-11-27 record). Whether one dispatch per batch
  would help is open: the per-operation counts include dispatches that return at once (no
  prediction needed, or past the end of the clip), so they don't show how long the working ones
  take. Count the working steps first. On the RTX the decoder's share has never been recorded: run
  `--gpu-operations` there.

### GPU subsampling in strips

- **Gain:** GPU memory. The subsampling planes (`bufSubA`, `bufSubB`) are 86% of the GPU scratch:
  about 78 MB at 20 s and 470 MB for 2 minutes of uncut audio. Processing them in strips of 64
  output frames with the right halo would need about 20 MB at any length. Since 2026-11-27 the
  planes grow only to the frames needed, which ended the MX450's spill into shared memory on
  2-minute files; strips would still cut the memory.
- **Open:** halo indexing in `SubConv0ReluMain` and `SubDwConvMain`, and a test against the
  whole-plane result for frame counts at and off strip boundaries.

### Coalesced decoder GEMV

- **Gain:** about 13% of the GPU decoder, which is 1% of GPU transcription on the MX450 (measured
  2026-11-26, 150 clips). Too small to build alone; worth it with another decoder change.

## CPU

### 7-bit activations on CPUs without VNNI

- **Gain:** about 15% of CPU kernel time on AVX2-only CPUs (this laptop: 31–38 ms → 28–29 ms per
  layer set at 250 frames).
- **Why not built:** it changes transcripts (34 of 300 clips, WER 358 → 357 errors) and breaks the
  contract that the CPU quantizes exactly as NeMo-Speech.cpp does (see
  [cpu-kernels.md](cpu-kernels.md)). Declined by the owner on 2026-11-26.

### Relative-attention position strips

- **Gain:** packing each 16-column strip of the position table once per task cut the attention
  kernel at 1,500 frames from about 180 ms to 100–150 ms. At 250 frames it showed no gain, and
  `SpeechEngine` never runs more than 251 frames at once, so only `--corpus` without `--cut` would
  benefit.

### Smaller ones

- Reuse the first subsampling convolution's shared row between neighbouring outputs: 33% fewer row
  evaluations, estimated 0.2–0.6% of a CPU transcription.
- Decode GGUF blocks straight into the packed layout: saves one pass over 666 MB at load; a
  prototype that also did this ran slower than loading the layers in parallel, which is built.

## API

### Keep the Direct3D device between loads

- **Gain:** about 0.25 s per GPU reload. Needs a decision on what `Dispose` releases, as part of
  the API design before 1.0.
