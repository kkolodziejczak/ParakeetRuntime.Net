# Ideas not built yet

Performance ideas that were proposed and checked on paper, or measured only partly, and are not in
the engine. Each says what it would gain, what is known, and what proving it takes. Measured values
belong in [benchmarks.md](benchmarks.md) once an idea is built; the figures here come from the
2026-11-26 research round on the laptop (i7-11850H, MX450, Intel UHD) and are marked as estimates
where nothing was timed. When an idea is built or rejected, remove it here.

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

### A lower large-tile threshold on small GPUs

- **Gain:** GPU transcription of 4–8 s clips. `RowsUpToWhichLargeTileUnderfillsTheGpu` (96 rows)
  was tuned on an RTX 5070 Ti. On the MX450 the 64 × 64 tile already beat the 32 × 32 one at 64 and
  96 rows in `--gemm-bench` (1.25× and 1.39×, contended).
- **Open:** pick the threshold per adapter (for example by its multiprocessor count, or a short
  calibration after load) and measure the corpus on the MX450, the UHD and the RTX.

### GPU subsampling in strips

- **Gain:** GPU memory. The subsampling planes (`bufSubA`, `bufSubB`) are 86% of the GPU scratch:
  about 78 MB at 20 s and 470 MB for 2 minutes of uncut audio. Processing them in strips of 64
  output frames with the right halo would need about 20 MB at any length.
- **Open:** halo indexing in `SubConv0ReluMain` and `SubDwConvMain`, a test against the whole-plane
  result for frame counts at and off strip boundaries, and a check that uncut long audio on the
  MX450 stops spilling into shared memory.

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
