This document describes what the decoder, encoder, subsampling and mel HLSL shaders compute, their data layouts, and the rules they rely on.

# GPU shaders

The shaders live in `src/ParakeetRuntime.Net/Shaders/`. The build compiles each entry point through the
inbox `d3dcompiler_47.dll`, for shader model 5.1 (`cs_5_1`) at optimization level 3, and embeds the
bytecode (see [gpu-path.md](gpu-path.md#pipelines-and-compilation)). The host refers to entry points by
their string names, so an entry point must never be renamed. An entry point is a function with a
`[numthreads(...)]` attribute directly before `void <name>(`, outside comments; the build finds them
that way. The engine names the ones it uses in `ParakeetGpu.Pipelines` and `LargeGemmPipeline`.

## Shared conventions

- Every file declares one constant buffer, `cbuffer Params : register(b0)`, of eight `uint` values
  set as root constants. The encoder, decoder, subsampling and mel files call them `P0` to `P7`
  and name them as locals in each entry point; `Q8Gemm.hlsl` names the fields directly (`M`, `N`,
  `K`, `hasBias`, `epilogue`, `xByteOffset`, `yByteOffset`, `pad0`).
- Inputs are `ByteAddressBuffer In0`, `In1`, … at `t0`, `t1`, …; outputs are
  `RWByteAddressBuffer Out0`, `Out1` at `u0`, `u1`. All addresses are in bytes, so element `i`
  of a float buffer is at `i << 2`.
- Element-wise shaders run 256 threads per group and flatten a 2D dispatch grid into one index:
  `(gid.y * 65535 + gid.x) * 256 + gtid.x`. D3D12 allows at most 65535 groups per dispatch
  dimension (`MaxGroupsPerDispatchDimension`), so large tensors spill into `gid.y`. Each shader
  returns early when the index passes the element count.
- Reductions (sums, maxima) use a 256-entry groupshared array and a tree that halves the active
  threads each step (128, 64, …, 1), with a barrier after each step.
- Q8_0 weights: `Wq` holds signed bytes, row-major `[N, K]`. `Wd` holds one fp16 scale per block
  of 32 values along K, `[N, K / 32]`, packed two per `uint` (even block in the low half).
  `DotS8x4` unpacks four signed bytes from one `uint` and dots them with a `float4`.

## DecoderOps.hlsl: TDT greedy decoder on the GPU

The greedy token-and-duration (TDT) decode loop runs entirely on the GPU. The host records
hundreds of decode steps into one command list; each step is a fixed sequence of dispatches,
and steps that are not needed turn into no-ops by reading the decoder state.

### Decoder state

`decState` is six `uint` values. The shaders address them by byte offset:

| Byte | Constant | Meaning |
| --- | --- | --- |
| 0 | `DecStateFrameByte` | `t`, the current encoder frame, `0..T`. Decoding is finished when `t >= T`. |
| 4 | `DecStateSymbolsByte` | Symbols emitted at the current frame. |
| 8 | `DecStateTokenCountByte` | Number of tokens written to the token buffer (may exceed its capacity; only the first `maxTokens` are stored). |
| 12 | `DecStateLastTokenByte` | Last emitted token id, or `NoToken` (`0xFFFFFFFF`) before the first one. |
| 16 | `DecStateNeedPredictByte` | 1 when the prediction network (LSTM and joint prediction projection) must run before the joint output, else 0. |
| 20 | `DecStateLstmSlotByte` | 0 or 1: which ping-pong slot of each `decH[i]` holds the current hidden state. |

`InitDecStateMain` writes `[0, 0, 0, NoToken, 1, 0]`.

### Uniform early return before barriers

Every gated shader reads `decState` from `StateIn`, which holds the same value for every thread,
and returns before its first `GroupMemoryBarrierWithGroupSync`. Because the condition is uniform,
either the whole group returns or none of it does, which keeps the barriers legal. Gating is
always `t >= T` and, for the prediction network, `needPredict == 0`.

`ArgMaxAndStepMain` is the exception: it writes the state it reads, so it reads it from `Out0`,
keeps every thread through the reduction barriers, and only acts when `t < T` (`active`).

### Entry points

- `InitDecStateMain` (1 thread): resets `decState`.
- `ZeroBufferMain` (256 threads): zeroes `P0` floats of `Out0`. Used to clear the LSTM state.
- `LstmLayerCondMain`: one LSTM layer, fused: load the input (embedding row of the last token, or
  the previous layer's new hidden state), compute `W_ih x + W_hh h + b` for the four gates, and
  apply the cell update. Gated on `t < T` and `needPredict`.
  - `P0 = H` (at most 1024, a multiple of 32), `P1 = T`, `P2` = 1 when the input is the
    embedding table (`LstmInputIsEmbedding`), `P3`/`P4` = how many bytes after the `ih` weights
    and scales the `hh` ones start, relative to the `ih` views bound at `In1`/`In2`, not absolute
    offsets.
  - `In0` = embedding `[vocab, H]` or the previous layer's `decH` (`2 * H` floats; the new
    hidden state is in that layer's destination slot), `In1` = Q8 weights, `In2` = scales,
    `In3` = pre-summed `ihBias + hhBias` `[4 * H]`, `StateIn` = `decState`.
  - `Out0` = this layer's `decH` (`2 * H` floats, two ping-pong slots), `Out1` = its cell state
    `decC` (`H` floats, updated in place).
  - Reads `h` from slot `lstmSlot` and writes the new `h` to slot `1 - lstmSlot`, so the two never
    overlap. With no previous token (`NoToken`) the embedding input is zero.
  - 128 threads: 32 gate rows (4 gates `i, f, g, o` × 8 hidden units) × 4 lanes along K. A group
    produces 8 hidden units; dispatch `((H + 7) / 8, 1, 1)`. Gate order in weights and bias is
    `i, f, g, o`.
  - Both `tanh`s (the `g` gate and the new cell state) go through `TanhWithoutOverflow`,
    `sign(x) * (1 - e) / (1 + e)` with `e = exp(-2|x|)`. The cell state is not bounded and reaches
    ±88 in real speech, and the compiled HLSL `tanh` overflows `exp` there: it returns 0 at 88 and
    NaN above about 88.7. One NaN hidden value turns the whole joint prediction into NaN, the fused
    ReLU turns that into zeros, and the decoder then emits the joint bias's token on every frame.
- `Q8GemvCondMain`: `Y[N] = X[K] · Wq^T * scale + bias`, one row at a time, with optional gating
  and fused input. 64 threads = 16 output rows × 4 lanes along K; dispatch `((N + 15) / 16, 1, 1)`.
  `P0 = N`, `P1 = K` (at most 1024, a multiple of 32), `P2` = has bias, `P3 = T`, `P4` = mode:

  | Mode | Constant | Gate | Input |
  | --- | --- | --- | --- |
  | 0 | `GemvUngated` | none | `In0[0..K-1]` |
  | 1 | `GemvWhileFramesLeft` | `t < T` | `In0[0..K-1]` |
  | 2 | `GemvWhilePredictPending` | `t < T` and `needPredict` | `In0[0..K-1]` |
  | 3 | `GemvJointPredFromLstmOutput` | `t < T` and `needPredict` | `In0[(1 - lstmSlot) * K ..]`: the last LSTM layer's new hidden state (joint prediction projection) |
  | 4 | `GemvJointOutFromEncoderPlusPred` | `t < T` | `max(0, In0[t * K + i] + In5[i])`: encoder projection at frame `t` plus prediction projection, then ReLU (joint output) |

- `ArgMaxAndStepMain` (256 threads, one group): advances the decoder by one step.
  - `In0` = logits `[vocab + numDur]`, `Out0` = `decState`, `Out1` = tokens `[maxTokens]`.
  - `P0 = vocab`, `P1 = blank`, `P2 = numDur`, `P3 = T`, `P4` = max symbols per frame,
    `P5 = maxTokens`, `P6`/`P7` = duration values packed one byte each: duration index `d` is
    byte `d & 3` of `P6` when `d < 4`, else of `P7` (so at most 8 durations).
  - Token = argmax of the first `vocab` logits; ties go to the lowest index. Duration = argmax of
    the next `numDur` logits; ties go to the first.
  - The frame advances by the chosen duration. A blank with duration 0 advances by 1.
  - If the LSTM ran this step (`needPredict` was 1), `lstmSlot` flips, so the slot just written
    becomes current.
  - A non-blank token is appended (when there is room), becomes `lastTok`, and sets `needPredict`
    to 1; a blank sets it to 0, so the next step reuses the prediction.
  - When the frame does not advance, the symbol count grows; at `maxSymbols` the frame advances
    by 1 and the count resets.

## EncoderOps.hlsl: conformer layer operations

- `LayerNormMain`: one group per row. `Out0[r, i] = (In0[r, i] - mean) / sqrt(var + 1e-5) *
  Gamma[i] + Beta[i]` with the population variance. `In1 = Gamma`, `In2 = Beta`; dispatch
  `(T, 1, 1)`, `P0 = D`.
- `RelPosAttentionMain`: relative-position multi-head self-attention, tiled like FlashAttention
  with an online softmax. See below.
- `GluMain`: `Out0[t, ch] = In0[t, ch] * sigmoid(In0[t, D + ch])` over `[T, 2D] → [T, D]`.
  `P0 = T * D`, `P1 = D`; `(T * D + 255) / 256` groups, spread over a 2D grid.
- `DepthwiseConvBnSiluMain`: depthwise 1D convolution over time with "same" zero padding
  (`(kernel - 1) / 2` on each side), then batch norm folded into a per-channel scale and shift,
  then SiLU. `In0 = Glu[T, D]`, `In1 = Dw[D, kernel]`, `In2 = BnScale[D]`, `In3 = BnShift[D]`,
  `Out0 = [T, D]`; `P0 = T`, `P1 = D`, `P2 = kernel`.

### Relative-position attention

Inputs: `In0 = QKV[T, 3D]` (Q, K, V side by side per frame), `In1 = PosP[2T - 1, D]` (projected
relative position embeddings), `In2 = BiasU[D]`, `In3 = BiasV[D]`. Output `Out0 = Ctx[T, D]`.
Dispatch `((T + 31) / 32, heads, 1)`; `P0 = T`, `P1 = D`, `P2 = dk` (a multiple of 4, at most
128), `P3 = asuint(scale)`.

Score for query `i`, key `j`: `((q_i + u) · k_j + (q_i + v) · p_{T-1-i+j}) * scale`. The shader
expands it to `q·k + q·p + u·k + v·p`, so Q is stored once instead of twice, and the `u·k` and
`v·p` terms are computed once per tile and shared by every query row.

- Tiles: 32 queries × 32 keys, stepping 32 head elements (8 `float4`) at a time.
- Threads: 16 × 16 = 256. Thread `(tx, ty)` computes scores for rows `ty`, `ty + 16` and columns
  `tx`, `tx + 16`, and accumulates output elements `tx`, `tx + 16` (as `float4`) for its two rows.
- Positions: a query tile at `i0` and key tile at `j0` need 63 position rows starting at
  `(T - 1) - (i0 + 31) + j0`. Score `(r, c)` uses tile position row `31 - r + c`. For the thread's
  four scores: `(r0, c0)` and `(r1, c1)` use `p00`, `(r0, c1)` uses `p00 + 16`, `(r1, c0)` uses
  `p00 - 16`. Rows outside `[0, 2T - 1)` load as zero.
- `sTile` is reused per phase. Phase 1 holds `sQ[32][8]` at float4 0, `sK[32][9]` at 256
  (`AttnKTileStart`) and `sP[63][9]` at 544 (`AttnPosTileStart`); rows of K and P are padded to 9
  float4s (`AttnPaddedRowFloat4s`) to avoid shared-memory bank conflicts. Phase 3 holds
  `sV[32][32]` at 0. The score tile `sS[32][33]` is padded to 33 for the same reason.
- Phase 1: score tile. Threads 0–31 also compute `u · K[c]` into `sUK`, threads 32–94 compute
  `v · P[p]` into `sVP`; then each thread adds them to its four scores and scales. V for the key
  tile is loaded after the last barrier of the K/P loop, when `sTile` is free again.
- Phase 2: online softmax, one thread per query row. New max `m`; `alpha = exp(mOld - m)`
  (0 on the first tile); weights `exp(s - m)`; running sum `l = l * alpha + sum(weights)`. Keys
  past `T` get weight 0. Rows past `T` get `alpha = 0`, `l = 1`.
- Phase 3: `O = alpha * O + W · V`. At the end, `O / l` is stored for rows inside `T`.
- Overflow guard (`ExpTreatingNaNAsZero`): the exponent is `min(x, 0)`. After subtracting the
  max, every finite `x` is already `<= 0`, so this changes nothing for them. When a score is
  infinite, `x` is `inf - inf = NaN`; D3D's `min` returns the other operand for a NaN, so the
  weight becomes `exp(0) = 1` instead of NaN, and the row still sums to 1.

## SubsampleOps.hlsl: FastConformer `dw_striding` subsampling

- `SubConv0ReluMain`: 3×3 convolution, stride 2, padding 1, one input channel to `C`, plus bias
  and ReLU. `In0 = feats[H, W]`, `In1 = weight[C, 9]`, `In2 = bias[C]`, `Out0 = [H1, W1, C]`
  (channels last). `P0..P4 = H, W, H1, W1, C`.
- `SubDwConvMain`: 3×3 depthwise convolution, stride 2, padding 1, `C → C`, plus bias.
  `In0 = x[H0, W0, C]`, `In1 = [C, 9]`, `In2 = [C]`, `Out0 = [H2, W2, C]`.
  `P0..P4 = H0, W0, H2, W2, C`.
- `F32GemmReluMain`: tiled float32 GEMM used as the 1×1 pointwise convolution:
  `Out0[M, N] = max(0, In0[M, K] · In1[N, K]^T + In2[N])`. A 16 × 8 group computes a 32 × 64
  tile, each thread 4 × 4 outputs. K must be a multiple of 32 (the subsampling uses `C = 256`).
  Shared tiles are padded (`[32][33]`, `[32][65]`) to avoid bank conflicts. Dispatch
  `((N + 63) / 64, (M + 31) / 32, 1)`; `P0..P2 = M, N, K`.
- `SubTransposeMain`: `[T, W, C] → [T, C, W]`, so each frame flattens channel-major before the
  output projection. `P0..P2 = T, W, C`.

## MelOps.hlsl: log-mel spectrogram

The shaders assume `nFft = 512`: 256 threads load two samples each, and the FFT has 9 stages.

- `MelFftLogMain`: one group per frame. Pre-emphasis, Hann window, 512-point radix-2
  decimation-in-time FFT, power spectrum, mel filterbank, log.
  - `In0 = audio[nSamples]`, `In1 = window[win]`, `In2 = fb[nMels, 257]`,
    `In3 = twiddles[1024]`, `Out0 = feats[frames, nMels]`.
  - `P0 = nSamples`, `P1 = hop`, `P2 = win`, `P3 = woff` (where the window starts inside the
    512-sample frame), `P4 = nMels`, `P5 = asuint(preemph)`, `P6 = validFrames` (groups at or past
    it return at once; 0 runs every group).
  - Frame `f` covers samples `f * hop - 256 .. f * hop + 255` (`FrameCentreOffset`): the CPU pads
    the signal with `nFft / 2` zeros on the left, and samples outside the audio read as zero.
    Pre-emphasis `x[n] - preemph * x[n - 1]` (the first sample is kept) is applied on the fly.
  - Samples are stored at bit-reversed positions (`BitReversedFftIndex`, 9 bits), so the
    butterflies run in place. Stage `s` (`halfLen = 2^s`) reads twiddle `halfLen + k`.
  - Power is kept for bins 0..256 (`PowerBins = 257`, bin 256 is the Nyquist bin).
  - Output is `log(sum + 2^-24)` (`LogGuard`).
- `MelNormalizeMain`: one group per mel bin, in place on `Out0[frames, nMels]`. Mean over valid
  frames; variance divided by `valid - 1` (by 1 when `valid <= 1`); value
  `(x - mean) / (std + 1e-5)`; frames at or past `valid` become 0. `P0 = frames`,
  `P1 = validFrames`, `P2 = nMels`; dispatch `(nMels, 1, 1)`.

### Twiddle table shared with Mel.cs

`Mel` precomputes the FFT twiddles stage by stage: for `len = 2, 4, …, nFft`, the `len / 2`
entries `cos(-2πk / len)` and `sin(-2πk / len)` follow each other, so stage `s` starts at index
`2^s - 1`. Each entry is `Math.Cos(ang * k)` / `Math.Sin(ang * k)` with `ang = -2π / len`, the
same values a butterfly would compute inline, so the table changes no result.

`Mel.BuildTwiddlesF32` packs them as `float[2 * nFft]`: cosines in `[0, nFft)`, sines in
`[nFft, 2 * nFft)` (`TwiddleSinStart = 512` in the shader), each shifted up by one so that stage
`s` starts at `halfLen = 2^s` and entry `k` is at `halfLen + k`. Index 0 of each half is unused.

### CPU reference in Mel.cs

- `fb` is `[nMels, nBins]` row-major, `nBins = nFft / 2 + 1`.
- For each filter, `firstNonzeroBin`/`endNonzeroBin` bound its nonzero weights. The bins outside
  add exactly zero, so skipping them does not change the sum.
- The window is a symmetric Hann window (`hann_periodic = false`): it divides by `win - 1`.
- `Compute` returns features laid out `[frame, mel]` and the frame count. Frames past
  `validFrames = min(n / hop, frames)` are zero, as on the GPU.

## What the GPU tests pin

`tests/ParakeetRuntime.Net.Tests/D3D12KernelsTests.cs` compares each shader with a CPU reference on
every eligible hardware adapter, and skips when there is none.

- Q8 GEMM (`Q8GemmMain` with 64-row tiles, `Q8GemmSmallMain` with 32-row tiles): the cases cover
  M and N that are not multiples of 32 with exact-size buffers (455 × 131), and several 64-wide
  tiles with both edges partial, where row 96 sits alone in the lower half of a tile (97 × 200).
- The engine picks the Q8 GEMM kernel by row count, and uses the small one until the large one
  has compiled. So on a given GPU, neither an utterance's length nor when it runs may change its
  transcript: both kernels must give the same bits for every epilogue, with and without bias, with
  the last 64-row tile partial in its lower half (150 rows: 22 in the last tile), in its upper half
  (170 rows: 42), or full (128), and on the scalar path for the last columns (`n % 4 != 0`).
- The positional projection reads its rows through an SRV that starts inside a larger buffer,
  plus a byte offset in the constants. The probe fills the bytes before the input with NaN, so a
  read that ignores either offset spoils the output. An odd row count (97) also covers the
  partial-row loads.
- The epilogue test uses 67 rows: a full 64-row tile, where every thread stores all its rows, then
  a partial one.
- Attention: one frame (`dk = 64`); exactly one 32 × 32 tile (`dk = 128`); several tiles plus a
  one-frame tail, which exercises the online-softmax rescaling and out-of-bounds reads through a
  root SRV (65 frames); several tiles plus a tail with `dk = 64`; and `dk = 96`, not a power of two.
- Attention over one frame whose query is infinite: the score overflows, softmax over one key is
  still 1, and the output must equal V (the overflow guard above).
- Mel: 8000 samples, and 10 samples, which give zero valid frames.
