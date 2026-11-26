This document explains how the CPU encoder kernels in `src/ParakeetRuntime.Net/Kernels.cs` work and why they are built this way.

# CPU kernels

## Instruction sets

- The fast paths need AVX2 and FMA (`HasAvx2AndFma`). Not every Windows 11 CPU has them (for
  example the Pentium Silver N6000). There, 128-bit vectors (SSE2 on x64, NEON on ARM) stand in.
  They are slower and give the same results within rounding.
- `MultiplyAdd` fuses `a * b + c` only where the CPU has FMA. Software FMA would be far slower than
  a separate multiply and add.

## int8 matrix products

`Int8Path` lists the int8 products from slowest to fastest. `Kernels.BestPath` is the fastest one
this CPU has, and the weights are packed for it once, at load.

| Path | Instruction | Operands | CPUs |
| --- | --- | --- | --- |
| `Portable` | 128-bit widening multiply | signed x signed | every x64 CPU, ARM |
| `Avx2` | `vpmaddubsw` + `vpmaddwd` | unsigned x signed, with sign tricks | AVX2 CPUs |
| `AvxVnni` | `vpdpbusd` | unsigned x signed | Alder Lake and later, Zen 5 |
| `AvxVnniInt8` | `vpdpbssd` | signed x signed | Arrow Lake, Lunar Lake and later |

`Linear` computes `Y[M,N] = act(X[M,K] * W[N,K]^T + bias)`. X arrives as floats and is quantized
first. `LinearAdd` computes `target[M,N] += scale * (X[M,K] * W[N,K]^T + bias)`: a residual step
without a buffer for Y. The `Epilogue` (bias, SiLU, scaled add) runs on each output while it is
still in a register.

### Packing

`Pack` takes over the arrays of the `Q8Matrix` it is given; that matrix is not usable afterwards.
Groups of 8 rows are packed in parallel, each into its own part of the output, moving the 4-byte
runs as whole `int` words.

- The AVX2 paths interleave 8 output rows, 4 bytes at a time. One 256-bit multiply-add then gives
  8 dot products of 4 values each, so a block of 32 needs 8 of them and one float scaling for 8
  outputs.
- When the row count is a multiple of 8, each group of 8 rows keeps its place in the arrays, so
  packing needs no second copy. Otherwise packing writes into new arrays, where the last partial
  group is padded with zero scales (and a zero bias).
- The portable path keeps plain rows.

### Tiles

- `PackedQ8Gemm` gives each thread a run of 8-row weight groups, which stay in L1 while all rows
  of X pass.
- Where two groups remain, `Tile4x16` takes four rows of X against both: 16 outputs per row. 6
  loads feed 8 multiply-adds (the 4 x 8 tile, `Tile4`, needs 5 loads for 4). The float sums spill
  once per block. Leftover rows go through `Tile1`.
- `IInt8Dot` is one step of the packed dot product: 8 lanes, each adding 4 products of x and w.

### The three dot products

- `VnniInt8Dot`: signed times signed, no correction needed.
- `VnniDot`: the instruction takes unsigned times signed. Flipping the top bit of w gives
  `w + 128` as unsigned, so the sum is `x.w + 128 * sum(x)`. Each block starts at
  `-128 * sum(x)` (`UnsignedWeightOffset`), which cancels the extra term. That is why quantizing
  stores each block's sum.
- `Avx2Dot`: `|w|` times x with the sign of w, in pairs to int16, then to int32. The pair fits in
  int16 (`2 * 128 * 127`). The absolute value goes to w, as in ggml: `|-128|` is 128 as unsigned,
  while x, quantized to -127..127, can always be negated.
- `PortableQ8Gemm` and `BlockDot` multiply two 32-value blocks with 128-bit vectors. Products of
  int8 fit in int16.

## Quantizing activations (Q8_0)

`Quantize` turns `X[M,K]` into int8 per 32-value block exactly as NeMo-Speech.cpp does on x86
(ggml's AVX2 `quantize_row_q8_0`, `arch/x86/quants.c`):

- scale = `amax / 127`, rounded to fp16;
- each value = `value * (127 / amax)`, rounded half to even;
- each block also stores the sum of its int8 values (`Q8Activations.Sums`), for `VnniDot`.

Edge cases:

- A block whose scale rounds to zero is silent: all zeros and a sum of 0. `127 / amax` could
  overflow there, and the values count for nothing anyway. Its inverse scale is
  `SilentBlockInverse` (0); every other block's inverse is at least `127 / float.MaxValue`.
- A block holding NaN or infinity gets a NaN or infinite scale (NaN when it holds both) and all
  zeros, wherever the bad value sits. What it multiplies is lost anyway.
- Finite blocks too loud for fp16 get an infinite scale and quantize as usual.

How it runs:

- The inverse scales of a row live on the stack, so `k` is limited to
  `WidestRowWithInverseScalesOnStack` (32 K values, 4 KB of inverses).
- Each row takes three passes: the largest magnitudes (vector), the fp16 scales (scalar), then the
  rounding (vector). .NET's `Half` conversion is much slower when it runs while 256-bit registers
  are live, so the two vector passes are `NoInlining` and the fp16 rounding sits between them.
- `LargestMagnitudes256` compares the floats as integers after clearing the sign bit. Without
  their sign bits, floats order like their bits: infinity above every number and NaN above
  infinity, so a NaN anywhere in the block becomes its amax. `Vector256.Max` spends 9 instructions
  per step on NaN and signed-zero rules, and `vmaxps` would keep or drop a NaN by its position.
- `Round256` packs with saturation, but the values never exceed 127. The packs work per 128-bit
  half, leaving the 4-value groups as `a0 b0 c0 e0 | a1 b1 c1 e1`, so a permute
  (`perHalfPackToSequentialOrder`) restores their order.
- `Quantize(..., avx2: false)` takes the 128-bit path of CPUs without AVX2, so tests can cover it
  on CPUs that have AVX2.

## Float kernels

- `GemmRelu` computes `Y[M,N] = ReLU(X[M,K] * Wt[K,N] + bias)`, with the weights transposed. N must
  be a multiple of two vectors (`RequireVectorPairs`); other widths are refused rather than
  written past. The subsampling's pointwise convolutions use it.
- `FloatRows` does up to 4 rows of `Y = X * W (+ bias, ReLU)` with row strides `ldx`, `ldw`, `ldy`.
  Tiles of 4 rows by 2 vectors keep 8 sums in registers.
- `Transpose` turns `[rows][columns]` into `[columns][stride]` with `stride >= rows`; the padding
  is zero.
- `NormRow` (layer norm) needs `d` to be a multiple of the vector width (the model uses 1024 and
  640).

## Relative-position attention

`RelativeAttention` is Transformer-XL relative-position self-attention, as FastConformer uses it,
for all heads:

```
score[i][j] = ((q_i + u).k_j + (q_i + v).p[first + T-1-i+j]) / sqrt(dk)
ctx_i       = softmax(score_i).V
```

`qkv` is `[T][3d]` (q | k | v), positions come from `TransposePositions`, and `ctx` is `[T][d]`.
`first > 0` reads a slice of a position table made for a longer utterance.

- `TransposePositions` stores the positional projections `p[P][d]` as `[d][stride]`, each head's
  dk rows together. The padding lets score tiles read whole vectors past the last position; those
  columns are ignored. Utterances over 20 s make a table per layer, so it comes from the scratch
  and all threads fill it, 16 destination rows at a time from whole source cache lines
  (`DestinationRowsPerSourceCacheLine`).
- The preparation pass lays out, per head, `q + u` and `q + v` as `[h][i][e]`; keys as
  `[h][pair of key columns][e][pair]`, so a score tile walks its dk rows of keys in one run instead
  of dk rows a whole width apart, each on its own page; and values as `[h][pair of columns][j][pair]`,
  so each pair of output columns reads one run. It goes head by head, so no element needs a
  division by `dk`, a runtime value.
- Each task takes `QueryRowsPerAttentionTask` (32) query rows of one head and streams that head's
  keys, positions and values once. That reads them 8 times less often than one 4-row tile at a
  time; on long utterances a head is larger than the L2 cache a core has to itself. The block's
  scores (32 x T floats) stay in L2. The count is a multiple of 4, so the row tiles, and with them
  every sum, are the ones a 4-row walk makes.
- Inside a task, a pair of key columns, with the positions it meets, is loaded once for every row
  tile; likewise a pair of value columns.
- In `ScoreTile4`, row r's position window starts one column further left than row r-1's: that is
  the relative shift.
- `Softmax` works over `s[0..n)`; `s[n..width)` is padding and comes out zero.
- `LargerOf` uses plain `vmaxps` where vectors are 256-bit AVX, as `Vector.Max` spends 9
  instructions per step on NaN and signed-zero rules. The softmax output is the same: a tie of 0
  and -0 may keep either, and exp gives both the same bits. A row holding NaN comes out all NaN
  either way, though not always the same NaN.

## Threads

`ThreadBudget.ForThisProcess` picks the thread count once, and `Kernels` uses it for every
parallel loop:

- one thread per fast core the process may run on, at most `ThreadBudget.Most` (8), so the
  remaining cores stay with the app, the call and audio capture;
- fast cores are the highest efficiency class (P-cores on hybrid CPUs); the hyper-threads of one
  core count once;
- a process pinned to E-cores only sees only those, so they are its fastest cores;
- without core information it falls back to 4 threads (or fewer logical processors).

The core list comes from `GetSystemCpuSetInformation`. Each `SYSTEM_CPU_SET_INFORMATION` entry is
Size (4 bytes), Type (4), then the CpuSet fields Id (4), Group (2), LogicalProcessorIndex,
CoreIndex, LastLevelCacheIndex, NumaNodeIndex and EfficiencyClass (1 byte each); the `CpuSet*Offset`
constants name those offsets. The process affinity mask covers its first processor group only,
as .NET's `ProcessorAffinity` does, so processors in other groups are always counted.

`Kernels.OverrideThreadBudgetForBenchmark` overrides the budget for `parakeet-runtime --cpu-bench --threads N` (and the
corpus benchmark), to measure other budgets. Call it before any model runs. `Kernels.ForRanges`
gives each thread one contiguous run of `[0, n)`, for work that keeps state along the run.

## CPU benchmark

`parakeet-runtime --cpu-bench [--frames 250,1500] [--affinity C03] [--threads N] [--int8-path Portable|Avx2|AvxVnni|AvxVnniInt8]`
times the CPU encoder's kernels alone, per Conformer layer, so a change can be compared in seconds
instead of a corpus run.

- It prints the best of several runs per kernel in ms per layer (for a matrix product, also the
  part spent quantizing its input), their total, and a checksum (FNV-1a over the output bits, the
  same in every process, unlike `HashCode`). Two builds that print the same checksums computed the
  same bits.
- Each kernel's inputs depend only on the kernel and the frame count, so the checksums don't change
  with the order of `--frames`.
- `LayerShapes` lists one Conformer layer's matrix products as the encoder runs them: output
  columns, input width, rows per frame, times per layer, and the output step (SiLU, or added into
  the residual, the feed-forwards' at half weight). No bias, like the encoder's layers.
- The position projection runs only above 251 frames (20 s); shorter utterances read the table
  made at load.
- A residual collects every run's sum, as many runs in every build.
- Buffers come from a scratch, reused run after run as the encoder reuses them layer after layer.
- The total leaves out the layer's smaller kernels (norms, convolution).

## Tests

`tests/ParakeetRuntime.Net.Tests/KernelsTests.cs`:

- Quantizing is checked against the ggml formula on both the AVX2 path and the 128-bit path. The
  first block has amax 127 (scale 1), so its halves land exactly on .5 and test the rounding.
- A loudness sweep covers fp16-subnormal scales up to finite blocks too loud for fp16, with
  all-negative blocks, negative zeros and a block of nothing else.
- Silent blocks and NaN or infinity blocks are quantized into buffers that a louder input just
  filled (`QuantizeOverLouder`), as the encoder reuses them layer after layer: blocks of zeros must
  be written, not left as they were. The bad values sit in every 8-value load and every lane of a
  4-value half.
- Every int8 path the CPU has must give the exact product of the quantized values, with and
  without bias, plain, with SiLU and added to a residual. Weights span the whole int8 range, -128
  included, because the file format allows it. The shapes cover whole 4 x 8 and 4 x 16 tiles
  (with 4 threads, a thread gets two or more groups of 8 columns from 64 columns on) and leftover
  rows and columns. xUnit theories are public and the engine's types are internal, so the paths are
  passed by name.
- Attention is checked against the formula: `T = 7` and `37` leave a partial row tile, `T = 37`
  needs several tiles of key columns, `T = 70` several blocks of query rows, and `dk = 32` is two
  vectors at any width up to 512 bits.

### Pinned bits

`Relative_attention_keeps_its_exact_float_sums` pins the exact bits of the attention output, so
that a faster loop order cannot slip in a reordered float sum: transcripts must stay identical. It
uses `dk = 128` as in the model; `T = 70` and `300` span several blocks of query rows, and
`T = 70` ends with a partial row tile.

The bits depend on 256-bit FMA vectors and on .NET's `Vector.Exp`, so the test runs only where
`Vector<float>` is 256 bits wide and FMA exists (`FmaFact`). Update the pinned values only together
with a corpus transcript comparison (`parakeet-runtime --corpus ... --compare ...`).
