# Benchmarks

This file is the record of every measured speed, accuracy and memory figure for the engine. It
is append-only: a record is never edited after it is written, and new records go at the bottom,
in date order. When a later measurement shows an earlier one was wrong, add a new record that
says so and names the earlier one.

The numbers come from `parakeet-runtime --corpus` (real-time factor, WER, memory on a clip
list), `--cpu-bench` (CPU kernels per encoder layer, with output checksums), `--gemm-bench` (GPU
Q8 GEMM per encoder shape) and `scripts/Run-Matrix.ps1` (the full device × language × length
matrix against NeMo-Speech.cpp). How to run them is in
[benchmarks-howto.md](benchmarks-howto.md).

The records up to 2026-11-23 come from the engine's history in LocalBuddy, where the
benchmark was `LocalBuddy.Spike`; their commits are LocalBuddy commits, and each number is
copied as its commit message or pull request states it. Records from then on name
ParakeetRuntime commits.

RTF is the real-time factor: processing time ÷ audio length, so lower is faster. "Best of N" is
the best pass of N timed passes after one untimed warm-up.

## Adding a record

Append a section in this shape at the bottom of the file:

```markdown
## YYYY-MM-DD · <short title>

- **Machine:** <CPU>, <GPU(s)>, driver <version or "not recorded">, RAM <size or "not recorded">
- **Commit:** ParakeetRuntime `<sha>` (and the baseline's `<sha>` for an A/B)
- **Setup:** model <file>, audio <corpus and clip count>, device <cpu | gpu:<adapter>>,
  affinity <mask> / <threads> threads, <passes>

| | Before | After |
| --- | --- | --- |
| <metric> | <value> | <value> |

<Anything the numbers can't say: noise, caveats, identical transcripts.>
```

Write "not recorded" for anything you don't know, rather than guessing.

## 2026-11-21 · In-process engine against NeMo-Speech.cpp

- **Machine:** Intel Core Ultra 7 270K Plus; driver and RAM not recorded
- **Commit:** LocalBuddy `e25e383`
- **Setup:** Parakeet TDT 0.6B v3 Q8_0, FLEURS dev, CPU, 4 pinned cores; passes not recorded

| | LocalBuddy | NeMo-Speech.cpp |
| --- | --- | --- |
| WER Polish | 5.57% | 5.72% |
| WER English | 5.19% | 5.31% |

- Words differing between the engines: 0.23% (Polish) / 0.34% (English).
- RTF 0.087-0.089; Spike replay of 112 s: RTF 0.095, worst lag 1.2 s.
- Memory 0.82 GB after load, 1.53 GB peak, 1.12 GB after 147 clips.

## 2026-11-21 · Portable 128-bit path for CPUs without AVX2

- **Machine:** not recorded
- **Commit:** LocalBuddy `ff8db07`
- **Setup:** CPU with `DOTNET_EnableAVX2=0`, 4 cores; 11 Polish clips for the transcript check

| | Before (plain loops) | After (128-bit vectors) |
| --- | --- | --- |
| RTF without AVX2/FMA | 1.37 | 0.244 |

Text identical to the AVX2 path on 11 Polish clips.

## 2026-11-21 · GPU slice 1: encoder Q8_0 linear on Direct3D 12

- **Machine:** NVIDIA GeForce RTX 5070 Ti; CPU, driver and RAM not recorded
- **Commit:** LocalBuddy `0e961e8`
- **Setup:** FLEURS dev clips ≤ 20 s (147 Polish, 149 English), GPU, CPU pinned to cores 8-11
  (`0xF00`); passes not recorded

| | Polish (147) | English (149) |
| --- | --- | --- |
| RTF | 0.03206 (31.2x RT) | 0.03193 (31.3x RT) |
| WER | 5.65% (NeMo CPU 5.72%) | 5.25% (NeMo CPU 5.31%) |

Peak RAM 1002-1022 MB, peak VRAM 638 MB.

## 2026-11-21 · GPU slice 2: the whole Conformer encoder on the GPU

- **Machine:** NVIDIA GeForce RTX 5070 Ti; CPU, driver and RAM not recorded
- **Commit:** LocalBuddy `987694f`
- **Setup:** FLEURS dev clips ≤ 20 s (147 Polish, 149 English), GPU, affinity `0xF00`; passes not
  recorded

| | Polish (147) | English (149) |
| --- | --- | --- |
| Encoder time | 32.8s -> 4.33s (7.6x faster) | 35.3s -> 4.58s (7.7x faster) |
| RTF | 0.01104 (90.6x RT) | 0.01068 (93.7x RT) |
| WER | 5.65% (NeMo CPU 5.72%) | 5.25% (NeMo CPU 5.31%, Vulkan 5.44%) |

Peak RAM 1079-1093 MB, peak VRAM 676 MB.

## 2026-11-21 · GPU slice 3A: subsampling on the GPU, precomputed mel twiddles

- **Machine:** NVIDIA GeForce RTX 5070 Ti; CPU, driver and RAM not recorded
- **Commit:** LocalBuddy `2266135`
- **Setup:** FLEURS dev clips ≤ 20 s (147 Polish, 149 English), GPU, affinity `0xF00`; passes not
  recorded

| | Polish (147) | English (149) |
| --- | --- | --- |
| Subsampling + encoder | 10.21s -> 4.44s | 11.01s -> 4.73s |
| Mel | 1.28s -> 0.40s (3.2x faster) | 1.35s -> 0.42s |
| RTF | 0.00593 (168.7x RT) | 0.00565 (177.1x RT) |
| WER | 5.61% (NeMo CPU 5.72%) | 5.25% (NeMo CPU 5.31%, Vulkan 5.44%) |

Peak RAM 1081-1094 MB, peak VRAM 762 MB.

## 2026-11-21 · CPU baseline of the corpus benchmark against NeMo-Speech.cpp

- **Machine:** Intel Core Ultra 7 270K Plus; driver and RAM not recorded
- **Commit:** LocalBuddy `f93c997`
- **Setup:** FLEURS dev clips ≤ 20 s (147 Polish, 149 English), CPU pinned to cores 4-7 (`0xF0`),
  both engines on the same model and cores; one untimed warm-up, passes not recorded

| | NeMo-Speech.cpp | LocalBuddy |
| --- | --- | --- |
| WER Polish (147) | 5.72% | 5.57% |
| WER English (149) | 5.31% | 5.19% |
| RTF Polish / English | 0.148 / 0.140-0.144 | 0.088 / 0.089 |
| RTF without AVX2, Polish | - | 0.231 |
| Peak memory | 897-910 MB | 1.55 GB |

LocalBuddy's time: feed-forward 55%, attention 24%, convolution 12%, subsampling 5%; int8 matrix
products are 71% of it.

## 2026-11-21 · GPU slice 3C: TDT decoder loop on the GPU

- **Machine:** NVIDIA GeForce RTX 5070 Ti; CPU, driver and RAM not recorded
- **Commit:** LocalBuddy `b529e41`
- **Setup:** FLEURS dev clips ≤ 20 s (147 Polish, 149 English), GPU, affinity `0xF00`; passes not
  recorded

| | Polish (147) | English (149) |
| --- | --- | --- |
| RTF | 0.00410 (243.9x RT, 5.44s) | 0.00396 (252.4x RT, 5.68s) |
| NeMo-Speech.cpp Vulkan RTF | 0.00966 (2.36x slower) | 0.00843 (2.13x slower) |
| WER | 5.61% | 5.25% |

Peak RAM 1097-1130 MB, peak VRAM 797 MB.

## 2026-11-21 · CPU: int8 weights packed 8 rows at a time

- **Machine:** Intel Core Ultra 7 270K Plus; driver and RAM not recorded
- **Commit:** LocalBuddy `1c1aa4f`
- **Setup:** FLEURS dev clips ≤ 20 s, CPU pinned to cores 4-7; passes not recorded

| | Before | After |
| --- | --- | --- |
| RTF English, best path (AVX-VNNI-INT8) | 0.0886 | 0.0466 |
| RTF Polish, best path | 0.0882 | 0.0461 |
| RTF Polish, AVX-VNNI path | 0.0882 | 0.0505 |
| RTF Polish, AVX2-only path | 0.0884 | 0.0819 |
| int8 product time, English | 89.5 s | 29.6 s |
| WER Polish | 5.57% | 5.57% (same 144 errors) |
| WER English | 5.19% | 5.31% (NeMo-Speech.cpp: 5.31%) |
| English words differing from NeMo-Speech.cpp | 0.34% | 0.25% |
| Peak memory | 1.55 GB | 1.61 GB |

## 2026-11-21 · CPU: activations quantized with vectors

- **Machine:** not recorded (same pin, cores 4-7, as the `f93c997` baseline)
- **Commit:** LocalBuddy `48596a7`
- **Setup:** English corpus (FLEURS dev ≤ 20 s), CPU pinned to cores 4-7; passes not recorded

| | Before | After |
| --- | --- | --- |
| RTF English | 0.0466 | 0.0423 |

Transcripts byte-identical to the previous commit (WER 5.31%).

## 2026-11-21 · GPU slice 4: 32x32 float4 Q8 GEMM, float4 attention, streaming VRAM load

- **Machine:** NVIDIA GeForce RTX 5070 Ti; CPU, driver and RAM not recorded
- **Commit:** LocalBuddy `4bde618`
- **Setup:** FLEURS dev clips ≤ 20 s (147 Polish, 149 English), GPU, affinity `0xF00`; passes not
  recorded

| | Polish (147) | English (149) |
| --- | --- | --- |
| RTF | 0.00328 (305.3x RT, 4.34s) | 0.00313 (319.8x RT, 4.48s) |
| Speed-up over NeMo Vulkan | 2.95x | 2.70x |
| WER | 5.61% | 5.25% |

Load time 3.72s -> 1.79-1.87s. Peak RAM 1077-1092 MB, peak VRAM 797 MB.

## 2026-11-21 · CPU: SiLU and residual fused into the int8 store, rest of a layer vectorized

- **Machine:** not recorded (same pin, cores 4-7, as the `f93c997` baseline)
- **Commit:** LocalBuddy `3526f82`
- **Setup:** FLEURS dev clips ≤ 20 s, CPU pinned to cores 4-7; passes not recorded

| | Before | After |
| --- | --- | --- |
| RTF English | 0.0423 | 0.0324 |
| RTF Polish | 0.046 | 0.0325 |
| RTF Polish without AVX2 | 0.2314 | 0.2237 |
| Layer norm (output norm stage), English | 0.82 s | 0.08 s |
| WER English | 5.31% | 5.31% (NeMo-Speech.cpp 5.31%) |
| WER Polish | 5.57% | 5.68% (NeMo-Speech.cpp 5.72%) |

The Polish change is 6 borderline clips, 2 better and 4 worse. Without AVX2: WER Polish 5.65%.

## 2026-11-21 · CPU: model load without a memory map or double copies

- **Machine:** not recorded (same pin, cores 4-7, as the `f93c997` baseline)
- **Commit:** LocalBuddy `83140dc`
- **Setup:** English corpus (FLEURS dev ≤ 20 s), CPU pinned to cores 4-7; passes not recorded

| | Before | After |
| --- | --- | --- |
| Peak during load | 1,534 MB | 794 MB |
| Peak over the whole corpus | 1,621 MB (with corpus audio) | 1,220 MB (engine only) |

After load 793 MB. RTF 0.0324, transcripts byte-identical.

## 2026-11-21 · CPU: vectorized subsampling

- **Machine:** not recorded (same pin, cores 4-7, as the `f93c997` baseline)
- **Commit:** LocalBuddy `5d42885`
- **Setup:** FLEURS dev clips ≤ 20 s, CPU pinned to cores 4-7; passes not recorded

| | Before | After |
| --- | --- | --- |
| Subsampling, English | 6.0 s | 1.64 s |
| RTF English | 0.0324 | 0.0296 |
| RTF Polish | 0.0325 | 0.0294 |
| RTF Polish without AVX2 | 0.2237 | 0.2200 |
| Peak memory, English | 1,220 MB | 1,030 MB |
| WER English | 5.31% | 5.31% |
| WER Polish | 5.68% | 5.61% (8 clips flipped, 5 better, 3 worse) |
| WER Polish without AVX2 | 5.65% | 5.76% (NeMo-Speech.cpp 5.72%) |

## 2026-11-21 · GPU slice 5: log-mel spectrogram on the GPU

- **Machine:** NVIDIA GeForce RTX 5070 Ti; CPU, driver and RAM not recorded
- **Commit:** LocalBuddy `32bdee4`
- **Setup:** FLEURS dev clips ≤ 20 s (147 Polish, 149 English), affinity `0xF00`; passes not
  recorded

| | Polish (147) | English (149) |
| --- | --- | --- |
| GPU RTF | 0.00292 (343.0x RT, 3.86s) | 0.00281 (355.8x RT, 4.03s) |
| Speed-up over NeMo Vulkan | 3.31x | 3.00x |
| GPU WER | 5.61% | 5.25% |
| GPU disabled (CPU) RTF | 0.08577 (11.7x RT) | 0.08052 (12.4x RT) |
| GPU disabled (CPU) WER | 5.72% | 5.31% |

GPU memory: 1072-1078 MB peak RAM (392-398 MB steady), 706 MB load / 798 MB peak VRAM.

## 2026-11-21 · CPU: attention position table projected once at load

- **Machine:** not recorded (same pin, cores 4-7, as the `f93c997` baseline)
- **Commit:** LocalBuddy `3462be1`
- **Setup:** English corpus (FLEURS dev ≤ 20 s), CPU pinned to cores 4-7; passes not recorded

| | Before | After |
| --- | --- | --- |
| Attention | 13.2 s | 10.3 s |
| RTF | 0.0296 | 0.0275 |
| Peak memory | 1,030 MB | 1,093 MB |

Transcripts byte-identical. Memory after load +57 MB (2 MB per layer).

## 2026-11-21 · CPU: relative-position attention in register tiles

- **Machine:** not recorded (same pin, cores 4-7, as the `f93c997` baseline)
- **Commit:** LocalBuddy `586b696`
- **Setup:** English corpus (FLEURS dev ≤ 20 s), CPU pinned to cores 4-7; passes not recorded

| | Before | After |
| --- | --- | --- |
| Attention | 10.3 s | 7.15 s |
| RTF | 0.0275 | 0.0250 |
| Peak memory | 1,093 MB | 1,122 MB |
| WER | 5.31% | 5.38% (173 errors, NeMo-Speech.cpp 171) |
| Words differing from NeMo-Speech.cpp | 0.31% | 0.19% |

15 clips flipped, 6 better and 6 worse, on the same knife-edge words as before.

## 2026-11-21 · CPU: activation quantizing with NeMo-Speech.cpp's x86 rounding

- **Machine:** not recorded (same pin, cores 4-7, as the `f93c997` baseline)
- **Commit:** LocalBuddy `9fa03a6`
- **Setup:** FLEURS dev clips ≤ 20 s, CPU pinned to cores 4-7; passes not recorded

| | Previous commit | This commit |
| --- | --- | --- |
| RTF English / Polish | 0.0250 / - | 0.0247 / 0.0250 |
| WER English | 5.38% | 5.31% (NeMo-Speech.cpp 5.31%) |
| WER Polish | - | 5.76% (NeMo-Speech.cpp 5.72%; 5.61% two commits ago) |
| Words differing from the council's fp32 ORT transcripts, English | 19 | 16 (NeMo-Speech.cpp 17) |
| Same, Polish | - | 8 (NeMo-Speech.cpp 7) |

## 2026-11-21 · CPU: precomputed FFT twiddles, filterbank zero bins skipped

- **Machine:** not recorded (same pin, cores 4-7, as the `f93c997` baseline)
- **Commit:** LocalBuddy `ab9f707`
- **Setup:** English corpus (FLEURS dev ≤ 20 s), CPU pinned to cores 4-7; passes not recorded

| | Before | After |
| --- | --- | --- |
| Mel | 1.44 s | 0.34 s |
| RTF | 0.0247 | 0.0243 |

Transcripts byte-identical.

## 2026-11-21 · CPU: int8 tile of 3 activation rows by 16 weight rows

- **Machine:** not recorded (same pin, cores 4-7, as the `f93c997` baseline)
- **Commit:** LocalBuddy `c3f392e`
- **Setup:** English corpus (FLEURS dev ≤ 20 s), CPU pinned to cores 4-7; passes not recorded

| | Before | After |
| --- | --- | --- |
| RTF, AVX-VNNI-INT8 (this CPU) | 0.0243 | 0.0215 |
| RTF, AVX-VNNI path | 0.0293 | 0.0267 |
| RTF, AVX2-only path | 0.0574 | 0.0551 |
| Feed-forward, AVX-VNNI-INT8 | 19.9 s | 17.2 s |

Transcripts byte-identical.

## 2026-11-21 · CPU: int8 tile widened to 4 rows by 16 outputs

- **Machine:** not recorded (same pin, cores 4-7, as the `f93c997` baseline)
- **Commit:** LocalBuddy `9e17be9`
- **Setup:** English corpus (FLEURS dev ≤ 20 s), CPU pinned to cores 4-7; passes not recorded

| | 3 x 16 | 4 x 16 |
| --- | --- | --- |
| RTF, AVX-VNNI-INT8 (this CPU) | 0.0215 | 0.0208, repeat 0.0207 |
| RTF, AVX2-only path | 0.0551 | 0.0521 |
| Feed-forward, AVX-VNNI-INT8 | 17.2 s | 16.3-16.5 s |

Transcripts byte-identical.

## 2026-11-21 · CPU engine against NeMo-Speech.cpp, one session

- **Machine:** not recorded ("this desktop", same pin, cores 4-7, as the `f93c997` baseline)
- **Commit:** LocalBuddy `416143e`
- **Setup:** FLEURS dev clips ≤ 20 s, both engines measured in one session on cores 4-7 with the
  corpus benchmark; passes not recorded

| | NeMo-Speech.cpp | LocalBuddy |
| --- | --- | --- |
| RTF Polish / English | 0.148 / 0.140-0.144 | 0.0211 / 0.0207 |
| WER Polish | 5.72% (148) | 5.76% (149) |
| WER English | 5.31% (171) | 5.31% (171) |
| Words from fp32, Polish / English | 7 / 17 | 8 / 16 |
| Peak memory | 897-910 MB | 1.13 GB |
| No AVX2: RTF, WER Polish / English | - | 0.2007, 5.65% / 5.34% |

Depthwise taps through the shared transpose helper: English transcripts byte-identical, RTF
0.0211.

## 2026-11-21 · GPU backend hardened for several GPUs

- **Machine:** NVIDIA GeForce RTX 5070 Ti and Intel(R) Graphics; CPU, driver and RAM not recorded
- **Commit:** LocalBuddy `4e23b1a`
- **Setup:** FLEURS dev clips ≤ 20 s (147 Polish, 149 English), GPU; affinity and passes not
  recorded

| | Polish (147) | English (149) |
| --- | --- | --- |
| RTX 5070 Ti RTF | 0.00292 (342x RT) | 0.00282 (355x RT) |
| Intel(R) Graphics RTF | 0.04216 (23.7x RT) | 0.04160 (24.0x RT) |
| WER (both GPUs) | 5.61% | 5.25% |

296/296 identical transcripts between the GPUs. Intel iGPU long clips: 18 s, 36 s, 72 s (1/1
exact against CPU) and 145 s complete with no hangs or TDRs.

## 2026-11-22 · CPU kernels hardened after review; PR #2 summary

- **Machine:** not recorded (same pin, cores 4-7, as the `f93c997` baseline)
- **Commit:** LocalBuddy `bb3493c` (head of PR #2, perf/engine-cpu)
- **Setup:** FLEURS dev clips ≤ 20 s, CPU pinned to cores 4-7; NeMo-Speech.cpp `serve --device
  cpu` on the same model and cores; passes not recorded

From the commit message (English corpus):

| | Before | After |
| --- | --- | --- |
| RTF, AVX2 product path | 0.0521 | 0.0485 |
| RTF, default path | - | 0.0206 (WER 5.31%) |

Transcripts identical (149 on the AVX2 path).

From the PR #2 description, the branch's result against NeMo-Speech.cpp:

| | NeMo-Speech.cpp | LocalBuddy before | LocalBuddy now |
| --- | --- | --- | --- |
| RTF PL / EN | 0.148 / 0.142 | 1.7× / 1.6× faster | 0.0211 / 0.0206 (7.0× / 6.9× faster) |
| WER PL / EN | 5.72% / 5.31% | −0.15 / −0.12 | +0.04 / ±0.00 |
| Peak memory | 0.91 GB | +0.64 GB | +0.22 GB (1.13 GB) |
| AVX-VNNI path (EN) | | | 0.0254 (5.6×) |
| AVX2-only path (EN) | | | 0.0485 (2.9×) |
| No AVX2 (128-bit) | | | 0.2007 / 0.2006, WER 5.65% / 5.34% |

## 2026-11-22 · GPU backend merged with the CPU engine; PR #1 summary

- **Machine:** Intel Core Ultra 7 270K Plus with NVIDIA GeForce RTX 5070 Ti (16 GB) and Intel(R)
  Graphics (integrated); Windows 11 26200; driver and RAM not recorded
- **Commit:** LocalBuddy `51e4b07` (head of PR #1, perf/engine-gpu)
- **Setup:** FLEURS dev clips ≤ 20 s (147 Polish, 149 English), cores 4-7 (`0xF0`); CPU on 4
  cores, AVX-VNNI-INT8; passes not recorded

| Backend | PL RTF (× RT) | EN RTF (× RT) | PL / EN WER | Steady / peak RAM | Peak VRAM |
| --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 0.00309 (323×) | 0.00297 (336×) | 5.61% / 5.25% | 411 MB / 414 MB | 798 MB |
| Intel(R) Graphics | 0.04213 (23.7×) | 0.04157 (24.1×) | 5.61% / 5.25% | 1.08 GB / 1.15 GB | 800 MB |
| LocalBuddy CPU | 0.0210 (47.6×) | 0.0206 (48.5×) | 5.76% / 5.31% | 850 MB / 1.13 GB | — |
| NeMo-Speech.cpp v0.2.0 `--device vulkan` (RTX 5070 Ti) | 0.00966 (103×) | 0.00843 (119×) | 5.53% / 5.44% | 122 MB / 774 MB | 941 MB |

- GPU peak RAM with the `RandomAccess` GGUF reader: 1.07 GB -> 414 MB on the RTX 5070 Ti,
  1.74 GB -> 1.15 GB on the Intel iGPU.
- CPU details: EN WER 5.31%, 0.34% vs `en-nemo.tsv` (11/3196 differ, 136/149 identical), peak
  1,093 MB; PL WER 5.76%, 0.27% vs `pl-nemo.tsv` (7/2577 differ, 139/147 identical), peak
  1,125 MB.
- 296/296 identical transcripts between the RTX 5070 Ti and the Intel iGPU.

## 2026-11-22 · One corpus benchmark for both engines on every device: smoke check

- **Machine:** NVIDIA GeForce RTX 5070 Ti and Intel iGPU; CPU, driver and RAM not recorded
- **Commit:** LocalBuddy `4670f47`
- **Setup:** first 5 English FLEURS clips, cores 4-7; passes not recorded

| Engine / device | RTF | GPU memory |
| --- | --- | --- |
| LocalBuddy CPU | 0.0294 | |
| LocalBuddy RTX | 0.0034 | 806 MB dedicated |
| LocalBuddy Intel | 0.0425 | 806 MB shared |
| NeMo-Speech.cpp CPU | 0.1246 | |
| NeMo-Speech.cpp Vulkan RTX | 0.0079 | 838 MB dedicated |

Text identical to `en-nemo.tsv` in every run except NeMo Vulkan (4/5, as before).

## 2026-11-22 · First full matrix against NeMo-Speech.cpp

- **Machine:** Intel Core Ultra 7 270K Plus with NVIDIA GeForce RTX 5070 Ti and Intel iGPU;
  driver and RAM not recorded
- **Commit:** not recorded; it ran after LocalBuddy `4670f47` (which added `Run-Matrix.ps1`) and
  before `c2c2769` (which compares against it)
- **Setup:** FLEURS Polish and English at ≤ 20 s, 60 s and 2 min; pin F0 (4 E-cores, see the
  `bf1b4a5` record); best of 2 passes. Source: PR #3's "Matrix before the speed-ups"

| Lang | Length | Device | NeMo RTF | LocalBuddy RTF | ×NeMo | WER NeMo / LB |
| --- | --- | --- | --- | --- | --- | --- |
| PL | ≤ 20 s | RTX / Intel / CPU | 0.00601 / 0.06982 / 0.14050 | 0.00300 / 0.04211 / 0.02068 | 2.00 / 1.66 / 6.79× | 5.53 / 5.61% (GPU), 5.72 / 5.76% (CPU) |
| PL | 60 s | RTX / Intel / CPU | 0.00558 / 0.05980 / 0.14840 | 0.00262 / 0.04453 / 0.02960 | 2.13 / 1.34 / 5.01× | 8.75–8.98% / 8.83–8.86% |
| PL | 2 min | RTX / Intel / CPU | 0.00564 / 0.06338 / 0.16572 | 0.00311 / 0.05475 / 0.06378* | 1.81 / 1.16 / 2.60×* | 10.08–10.24% / 9.84–9.92% |
| EN | ≤ 20 s | RTX / Intel / CPU | 0.00549 / 0.06490 / 0.14157 | 0.00292 / 0.04156 / 0.02045 | 1.88 / 1.56 / 6.92× | 5.31–5.44% / 5.25–5.31% |
| EN | 60 s | RTX / Intel / CPU | 0.00422 / 0.05410 / 0.14964 | 0.00252 / 0.04363 / 0.02902 | 1.67 / 1.24 / 5.16× | about 31%, model drops sentences after pauses |
| EN | 2 min | RTX / Intel / CPU | 0.00436 / 0.05729 / 0.16052 | 0.00302 / 0.05434 / 0.04416 | 1.44 / 1.05 / 3.63× | about 43%, same |

\* The PL 2-minute CPU run was disturbed by other load: a quiet re-check of 3 of those clips gave
0.044.

## 2026-11-22 · Quick A/B subset for Run-Matrix

- **Machine:** not recorded (CPU, Intel and RTX devices, as in the first matrix)
- **Commit:** LocalBuddy `c2c2769`
- **Setup:** Polish, 3 × 2 min + 30 clips ≤ 20 s, best of 3 passes; affinity not recorded
  (the matrix's pin then was F0)

| | CPU | Intel | RTX |
| --- | --- | --- | --- |
| 2 min | 0.0439 | 0.0543 | 0.00308 |
| ≤ 20 s | 0.0209 | 0.0423 | 0.00305 |

3.5 min for the set; passes within 1%; identical text to the matrix in every run.

## 2026-11-22 · Memory given back when a session ends

- **Machine:** not recorded
- **Commit:** LocalBuddy `6a0132d`
- **Setup:** `Spike --session-memory`: load, transcribe, dispose, engine still referenced, no
  forced GC from the caller; 3 sessions on CPU and on GPU

| | Before | After |
| --- | --- | --- |
| CPU, private memory after sessions 1 / 2 / 3 | 979 MB / 1.8 GB / 2.57 GB | 26 MB after every session |
| CPU Dispose | | 51-54 ms |
| GPU, 10 s after stop | 102-156 MB private, GC heap 76-80 MB kept | 27-29 MB private, GC heap 3 MB |
| GPU Dispose | | 32-37 ms |

## 2026-11-22 · CPU attention: 32 query rows per task

- **Machine:** not recorded
- **Commit:** LocalBuddy `c12c004`
- **Setup:** `pl_pl/long120` (2-minute Polish clips), `--limit 2`, `--affinity F0`; passes not
  recorded

| | Before | After |
| --- | --- | --- |
| Attention | 6.72 s | 4.24 s |
| RTF | 0.0444 | 0.0342 |
| Peak RAM | 1584 MB | 1425 MB |

Identical text 2/2.

## 2026-11-22 · GC pauses per corpus pass

- **Machine:** not recorded
- **Commit:** LocalBuddy `2f48b9e`
- **Setup:** long corpus passes; 2 `long120` clips for the GC check

Long-corpus passes differed by up to 40% (60 s: RTF 0.0296 vs 0.0420). GC pause time on 2
`long120` clips was 0.01 s, so garbage collection does not explain the gap.

## 2026-11-22 · CPU thread budgets and core pins

- **Machine:** not recorded (24 cores; P-cores are CPUs 0, 1, 10-13, 22, 23; mask F0 is E-cores)
- **Commit:** LocalBuddy `bf1b4a5`
- **Setup:** Polish subsets (3 × 2 min, 30 × ≤ 20 s), CPU, best of 2 passes; transcripts
  identical at every budget

| Pin / threads | RTF ≤ 20 s | RTF 2 min |
| --- | --- | --- |
| 4 E-cores, mask F0 (the benchmark's pin) | 0.0214 | 0.0391 |
| 4 E-cores, one cluster, 3C | 0.0222 | 0.0355 |
| no pin, 4 threads (the app then) | 0.0233 | 0.0310 |
| 4 P-cores, C03 | 0.0144 | 0.0257 |
| 8 P-cores, C03C03, 8 threads | 0.0109 | 0.0155 |
| all 24 cores, 8 threads | 0.0155 | 0.0155 |
| all 24 cores, 16 threads | 0.0140 | 0.0120 |

## 2026-11-22 · CPU: one thread per fast core, up to 8

- **Machine:** not recorded (same hybrid CPU as the `bf1b4a5` record)
- **Commit:** LocalBuddy `8122b32`
- **Setup:** Polish subsets (≤ 20 s and 2 min), CPU, best of 3 passes; transcripts identical
  everywhere

| Pin | Threads | RTF ≤ 20 s | RTF 2 min |
| --- | --- | --- | --- |
| no pin (the app), before | 4 | 0.0233 | 0.0310 |
| no pin (the app), now | 8 | 0.0124 | 0.0153 |
| C03 | 4 | 0.0145 | 0.0259 |
| F0 | 4 | 0.0207 | 0.0335 |

Unpinned passes vary 0.0124-0.0160. Benchmarks pin C03 (4 P-cores) from this commit on.

## 2026-11-22 · GPU attention tiled into 32x32 FlashAttention blocks

- **Machine:** Intel iGPU (and RTX 5070 Ti for the tests); CPU, driver and RAM not recorded
- **Commit:** LocalBuddy `4fbee57`
- **Setup:** 2 clips of `pl_pl/long120`, Intel iGPU; affinity and passes not recorded

| | Before | After |
| --- | --- | --- |
| Intel 2-clip RTF | 0.05502 | 0.04373 |
| Encoder | 12.80 s | 10.04 s |

0.00% difference, 2/2 identical text.

## 2026-11-22 · GPU decoder dispatches fused

- **Machine:** Intel iGPU and RTX 5070 Ti; CPU, driver and RAM not recorded
- **Commit:** LocalBuddy `1ed2dee`
- **Setup:** 2 clips of `pl_pl/long120`; affinity and passes not recorded

| | Intel | RTX |
| --- | --- | --- |
| 2-clip RTF | 0.04309 | 0.00219 |

0.00% difference, 2/2 identical text.

## 2026-11-22 · GPU decoder batches bounded by a time budget

- **Machine:** Intel Core Ultra 7 270K Plus with NVIDIA GeForce RTX 5070 Ti and Intel iGPU;
  driver and RAM not recorded
- **Commit:** LocalBuddy `3022ad7`
- **Setup:** the matrix's clips (≤ 20 s, 60 s, 2 min), GPU; pin and passes not recorded

| | ≤ 20 s | 60 s | 2 min | Identical transcripts |
| --- | --- | --- | --- | --- |
| RTX PL | 0.00277 | 0.00203 | 0.00211 | 147, 22, 11 |
| RTX EN | 0.00267 | 0.00198 | 0.00204 | 149, 23, 12 |
| Intel PL | 0.04136 | 0.03962 | 0.04298 | 147, 22, 11 |

Speed unchanged by the bound; transcripts identical to the matrix.

## 2026-11-22 · GPU Q8 GEMM: 64x64 tile above 96 rows

- **Machine:** Intel Core Ultra 7 270K Plus with NVIDIA GeForce RTX 5070 Ti and Intel(R)
  Graphics; driver and RAM not recorded
- **Commit:** LocalBuddy `b891d67`, against the build before it
- **Setup:** `Spike --gemm-bench` per encoder layer; corpus best of 3, pin C03, old build against
  new

GEMM bench, new kernel against the old:

| Frames | 25 | 96 | 250 | 1500 |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | 1.09x | 1.01x | 1.38x | 1.54x |
| Intel Graphics | 1.05x | 1.11x | 1.66x | 1.61x |

Corpus RTF (from PR #3, which gives every value the commit message abbreviates):

| Device | PL ≤ 20 s | PL 60 s | PL 2 min | EN 60 s | EN 2 min |
| --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 0.00273 → 0.00256 (1.07×) | 0.00200 → 0.00145 (1.38×) | 0.00207 → 0.00154 (1.34×) | 0.00192 → 0.00135 (1.42×) | 0.00202 → 0.00147 (1.37×) |
| Intel iGPU | 0.04152 → 0.03056 (1.36×) | 0.03929 → 0.02745 (1.43×) | 0.04270 → 0.03057 (1.40×) | 0.03857 → 0.02663 (1.45×) | 0.04224 → 0.03020 (1.40×) |

- Transcripts identical on every clip; WER and VRAM unchanged.
- Model load 2.0 s -> 2.7 s (unpinned): FXC takes 2.5 s on the unrolled 64x64 kernel (0.36 s
  before).
- The commit's "Intel small kernel 0.94x -> 1.05x" is left out: `fc6b30d` says the 0.94x figure
  came from a per-row bounds check that was tried and dropped, not from the old kernel.

## 2026-11-22 · GEMM review fixes

- **Machine:** not recorded
- **Commit:** LocalBuddy `fc6b30d`
- **Setup:** GPU model load, unpinned

Load time unchanged: 2.73 s unpinned (2.0 s before `b891d67`). The GEMM bench now counts outputs
a variant never writes: a 128 tile on a 64 kernel reports 2,475,008 outputs not written, where it
had shown a false 1.92x win.

## 2026-11-23 · CPU: 3x faster activation quantizing, faster attention

- **Machine:** Intel Core Ultra 7 270K Plus (the main machine, per PR #3); driver and RAM not
  recorded
- **Commit:** LocalBuddy `95c8512`, against its parent
- **Setup:** CPU, 4 P-cores, affinity C03; `Spike --cpu-bench` per layer; corpus old build against
  new, 4 alternating rounds for ≤ 20 s

Kernel bench per layer:

| | Before | After |
| --- | --- | --- |
| T=250 | 14.00 ms | 11.44 ms (1.22x) |
| T=1500, before the attention changes | 102.5 ms | 96.9 ms |
| Attention alone, T=250 | 3.0 ms | 2.8 ms |
| Attention alone, T=1500 | 47.4 ms | 39.9 ms |
| Quantize, 8000 blocks, one thread | 565 us | 184-201 us |
| Attention per layer at 2 min, softmax max only | 47.3 ms | 45.0 ms |

Corpus RTF:

| | Before | After |
| --- | --- | --- |
| PL ≤ 20 s (30 clips), best | 0.01592 | 0.01378 (1.16x) |
| PL ≤ 20 s (30 clips), median | 0.01657 | 0.01391 (1.19x) |
| PL 2 min (3 clips) | 0.02629 | 0.02496 (1.05x) |
| EN 2 min (3 clips) | 0.02661 | 0.02529 (1.05x) |

Transcripts identical on every clip (30/30, 3/3, 3/3); every kernel's checksum unchanged. The
kernel bench ran with default tiering; see the `ca13477` record for what that means for its
quantize times.

## 2026-11-23 · Quantize review fixes

- **Machine:** Intel Core Ultra 7 270K Plus (the main machine, per PR #3); driver and RAM not
  recorded
- **Commit:** LocalBuddy `304b764`, against `95c8512`
- **Setup:** `Spike --cpu-bench`, 4 P-cores, 3 alternating rounds; default tiering

| | `95c8512` | `304b764` |
| --- | --- | --- |
| Quantize, ff in, T=250 (best) | 0.20 ms | 0.23 ms |
| Quantize, ff out, T=1500 (best) | 0.87 ms | 0.93 ms |

Within run-to-run noise; all 14 kernel checksums identical in every run. `ca13477` later found
that tiered bench runs time Tier0 code for Quantize, so these times are Tier0 times.

## 2026-11-23 · CPU bench with tiering off

- **Machine:** Intel Core Ultra 7 270K Plus (the main machine, per PR #3); driver and RAM not
  recorded
- **Commit:** LocalBuddy `ca13477`, against `95c8512`'s kernels
- **Setup:** `Spike --cpu-bench`, 4 P-cores, 3 alternating rounds, `DOTNET_TieredCompilation=0`

| | `95c8512` | `ca13477` |
| --- | --- | --- |
| Quantize, ff in, T=250 | 0.04-0.06 ms | 0.04-0.06 ms |
| Quantize, ff out, T=1500 | 0.86-0.88 ms | 0.87-0.90 ms |

T=250 kernels total: 13-14 ms with default tiering, 9.5 ms with tiering off.

All 13 kernel checksums identical in every run. With default tiering, Quantize stays at
instrumented Tier0 for the whole bench, so a tiered A/B showed a false 40% quantize slowdown; it
is not a result.

## 2026-11-23 · GPU: 64x64 GEMM compiled in the background; load time

- **Machine:** Intel Core Ultra 7 270K Plus with NVIDIA GeForce RTX 5070 Ti and Intel iGPU;
  driver and RAM not recorded
- **Commit:** LocalBuddy `953f41c` and its review fixes `19d2905`, against `3022ad7` (before the
  64x64 tile) and `ca13477`
- **Setup:** GPU model load, unpinned like the app, 3 rounds each; transcripts on 40 Polish and
  40 English clips up to 20 s per GPU

| Load time | `3022ad7` | `ca13477` | `953f41c` | `19d2905` |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | 2.01-2.02 s | 2.74-2.75 s | 1.90-1.92 s | 1.93-1.94 s |
| Intel iGPU | 1.99-2.00 s | 2.69-2.73 s | 2.30 s | 2.29-2.31 s |

- Starting the compile after the other pipelines left Intel at 2.29-2.30 s, so its remaining
  0.3 s is not this compile competing for the CPU.
- Transcripts identical to `ca13477`'s in 160/160, including clips that ran while the 64x64
  kernel still compiled.

## 2026-11-23 · Final matrix against NeMo-Speech.cpp

- **Machine:** Intel Core Ultra 7 270K Plus with NVIDIA GeForce RTX 5070 Ti and Intel iGPU;
  driver and RAM not recorded
- **Commit:** reported at LocalBuddy `19d2905` (head of PR #3); the exact build of each column is
  not recorded, but the CPU column ran before `95c8512`
- **Setup:** FLEURS Polish and English at ≤ 20 s, 60 s and 2 min; pin C03 (4 P-cores); passes not
  stated (`Run-Matrix.ps1` kept the best of 3 from `c2c2769` on). Source: PR #3's final matrix

LocalBuddy RTF, and in brackets NeMo-Speech.cpp RTF ÷ LocalBuddy RTF on the same device:

| | RTX 5070 Ti | Intel iGPU | CPU |
| --- | --- | --- | --- |
| PL ≤ 20 s | 0.00260 (2.3×) | 0.03054 (2.2×) | 0.01505 (4.2×) |
| PL 60 s | 0.00148 (3.1×) | 0.02745 (2.1×) | 0.02223 (4.1×) |
| PL 2 min | 0.00156 (3.1×) | 0.03076 (2.0×) | 0.02614 (3.9×) |
| EN ≤ 20 s | 0.00245 (2.1×) | 0.02971 (2.2×) | 0.01467 (4.3×) |
| EN 60 s | 0.00141 (2.6×) | 0.02684 (2.0×) | 0.02269 (4.2×) |
| EN 2 min | 0.00149 (2.5×) | 0.03002 (1.9×) | 0.02589 (4.2×) |

- WER on par with NeMo everywhere (within about 0.1–0.2 points).
- Model load: LocalBuddy CPU about 1.0 s; NeMo 0.55–1.6 s. GPU load is in the previous record.
- GPU transcripts identical to the matrix on every clip: RTX 364/364, Intel 180/180.

## 2026-11-23 · Figures that were kept in the engine's code comments

- **Machine:** as listed per row. "Inferred" means the source names only the core pin, not the CPU.
- **Commit:** LocalBuddy `19d2905`, the state that moved here. Each row names the commit that
  wrote the comment, found with `git blame`.
- **Setup:** as listed per row. These figures sat in code comments rather than in commit
  messages. They moved here when the comments were removed.

| Figure | Value | Source commit | Machine |
| --- | --- | --- | --- |
| Quantize when .NET `Half` conversion runs while 256-bit registers are live | about 4x slower | `95c8512` | Core Ultra 7 270K Plus |
| Attention keys in runs of key-column pairs instead of rows a width apart | score phase 11% faster at 2 min | `95c8512` | Core Ultra 7 270K Plus (inferred), affinity C03 |
| Attention prep head by head, without two integer divisions by dk per element | attention 10% faster at 20 s, 5% at 2 min | `95c8512` | Core Ultra 7 270K Plus (inferred), affinity C03 |
| One attention head's data per layer at 2 min | 2.5 MB (L2 is 2 MB per E-core, 3 MB per P-core) | `c12c004` | Core Ultra 7 270K Plus (inferred), affinity F0 |
| Block scores, 32 × T floats at 2 min | 210 KB | `c12c004` | any |
| Position table per layer at 2 min | 13 MB | `c12c004` | any |
| Positional projections computed at load for 251 frames (20 s) | 2 MB per layer | `3462be1` | any |
| Output of subsampling conv 0 for one 20 s utterance (no longer built on the CPU) | 65 MB | `5d42885` | any |
| CPU weights held as managed arrays | about 860 MB | `6a0132d` | any |
| Thread budget: 8 threads on 8 P-cores against the earlier fixed 4 threads | about 2x faster | `8122b32` | Core Ultra 7 270K Plus |
| Small Q8 GEMM tile, one bounds check per row instead of per four rows, 25 frames | RTX 1.26x (1.09x with the four-row check); Intel 0.94x (1.05x) | `fc6b30d` | RTX 5070 Ti, Intel Graphics |
| Partial unrolling of the GEMM k-loop | Intel 11% slower | `b891d67` | Intel Graphics |
| 64 × 64 tile slower than 32 × 32 up to 96 rows, faster from 128 rows | 96 rows ≈ 7.7 s of audio (encoder products), ≈ 3.9 s (positional projection, 2T − 1 rows) | `19d2905` | RTX 5070 Ti |
| Dispose or a failed load waiting for the background 64 × 64 compile | up to about 2.5 s | `19d2905` | RTX 5070 Ti |
| Variation between passes of long CPU matrix runs | up to 40% | `c2c2769` | Core Ultra 7 270K Plus |
| Windows GPU timeout (TDR) against the decoder submit budget in `DecodeBatchTests` | 2 s timeout, 350 ms test budget | `3022ad7` | any (Windows default) |

## 2026-11-23 · The move into ParakeetRuntime changed no transcript

- **Machine:** Intel Core Ultra 7 270K Plus, NVIDIA GeForce RTX 5070 Ti, driver not recorded, RAM not recorded
- **Commit:** ParakeetRuntime `4738960` (verbatim move) against the comment cleanup that follows it
- **Setup:** model `parakeet-tdt-0.6b-v3.q8_0.gguf`; FLEURS `pl_pl` and `en_us` `wav16`, first 20 clips ≤ 20 s
  each, plus `pl_pl` `long120`, first 2 clips; devices `gpu` (RTX 5070 Ti) and `cpu`; no affinity; 1 pass

| Run | Identical text | WER against the other build |
| --- | --- | --- |
| pl_pl ≤ 20 s, GPU | 20/20 | 0.00% |
| pl_pl ≤ 20 s, CPU | 20/20 | 0.00% |
| en_us ≤ 20 s, GPU | 20/20 | 0.00% |
| en_us ≤ 20 s, CPU | 20/20 | 0.00% |
| pl_pl 2 min, GPU | 2/2 | 0.00% |
| pl_pl 2 min, CPU | 2/2 | 0.00% |

The 93 tests also pass on both builds, and the two Q8 GEMM shaders compile to identical FXC
bytecode. No speed was measured for this record: these runs check transcripts only.

## 2026-11-23 · Corrections: machine attributions and the full GEMM tile table

- **Machine:** not applicable. This record corrects earlier records, which stay as they were written.
- **Commit:** the sources named below
- **Setup:** a check of every seeded record against its commit message and pull request

**Machine attributions.** The CPU named in these records is inferred, not stated by their source.
Read their **Machine** line as "inferred: Intel Core Ultra 7 270K Plus".

| Record | Why it's inferred |
| --- | --- |
| 2026-11-22 · First full matrix against NeMo-Speech.cpp | PR #3 names the GPUs and the pin, not the CPU. |
| 2026-11-22 · GPU decoder batches bounded by a time budget | `3022ad7` names no CPU. |
| 2026-11-22 · GPU Q8 GEMM: 64x64 tile above 96 rows | `b891d67` and PR #3 name no CPU. |
| 2026-11-23 · CPU: 3x faster activation quantizing, faster attention | `95c8512` says only "4 P-cores, affinity C03". |
| 2026-11-23 · Quantize review fixes | `304b764` names no CPU. |
| 2026-11-23 · CPU bench with tiering off | `ca13477` names no CPU. |
| 2026-11-23 · GPU: 64x64 GEMM compiled in the background; load time | `953f41c` names no CPU. |
| 2026-11-23 · Final matrix against NeMo-Speech.cpp | PR #3 names the GPUs and the pin, not the CPU. |
| 2026-11-23 · Figures that were kept in the engine's code comments: the rows for `95c8512`, `8122b32` and `c2c2769` | Their sources name no CPU. |

The inference rests on `Run-Matrix.ps1` at LocalBuddy `19d2905`, whose `C03` pin is "4 P-cores
of the Core Ultra 7 270K Plus (CPUs 0, 1, 10, 11)", and on PR #1, which names that machine. The
2026-11-22 record "CPU: one thread per fast core, up to 8" says "same hybrid CPU as the
`bf1b4a5` record". Neither source names the CPU there either. `8122b32` mentions an i7-11850H only
as an example of an 8-core laptop, not as the benchmark machine.

**Full GEMM tile table** from the `Q8Gemm.hlsl` comment at LocalBuddy `19d2905` (written in
`b891d67`, measured 2026-11-22). The 2026-11-22 GEMM record gives only part of it. The comment
names no CPU or driver. Values are speed-ups against the single 32 x 32 kernel used before, from
`--gemm-bench` with the encoder's per-layer shapes. "Frames" is the row count of most of the
layer's products.

| GPU | Kernel | 25 | 64 | 96 | 128 | 250 | 1500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | small (32 x 32) | 1.09x | 1.03x | 1.01x | 1.01x | 1.02x | 1.00x |
| RTX 5070 Ti | large (64 x 64) | 0.82x | 0.86x | 0.95x | 1.03x | 1.38x | 1.54x |
| Intel Graphics | small (32 x 32) | 1.05x | 1.10x | 1.11x | 1.09x | 1.11x | 1.12x |
| Intel Graphics | large (64 x 64) | 0.88x | 1.54x | 1.29x | 1.65x | 1.66x | 1.61x |
