# Model file and decoding

How the engine reads the Parakeet GGUF file, runs the encoder and decoder, and installs and verifies the model.

The engine is a C# port of NeMo-Speech.cpp. File and line references below point into
NeMo-Speech.cpp.

## Reading the GGUF file

`Gguf` reads GGUF version 3 files. Everything about the file is checked where it enters, so the
model code can trust shapes and offsets.

The layout, as llama.cpp's GGUF writer produces it:

1. Magic `GGUF` (the little-endian `uint32` `0x46554747`) and version `3`.
2. Tensor count and metadata count, each an `int64` from 0 to 1,000,000.
3. The metadata key-values. Value types 0 to 12 are read (integers, floats, bool, string, array).
   Arrays may not contain arrays.
4. One info record per tensor: name, 1 to 4 dimensions (each from 1 to `int.MaxValue - 1`), type
   and offset.
5. Tensor data, starting at the next multiple of `general.alignment` (32 when absent; 1 byte to
   1 MiB is accepted). Tensor offsets count from there.

When the file is opened, every tensor must fit inside the file. Sizes are computed with overflow
checks, so a crafted shape (for example 2^62 values, whose 2^64 bytes would wrap to 0) fails
instead of passing as "fits".

Tensor types read: F32 (0), F16 (1) and Q8_0 (8). A tensor of any other type does not stop the
file from opening, but reading it fails.

Tensors are read one at a time with random-access reads into a reused buffer, not memory-mapped,
so the file's pages never count towards the process's memory on top of the weights copied out of
it. A float tensor comes back row-major with the first dimension fastest.

### Q8_0 blocks

A Q8_0 tensor's rows are a multiple of 32 values. Each block of 32 values takes 34 bytes: an fp16
scale, then 32 `int8` values. A value is `int8 × scale`.

`Gguf.Q8` unpacks one or more Q8_0 matrices of `[rows][K]` with the same `K` into `int8` values and
one float scale per block, stacked in the order given. The engine stacks each layer's
`linear_q`, `linear_k` and `linear_v` this way, so one matrix product yields `[T][3d]`: q, then k,
then v. `Gguf.CopyQ8Raw` copies the raw `int8` values and fp16 scale bits straight into GPU staging
buffers, without managed arrays per tensor.

## Metadata

| Key | Use |
| --- | --- |
| `asr.encoder.d_model`, `n_heads`, `d_ff`, `conv_kernel_size`, `subsampling_conv_channels`, `n_layers` | Encoder shape. |
| `asr.rnnt.pred_hidden`, `pred_num_layers`, `vocab_size`, `blank_id`, `max_symbols_per_step` | Prediction network and decoding. |
| `asr.tdt.durations` | The frame skips the joint can predict. |
| `asr.tokenizer.vocab` | SentencePiece pieces. |
| `asr.preprocessor.sample_rate`, `n_fft`, `window_size`, `window_stride`, `features`, `preemph` | Mel front end. |

## Tensors

| Tensor | Use |
| --- | --- |
| `preprocessor.fb` | Mel filterbank, `[features][n_fft / 2 + 1]`. |
| `encoder.pos_enc.pe` | Sinusoidal position table, `[rows][d_model]`. |
| `encoder.pre_encode.conv.{0,2,3,5,6}.weight/bias` | Subsampling convolutions. |
| `encoder.pre_encode.out.weight/bias` | Subsampling output projection. |
| `encoder.layers.N.norm_{feed_forward1,self_att,conv,feed_forward2,out}.weight/bias` | Layer norms. |
| `encoder.layers.N.feed_forward{1,2}.linear{1,2}.weight` | Feed-forward modules. |
| `encoder.layers.N.self_attn.linear_{q,k,v,out,pos}.weight`, `pos_bias_u`, `pos_bias_v` | Attention. |
| `encoder.layers.N.conv.pointwise_conv{1,2}.weight`, `depthwise_conv.weight`, `batch_norm.{weight,bias,running_mean,running_var}` | Convolution module. |
| `decoder.prediction.embed.weight` | Token embedding. |
| `decoder.prediction.dec_rnn.lstm.{ih,hh}_lN.weight/bias` | Prediction LSTM layers. |
| `joint.enc`, `joint.pred`, `joint.joint_net.2` (`.weight/.bias`) | Joint network. |

Matrices are Q8_0; norms, biases, convolution taps and the tables are read as floats.

## Mel front end

Audio is 16 kHz mono. The front end applies pre-emphasis, pads `n_fft / 2` on each side, and
takes one frame every `window_stride` (10 ms, so 100 frames per second) with a symmetric Hann
window of `window_size`. It computes the power spectrum, applies the filterbank, takes
`log(x + 2^-24)`, and normalizes each feature to zero mean and unit variance over the frames.

## Encoder

### Subsampling

The `dw_striding` subsampling reduces the frame rate 8 times:

1. Conv 0: 3×3, stride 2, one input channel, ReLU.
2. Depthwise 3×3 with stride 2, then pointwise 1×1 and ReLU; twice.
3. Per frame, the features are flattened channel by channel (`c * W + w`) and projected to
   `d_model` by `pre_encode.out`.

The front end makes `n / hop + 1` frames for `n` samples, and the last one is zeroed
(`Mel.ZeroedLastFrames`), so `n / hop` frames are valid. Like NeMo, the encoder runs on
`ceil(valid / 8)` frames (`ParakeetModel.EncoderFramesOfValidMelFrames`, on the CPU and the GPU),
not on all the frames the three stride-2 convolutions produce. The two differ by one when the valid
count is a multiple of 8, about one clip in eight, and the extra frame came only from padding. On
the 300 FLEURS clips of 20 s or less (2026-11-26, laptop MX450) dropping it changed 6 transcripts:
WER 5.97 → 5.90% in English and 6.64 → 6.60% in Polish.

The CPU path keeps activations as `[time][frequency][channel]`, with channels in SIMD vectors,
convolution taps as `[tap][channel]` and pointwise weights transposed to `[in][out]`. The 3×3 sums
add in the same order as plain loops would. Conv 0's output is never built whole: it is computed
three rows at a time, the rows that one output row of the first depthwise convolution reads. A
null input row stands for padding.

### Conformer layer

Each of the `n_layers` layers runs, each step adding to the residual:

1. Feed-forward 1 (layer norm, linear, SiLU, linear), added at half weight (0.5).
2. Relative-position multi-head self-attention (layer norm, stacked q/k/v, attention with
   `pos_bias_u` and `pos_bias_v`, `linear_out`).
3. Convolution module: layer norm, `pointwise_conv1`, GLU (`a × sigmoid(gate)` over each frame's
   `[a | gate]` halves), depthwise convolution over time, batch norm, SiLU, `pointwise_conv2`.
   Batch norm is folded at load into a per-channel scale `weight / sqrt(running_var + 1e-5)` and
   shift `bias - running_mean × scale`.
4. Feed-forward 2, added at half weight.
5. Output layer norm.

### Positional table

For `T` frames, attention needs `linear_pos` applied to positions `T - 1` down to `-(T - 1)`. These
are pe rows `center - T` to `center + T - 2` (`2T - 1` rows), where `center = (rows + 1) / 2`. The
projection is stored transposed.

Each projected row depends only on its own pe row (activations are quantized per row), so a slice
of a longer projection is exactly what a shorter utterance would compute. The CPU path therefore
projects the rows for 251 encoder frames (20 s of audio) once per layer at load. An utterance of
`T ≤ 251` frames reads that table from column `251 - T`, the column of position `T - 1`. A longer
utterance projects its own.

The table also limits the audio length of one model call: the longest audio is
`(rows + 1) / 2 × 8 / 100 - 1` seconds. Longer audio throws `ArgumentException`. `SpeechEngine`
never reaches that limit, because it cuts long audio first (see [Long audio](#long-audio)).

## Long audio

The model drops whole sentences when one call holds long audio made of separate utterances: on
FLEURS clips joined into files of 60 s and 120 s, English loses 30–45% of its words, in
ParakeetRuntime.Net and in NVIDIA's PyTorch model alike, with full or local attention. See the
2026-11-24 long-audio record in [benchmarks.md](benchmarks.md).

`SpeechEngine.TranscribeAsync` therefore cuts (`QuietestPointCutter`):

- Audio of 320,000 samples (20 s) or less is one model call, on the caller's array.
- Longer audio is measured in 30 ms frames (480 samples), each frame's loudness being its RMS. Of
  the 666 frames that fit in 20 s from the piece's start, each window of 10 frames (300 ms) that lies
  wholly inside them and is centred at frame 333 or later is a candidate. The piece ends at the
  centre of the candidate with the lowest summed loudness; on a tie the later centre wins. A piece is
  therefore 9.99 to 19.83 s long, and the last piece takes the rest.
- Every sample belongs to exactly one piece. The pieces' trimmed texts are joined with one space,
  and empty texts are left out.
- One call holds the engine for all of its pieces. Its cancellation token is checked before each
  piece; a piece that has started finishes.

Cutting brings the joined files back to the error rate of single clips (6.0–6.2%). Measured pieces
of 10, 15, 20, 25 and 30 s gave 6.8, 5.8, 6.0, 6.6 and 7.2% on English `long120`, and cutting at
every 700 ms pause gave 10.5–13.8% on the same files (council `parakeet-consumer-api` prototype,
2026-11-23: Core Ultra 7 270K Plus with RTX 5070 Ti, `parakeet-tdt-0.6b-v3.q8_0.gguf`, the FLEURS `long60` and
`long120` files of the 2026-11-24 record).
`parakeet-runtime --corpus … --cut` runs the same cutter, for benchmarks.

## Decoder

### Prediction network and joint

The prediction network embeds the last emitted token (zeros before the first) and runs it through
`pred_num_layers` LSTM layers, gates in the order input, forget, cell, output. The joint adds the
projected encoder frame and the projected prediction, applies ReLU, and projects to
`vocab_size` token logits followed by one logit per entry of `asr.tdt.durations`.

### TDT greedy decoding

Greedy decoding follows NeMo's `GreedyTDTInfer` (NeMo-Speech.cpp
`rnnt_greedy_decoder.cpp:388-467`):

1. Start at frame `t = 0`, with the prediction for "no token yet".
2. At frame `t`, pick the best token and, separately, the best duration.
3. A blank with duration 0 counts as duration 1.
4. A non-blank token is emitted, and the prediction network advances with it.
5. Move `t` forward by the duration. Stay on the frame while the duration is 0, up to
   `max_symbols_per_step` symbols; after that many with duration 0, move on by one frame.
6. Stop when `t` reaches the last frame.

### Text

Tokens are SentencePiece pieces: `▁` becomes a space, and pieces of the form `<0xNN>` are raw
UTF-8 bytes. The decoded text is trimmed, then punctuation spacing is fixed as in NeMo-Speech.cpp
`src/asr/postproc/pipeline.cpp:81`: spaces before `. , ? ! ; : % ) ] } -` and after `( [ { -`
are removed. The result keeps the punctuation and capitals the model writes.

## Runtime behaviour

- `ParakeetModel` is not thread-safe. `SpeechEngine` runs one transcription at a time. Once a
  transcription starts it runs to the end, even when the caller cancels, because a cancelled wait
  would release the lock while the model still runs.
- The CPU path reuses its buffers through two `EncoderScratch` instances the model owns, one for
  the subsampling and one for the encoder layers, kept from one utterance to the next. Within an
  utterance a slot holds the same buffer for every layer, so a rented buffer still holds whatever
  the previous layer left in it. The first rent of a slot in an utterance zero-fills the requested
  length, so every utterance starts as it would with fresh arrays (the attention's key runs rely on
  zero padding). A slot only grows, so a buffer can be longer than requested; code uses the
  lengths it was given, never `.Length`. After an utterance over 20 s, and on `Dispose`, the
  model drops both scratches, so a long call doesn't pin its larger buffers. A CPU transcription
  of 10 s allocates about 66 MB without the reuse and under 8 MB with it.
- When the GPU fails with a lost device (a GPU timeout reset, a driver reset or running out of video
  memory, seen as `COMException`, `InvalidOperationException` or `OutOfMemoryException`), the
  engine retires the GPU for good, loads the CPU weights, and finishes the utterance on the CPU.
- The GPU decoder runs several decoder steps per submit: enough to save round trips to the CPU, few
  enough that a slow GPU never runs one submit anywhere near Windows' 2 second GPU timeout.
- Disposing a `SpeechEngine` drops its model and runs one aggressive, compacting full garbage
  collection. The CPU weights are managed arrays, and without that collection they would stay in
  the process until some later full collection, which an idle application may not run for a long
  time. The model also drops its own weight arrays when disposed: a caller that disposes in the
  continuation of an awaited transcription still holds the model on its stack (in the awaited call's
  state), and the collection would otherwise find the 780 MB of CPU weights still reachable.
- `LoadAsync` also collects once at the end of the load. A GPU load leaves about 73 MB of float
  tensors it read only to upload, and the GPU path allocates almost nothing afterwards, so no
  collection would ever free them; there the live heap is nearly empty and a compacting collection
  costs a few milliseconds. A CPU load gets the background collection instead, because compacting
  would move 780 MB of weights.
- After each transcription that ran on the CPU, the engine starts a background full garbage
  collection, which doesn't block the caller. The CPU path's buffers for one utterance are large
  objects, and the 860 MB of managed weights let the runtime put off a full collection for many
  utterances, so without it the working set grows by about the size of those buffers per utterance.
  GPU transcriptions allocate almost nothing managed and skip it.

## Model install and verification

`SpeechModel.Parakeet` pins NVIDIA's own Q8_0 conversion of `parakeet-tdt-0.6b-v3` at a fixed
Hugging Face revision: its URL, its size (713,975,456 bytes) and its SHA-256. The model is licensed
under CC BY 4.0. `SpeechModel.Default()` keeps it in `%LOCALAPPDATA%\ParakeetRuntime\models`.

- **Installed** means: the file exists, has the pinned size, and a `<file>.verified` record next to
  it holds `<sha256> <size> <last write time in ticks>` matching the file. A start therefore checks
  size and write time instead of hashing the whole file. A file put in the folder by hand, or
  changed later, is not installed until it is checked again. If another process (for example a
  virus scanner) holds the file or its record, or one is deleted meanwhile, the model counts as not
  installed.
- **Download and install from a file** write to `<file>.<guid>.part` while hashing. The unique name
  keeps two installers running at the same time apart. The copy stops early once it is longer than
  the pin. The part file is flushed to disk before the rename, so a power cut cannot leave a short
  file under the final name. Only a matching file is renamed to the final name and recorded as
  verified; a crash or a failed check leaves no file under the final name, and the next install
  starts over.
- A cancellation that arrives together with the last bytes still keeps nothing.
- **Leftover part files** older than one hour are deleted before an install. A younger one may
  belong to another installer that is still writing or about to rename it, so it stays.
- Deleting a part file is best effort: a scanner or another installer may still hold it, and that
  must not hide the real error.
- **Stalls**: a read that gets no data for `StallTimeout` (60 seconds by default) fails with
  `TimeoutException`, because a proxy can hold a dead connection open forever.
- **A file already at the final path**, passed to `InstallFromFileAsync`, is hashed where it is and
  deleted if it does not match.
