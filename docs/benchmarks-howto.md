# Running the benchmarks

How to run the corpus benchmark, the session memory check and the benchmark matrix, with every option.

All three need a Release build of `parakeet-runtime.exe` (`src/ParakeetRuntime.Cli`). Without
`--model`, they use the model installed in `%LOCALAPPDATA%\ParakeetRuntime\models`.

## Corpus benchmark

Measures the speed, accuracy and memory of one engine on one device, on a list of 16 kHz mono
WAV clips.

```powershell
parakeet-runtime --list-gpus
parakeet-runtime --corpus <list.txt | wav folder> [--device cpu|gpu|gpu:<adapter>|vulkan:N] [--nemo nemo-speech.exe]
                 [--affinity C03] [--passes N] [--out hyp.tsv] [--reference dev.tsv] [--compare other-hyp.tsv]
                 [--model parakeet.gguf] [--max-seconds S] [--limit N] [--int8-path Avx2|AvxVnni|...] [--threads N]
                 [--results matrix.tsv] [--cut]
```

`--list-gpus` prints every DXGI adapter with its memory, whether it is a software adapter, and
whether it has enough memory for the engine.

| Option | Meaning |
| --- | --- |
| `--corpus` | A list file with one WAV path per line, or a folder of WAVs (taken in name order). Every clip must be 16 kHz mono. |
| `--device` | Without `--nemo`: `cpu` (the default), `gpu` (the adapter the engine picks on its own), or `gpu:<part of the adapter name from --list-gpus>`. The run stops if that GPU is not used. With `--nemo`, the value is passed on to NeMo-Speech.cpp (`cpu` or `vulkan:N`). |
| `--nemo` | Runs NeMo-Speech.cpp's `serve` command instead of ParakeetRuntime, on the same model and the same `--affinity` mask but with `--threads 1`, over its HTTP API. Its load time counts until `/ready` answers, which includes the server's own warmup. |
| `--affinity` | A hex mask of the CPUs to run on, for example `C03` for CPUs 0, 1, 10 and 11. Pick cores of one kind, so runs are comparable. |
| `--passes` | How many times every clip is transcribed. Each pass prints its time; the best pass is the one reported in `--results`. |
| `--out` | Writes the transcripts as a TSV of file name and text. |
| `--reference` | A `dev.tsv` of id, file name and text; prints the word error rate against it. Only clips in this file are run. |
| `--compare` | Another run's `--out`; prints the word error rate against it and how many clips have identical text. Only clips in this file are run. |
| `--model` | Any Parakeet GGUF file, for comparing conversions. Nothing checks this file. |
| `--max-seconds` | Skips clips longer than this. Use 20 for short-utterance corpora. |
| `--limit` | Runs only the first N clips. |
| `--int8-path` | Forces a slower CPU int8 kernel than the CPU's best, to time older CPUs on a newer one. Only for `--device cpu`. |
| `--threads` | Overrides the CPU thread budget (by default one thread per fast core the process may run on, at most 8). |
| `--results` | Appends one summary row per run to a TSV, with a header when the file is new. `Run-Matrix.ps1` uses it. |
| `--cut` | Cuts clips longer than 20 s at their quietest points, as `SpeechEngine.TranscribeAsync` does (see [Long audio](model-and-decoding.md#long-audio)). Without it, each clip is one model call. Not with `--nemo`. |

What a run does and prints:

- It transcribes the first clip once untimed, then every clip once per pass. Each clip's audio is
  read from disk again before each transcription, so the corpus's audio does not count as engine
  memory.
- Each pass prints its compute time, its real-time factor (compute time divided by audio time), and
  this process's garbage-collection pauses and full collections during the pass. These pause the
  engine's threads; a NeMo-Speech.cpp server runs in its own process and is not affected.
- After the passes, ParakeetRuntime prints where the time went. On the CPU every stage is timed
  (mel, subsampling, feed-forward, attention, convolution, output norm, decoder); on a GPU only the
  encoder and decoder are.
- GPU memory is the process's dedicated and shared usage over all adapters, as Task Manager shows
  it, from Windows' "GPU Process Memory" counters. It is read after loading and after each clip,
  outside the timed work, so both engines are measured the same way. On an integrated GPU the model
  sits in shared memory, which also counts in the working set.
- The word error rate counts words the way the reference corpus's scoring script does: text is
  lower-cased, and anything other than letters and digits separates words.
- In `--results`, the `corpus` column is the corpus's last two folders, for example `en_us/long60`
  for `...\en_us\long60\list.txt`, with `<=20s` added when `--max-seconds 20` is set, and ` cut` when `--cut` is.

## Session memory check

Shows whether the engine's memory goes back to the system once a session ends.

```powershell
parakeet-runtime --session-memory <clip.wav> [--device cpu|gpu] [--sessions 3] [--idle-seconds 10]
```

| Option | Meaning |
| --- | --- |
| `--session-memory` | The 16 kHz mono clip each session transcribes. |
| `--device` | `gpu` (the default) loads the engine as `SpeechEngine.LoadAsync` does, with the engine's own choice of device; `cpu` loads it with the GPU turned off. |
| `--sessions` | How many sessions to run one after another. |
| `--idle-seconds` | How long to watch memory after each session. |

Each session loads the engine, transcribes the clip and disposes the engine. The engine object
stays referenced afterwards, as an application that keeps a finished session around would. Memory
is then read 0, 1 and 5 seconds after the session and at the end of `--idle-seconds`, with nothing running, and nothing in the check forces a garbage
collection. Each line shows private bytes, working set, the managed heap, GPU dedicated and shared
memory, and the number of full collections so far.

## Benchmark matrix

`scripts/Run-Matrix.ps1` builds `parakeet-runtime.exe` and runs the corpus benchmark for both
engines on every device, one run at a time, pinned to the same cores. Every run appends a row to
`<Out>\matrix.tsv`; its transcripts (`<run>.tsv`) and log (`<run>.log`) go next to it. After each
run the script prints the log's pass, WER, identical-text and memory-peak lines.

Full matrix:

```powershell
powershell -File scripts/Run-Matrix.ps1 -Nemo <nemo-speech.exe> [-Out <folder>]
```

NeMo-Speech.cpp on the CPU runs first for each corpus, and every other run is compared with its
transcripts.

Quick A/B of a change: run without `-Nemo` on a subset, then run again from the changed checkout
with `-Baseline`. Each run is then compared with the transcript of the same run in the baseline
folder.

```powershell
powershell -File scripts/Run-Matrix.ps1 -Languages pl_pl -Lengths long120 -LocalDevices cpu -Limit 3 -Out before
powershell -File <changed checkout>\scripts\Run-Matrix.ps1 -Languages pl_pl -Lengths long120 -LocalDevices cpu -Limit 3 -Out after -Baseline before
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `-Nemo` | none | Path to `nemo-speech.exe`. Without it only ParakeetRuntime runs. |
| `-Corpus` | `%USERPROFILE%\.councils\local-parakeet-engine\fable\corpus` | The corpus root: `<language>\wav16\list.txt` with `<language>\dev.tsv`, and `<language>\<length>\list.txt` with its own `dev.tsv`. `python scripts/Build-Corpus.py [<root>]` builds it there from FLEURS dev (about 300 MB of downloads): the first 150 clips of each language converted to 16-bit WAV, and `long60` and `long120` files that join those clips with 0.5 s of silence. |
| `-Out` | `%TEMP%\parakeet-matrix` | Where results, transcripts and logs go. |
| `-Baseline` | none | An earlier `-Out` folder; each run is compared with the same run's transcript there instead of with NeMo-Speech.cpp on the CPU. |
| `-Languages` | `pl_pl`, `en_us` | Corpus languages. |
| `-Lengths` | `wav16`, `long60`, `long120` | Corpus subsets. `wav16` runs with `--max-seconds 20`. |
| `-LocalDevices` | `gpu:RTX`, `gpu:Intel`, `cpu` | ParakeetRuntime devices; `gpu:<part of the adapter name>`. |
| `-NemoDevices` | `vulkan:0`, `vulkan:1` | NeMo-Speech.cpp devices on the same hardware, run in addition to its CPU run. |
| `-Passes` | 3 | Passes per run; every run keeps its best pass, because long CPU runs vary between passes. |
| `-Limit` | 0 | Only the first N clips of each corpus; 0 runs all of them. |
| `-Affinity` | `C03` | The CPU mask passed as `--affinity`. Choose cores of one kind on hybrid CPUs. |

Run names are `<language>-<length>-<engine>-<device without the colon>`, for example `pl_pl-long120-parakeet-cpu`
or `en_us-wav16-nemo-vulkan0`.
