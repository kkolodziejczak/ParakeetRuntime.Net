# ParakeetRuntime.Net

<p align="center">
  Offline speech-to-text for .NET on Windows, running NVIDIA's Parakeet TDT models in GGUF format.
</p>

<p align="center">
  <a href="https://github.com/kkolodziejczak/ParakeetRuntime.Net/actions/workflows/ci.yml"><img src="https://github.com/kkolodziejczak/ParakeetRuntime.Net/actions/workflows/ci.yml/badge.svg" alt="Build and test status" /></a>
  <a href="https://github.com/kkolodziejczak/ParakeetRuntime.Net/releases/latest"><img src="https://img.shields.io/github/v/release/kkolodziejczak/ParakeetRuntime.Net" alt="Latest release" /></a>
  <a href="https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/LICENSE"><img src="https://img.shields.io/github/license/kkolodziejczak/ParakeetRuntime.Net" alt="MIT license" /></a>
</p>

ParakeetRuntime.Net is a speech recognition engine written in C#. It loads NVIDIA's Parakeet TDT 0.6B v3
model from a GGUF file and transcribes audio on the processor, with int8 kernels, or on the graphics
card, through Direct3D 12. Audio never leaves your computer, and no Python or separate native runtime
is needed.

> **Status: before the first release.** The engine moved here from
> [LocalBuddy](https://github.com/kkolodziejczak/LocalBuddy), where it was built and tuned. Its
> API may still change before version 1.0.

## Two ways to use it

| Download | Choose it when |
| --- | --- |
| `ParakeetRuntime.Net-<version>-win-x64.zip` from [Releases](https://github.com/kkolodziejczak/ParakeetRuntime.Net/releases/latest) | You want `parakeet-runtime.exe`: a page in your browser, a command line and a local HTTP API. No .NET installation needed. |
| The `ParakeetRuntime.Net` NuGet package | You want speech-to-text inside your own .NET 10 application. |

## Try it in a minute

1. Unzip the release and double-click `parakeet-runtime.exe`. It opens a page in your browser at
   `http://127.0.0.1:5092/`.
2. The first time, the page offers to download the speech model (714 MB). Nothing downloads until you
   click **Download model**.
3. Drop a WAV file on the page. Its text appears with the time it took, a **Copy text** button and the
   `curl` command that does the same.

From a terminal in the unzipped folder:

```powershell
.\parakeet-runtime.exe download                 # installs the model once
.\parakeet-runtime.exe transcribe hello.wav     # prints the text
.\parakeet-runtime.exe record                   # records the default microphone until Enter (at most 10 minutes), prints the text
.\parakeet-runtime.exe serve --no-browser       # the page and the HTTP API, without opening the browser
curl.exe -F file=@hello.wav http://127.0.0.1:5092/v1/audio/transcriptions
```

Write `curl.exe`, not `curl`: in Windows PowerShell 5.1, `curl` is another command. `transcribe`, `record`
and `serve` take `--device cpu`, `gpu` or `gpu:<part of the adapter's name>`; without it, a usable GPU is
chosen, else the processor. `.\parakeet-runtime.exe --help` lists everything.

## Use it in your app

```powershell
dotnet add package ParakeetRuntime.Net
```

Install the model once, load it once, and keep the engine for the life of your app. Loading takes a few
seconds.

```csharp
using ParakeetRuntime.Net;

var model = SpeechModel.Default();                      // %LOCALAPPDATA%\ParakeetRuntime\models
await model.EnsureInstalledAsync(new Progress<double>(p => Console.Write($"\r{p:P0}")));
using var engine = await SpeechEngine.LoadAsync(model);

string text = await engine.TranscribeWavAsync("hello.wav");
```

Record from the microphone, then transcribe with the same engine:

```csharp
using var recording = Microphone.StartRecording();      // Windows' default microphone; Microphone.Devices() lists the others
Console.ReadLine();
float[] samples = recording.Stop();                     // 16 kHz mono
string spoken = await engine.TranscribeAsync(samples);
```

- **Your own audio:** `TranscribeAsync(float[])` takes 16 kHz mono samples in the range -1 to 1, of any
  length.
- **CPU or GPU:** `SpeechEngine.LoadAsync(model, new SpeechEngineOptions(UseGpu: false))` stays on the
  processor; `GpuNameContains: "RTX"` prefers the graphics card whose name contains "RTX", and falls back to the processor
  when no such card is usable. `engine.Backend` says where the model runs,
  and `engine.IsUsingGpu` whether that is a GPU.
- **The model folder and offline installs:** `new SpeechModel(folder, SpeechModel.Parakeet)` keeps the model
  elsewhere. `model.InstallFromFileAsync(path, progress, cancellationToken)` installs a copy you downloaded
  yourself, after checking its SHA-256.
- One engine transcribes one call at a time; other calls wait their turn. A cancellation token stops a
  waiting call, and a long one between its pieces.

## What to expect

- **Audio:** WAV files of up to 10 minutes: PCM at 8, 16, 24 or 32 bits or float at 32 or 64 bits,
  1 to 8 channels, 8 to 192 kHz. Other formats are refused with an `InvalidDataException` that says why;
  convert them first.
- **Long audio** is cut at its quietest points into pieces of 20 seconds or less, because the model drops
  whole sentences from long audio. The texts of the pieces are joined with a space.
- **Silence** can come back as a stray word, for example "Yeah.".
- **Languages:** the 25 European languages of Parakeet TDT 0.6B v3, with punctuation and capitals. The
  language is detected by the model; there is no setting.
- **Speed and accuracy** on a laptop (Intel Core i7-11850H on 4 cores, NVIDIA GeForce MX450 2 GB, Intel UHD
  Graphics, 32 GB RAM), measured 2026-11-25 with `parakeet-tdt-0.6b-v3.q8_0.gguf` on the first 150 FLEURS dev
  clips of 20 s or less in Polish and in English:

  | | Processor | MX450 | Intel UHD |
  | --- | --- | --- | --- |
  | Seconds of work per minute of audio | 3.4–3.6 s | 1.4 s | 5.9 s |
  | Words wrong (word error rate) | 6.0–6.7% | 6.0–6.6% | 6.0–6.6% |
  | Loading the model (2026-11-26) | 0.6–1.4 s | 0.4–0.6 s | 0.4–0.5 s |

  On a desktop (Intel Core Ultra 7 270K Plus, NVIDIA GeForce RTX 5070 Ti), measured 2026-11-23 on the same
  model and FLEURS Polish and English clips of 20 s to 2 minutes, the RTX took under 0.2 s and the processor
  0.9–1.6 s per minute of audio. FLEURS clips joined into files of about 60 and 120 s, cut into pieces, gave
  5.9–6.3% wrong words (2026-11-24). Every measurement, with its machine and command, is in
  [benchmarks.md](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/benchmarks.md).

## How it works

1. **Audio in:** a WAV is converted to 16 kHz mono. Audio over 20 seconds is cut at its quietest points.
2. **Features:** each piece becomes a log-mel spectrogram, 128 bands every 10 ms.
3. **Encoder:** a 24-layer FastConformer turns the spectrogram into one vector per 80 ms. This is most of the
   work: on the processor it runs int8 matrix kernels (AVX-VNNI-INT8, AVX-VNNI or AVX2, the fastest the CPU
   has, and plain 128-bit vectors on CPUs without AVX2) with one thread per fast core; on a GPU it runs as Direct3D 12 compute shaders.
4. **Decoder:** TDT greedy decoding picks the next word piece, and how many frames to skip, until the audio
   ends. The pieces become text with punctuation and capitals.

The large matrices stay 8-bit in memory, as in the GGUF file: int8 values with one scale per block of 32, on
the processor or the graphics card. Norms, biases and the small tables are floating point. The details are in
[docs](https://github.com/kkolodziejczak/ParakeetRuntime.Net/tree/master/docs).

## HTTP API

`parakeet-runtime serve` (or the exe with no arguments) answers on `http://127.0.0.1:5092` only, never on
the network. `--port` changes the port.

| Route | What it does |
| --- | --- |
| `GET /` | The page. |
| `GET /health` | `{"status","progress","version","backend","model","error"}`. `status` is `no-model`, `downloading`, `loading`, `ready` or `failed`. HTTP 200 when ready, 503 otherwise. |
| `POST /v1/audio/transcriptions` | `multipart/form-data` with one `file` part holding a WAV, and optionally `response_format=text`. Answers `{"text":"…","duration":17.54}` (seconds of audio), or the plain text. Other fields are ignored, so OpenAI-style clients work. |

A transcription request that fails answers `{"error":{"code","message"}}`:

| Status | When |
| --- | --- |
| 400 | The request or the WAV is malformed. |
| 403 | The request came from another web page. |
| 408 | The request body did not arrive within one minute of its turn starting. |
| 413 | The request body is over 256 MiB. |
| 415 | The request is not `multipart/form-data`, the file is not a WAV, or its encoding is not supported. |
| 421 | The request used another address than `127.0.0.1:<port>`, for example `localhost`. |
| 422 | The audio is longer than 10 minutes. |
| 503 | The model is not ready, or one transcription runs and another already waits. |

Requests must use the address the exe prints (`127.0.0.1:<port>`) and must not come from another web
page, so websites you visit can't use the server. Nothing about the audio or its text is logged or written
to disk.

## The speech model

The model is not included. The page, `parakeet-runtime.exe download` and `SpeechModel.EnsureInstalledAsync`
download it once and check it:
[`parakeet-tdt-0.6b-v3.q8_0.gguf`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.q8_0.gguf).
Its SHA-256 must be `e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e`. The model
is © NVIDIA, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

On a computer without internet access, copy the file over and install it with
`.\parakeet-runtime.exe download --from <file>`, or `SpeechModel.InstallFromFileAsync`. Both check it the same way.

Load only model files you trust: a GGUF file is input to the engine like any other file.

## Verify a download

Each release file has a `.sha256` file next to it. The hash must match exactly:

```powershell
Get-FileHash .\ParakeetRuntime.Net-0.1.0-win-x64.zip -Algorithm SHA256
```

`parakeet-runtime.exe`, `ParakeetRuntime.Net.dll` and the NuGet package are signed by
**Open Source Developer Krzysztof Kołodziejczak** (certificate issued by Certum). For the exe or
the DLL, right-click the file, choose **Properties → Digital Signatures**, and confirm that name.
For the package, run `dotnet nuget verify ParakeetRuntime.Net.<version>.nupkg`. Windows SmartScreen
may still warn while the certificate builds reputation. Download only from this repository.

## Requirements

- Windows 10 or 11, 64-bit, on an x86-64 processor.
- For the GPU path, a graphics card or integrated graphics with Direct3D 12 support and a video memory
  budget of at least 900 MiB, as Windows reports it. Without one, the engine runs on the processor.
- For the NuGet package, .NET 10. Target `net10.0-windows`, or the build warns that the API is
  Windows-only.

## Help and feedback

- [Get help or report a problem](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/SUPPORT.md)
- [Report a security vulnerability](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/SECURITY.md)
- [Contribute](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/CONTRIBUTING.md)
  and the [Code of Conduct](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/CODE_OF_CONDUCT.md)

## License

ParakeetRuntime.Net is [MIT licensed](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/LICENSE).
Its engine is a C# port of parts of NVIDIA's NeMo-Speech.cpp (Apache-2.0); see
[THIRD-PARTY-NOTICES.md](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/THIRD-PARTY-NOTICES.md).

ParakeetRuntime.Net is an independent open-source project. It is not affiliated with, endorsed by, or
supported by NVIDIA.
