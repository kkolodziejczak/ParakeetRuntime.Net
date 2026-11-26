# ParakeetRuntime.Net

[![CI](https://github.com/kkolodziejczak/ParakeetRuntime.Net/actions/workflows/ci.yml/badge.svg)](https://github.com/kkolodziejczak/ParakeetRuntime.Net/actions/workflows/ci.yml)
[![NuGet](https://img.shields.io/nuget/vpre/ParakeetRuntime.Net?label=NuGet)](https://www.nuget.org/packages/ParakeetRuntime.Net)
[![NuGet downloads](https://img.shields.io/nuget/dt/ParakeetRuntime.Net?label=downloads)](https://www.nuget.org/packages/ParakeetRuntime.Net)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/LICENSE)

**Offline speech-to-text for .NET on Windows.** NVIDIA's Parakeet TDT 0.6B v3 runs inside your process, on
the processor or the graphics card, in 25 European languages. Audio never leaves your computer.

```csharp
using ParakeetRuntime.Net;

var model = SpeechModel.Default();
await model.EnsureInstalledAsync();                       // downloads and checks the model once (714 MB)
using var engine = await SpeechEngine.LoadAsync(model);   // under a second on a GPU
string text = await engine.TranscribeWavAsync("hello.wav");
```

## Why

- **Faster than NVIDIA's own C++ engine** on most hardware, with the same accuracy (table below).
- **One NuGet package, nothing else to install.** Pure C#: no Python, no native runtime, no CUDA. The GPU path
  uses Direct3D 12 (tested on NVIDIA and Intel graphics); without a usable GPU it runs on the processor.
- **Private by design.** Transcription is local. The model is pinned to one file and its SHA-256 is checked.
- **Also an app.** `parakeet-runtime.exe` gives you a page in the browser, a command line and a local HTTP API,
  with no .NET installation.

## Compared with NeMo-Speech.cpp

NVIDIA's NeMo-Speech.cpp against ParakeetRuntime.Net, on the same device, the same model and the same clips
(FLEURS, Polish and English, 20 s or less). "Wrong words" counts how many words in 100 come out different from
what was said; lower is better.

| Machine | Runs on | Speed | Wrong words in 100 | Model load |
| --- | --- | --- | --- | --- |
| Desktop | Processor (Core Ultra 7 270K Plus) | **4.2–4.3× faster** | 5–6, same as NeMo | 1.0 s, NeMo 0.6–1.6 s |
| Desktop | Graphics card (RTX 5070 Ti) | **2.1–2.3× faster** | 5–6, same as NeMo | 1.9 s, NeMo 0.6–1.6 s |
| Desktop | Integrated graphics | **2.2× faster** | 5–6, same as NeMo | 2.3 s, NeMo 0.6–1.6 s |
| Laptop | Processor (Core i7-11850H) | **3.4–3.5× faster** | 6–7, same as NeMo | 0.4 s, NeMo 1.1–2.2 s |
| Laptop | Graphics card (MX450) | **1.1–1.2× faster** | 6–7, same as NeMo | 0.5–1.0 s, NeMo 1.1–2.2 s |
| Laptop | Integrated graphics (UHD) | about the same | 6–7, same as NeMo | 0.5–0.6 s, NeMo 1.1–2.2 s |

Measured on:

- **Desktop:** 2026-11-22 and 2026-11-23, on the engine from before it moved into this repository.
- **Laptop:** 2026-11-25 at [`1078f4e`](https://github.com/kkolodziejczak/ParakeetRuntime.Net/commit/1078f4e);
  load 2026-11-26 at [`d5feee7`](https://github.com/kkolodziejczak/ParakeetRuntime.Net/commit/d5feee7).

Every run, with its command and exact numbers, is in
[benchmarks.md](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/benchmarks.md).

Measured before the 2026-11-26 speed-ups; every run, with its command, is in
[benchmarks.md](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/benchmarks.md).

## Quick start

**In your app:** `dotnet add package ParakeetRuntime.Net --prerelease`, then the code above.
Target `net10.0-windows`.

**As an app:** download `ParakeetRuntime.Net-<version>-win-x64.zip` from
[Releases](https://github.com/kkolodziejczak/ParakeetRuntime.Net/releases/latest), unzip it and run:

```powershell
.\parakeet-runtime.exe                        # opens the page in your browser; drop a WAV on it
.\parakeet-runtime.exe transcribe hello.wav   # prints the text
.\parakeet-runtime.exe record                 # records the microphone until Enter, prints the text
curl.exe -F file=@hello.wav http://127.0.0.1:5092/v1/audio/transcriptions
```

Requirements: Windows 10 or 11, 64-bit. The GPU path needs Direct3D 12 and 900 MiB of video memory.

## Documentation

- [Using the library](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/library.md): loading, CPU or GPU, microphone, audio formats, long audio
- [Using the app](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/app.md): the page, commands, the HTTP API and its errors
- [The speech model](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/speech-model.md): download, offline install, verifying release files
- [How it works](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/how-it-works.md) and the engine pages it links to
- [Benchmarks](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/benchmarks.md) · [Contributing](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/CONTRIBUTING.md) · [Support](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/SUPPORT.md) · [Security](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/SECURITY.md) · [Code of Conduct](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/CODE_OF_CONDUCT.md)

## License

[MIT](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/LICENSE). The engine is a C# port of parts
of NVIDIA's NeMo-Speech.cpp (Apache-2.0, see
[notices](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/THIRD-PARTY-NOTICES.md)). The model is
© NVIDIA under CC BY 4.0. This project is not affiliated with or endorsed by NVIDIA.
