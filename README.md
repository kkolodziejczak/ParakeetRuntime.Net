# ParakeetRuntime.Net

[![CI](https://github.com/kkolodziejczak/ParakeetRuntime.Net/actions/workflows/ci.yml/badge.svg)](https://github.com/kkolodziejczak/ParakeetRuntime.Net/actions/workflows/ci.yml)
[![NuGet](https://img.shields.io/nuget/vpre/ParakeetRuntime.Net?label=NuGet)](https://www.nuget.org/packages/ParakeetRuntime.Net)
[![NuGet downloads](https://img.shields.io/nuget/dt/ParakeetRuntime.Net?label=downloads)](https://www.nuget.org/packages/ParakeetRuntime.Net)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/LICENSE)

Offline speech-to-text for .NET on Windows. ParakeetRuntime.Net runs NVIDIA's Parakeet TDT 0.6B v3 inside your process, on the CPU or GPU, across 25 European languages. Your audio never leaves the machine.

```csharp
using ParakeetRuntime.Net;

var model = SpeechModel.Default();
await model.EnsureInstalledAsync();                       // downloads and checks the model once (714 MB)
using var engine = await SpeechEngine.LoadAsync(model);   // uses the GPU if one is usable, else the CPU
string text = await engine.TranscribeWavAsync("hello.wav");
```

## Why

It is pure C# in one NuGet package, so you don't have to ship Python, CUDA, or native binaries alongside your app. On a GPU, the engine talks to Direct3D 12 (tested on NVIDIA and Intel) through the DLLs Windows already ships; if there isn't a usable discrete GPU, it falls back to the CPU, which beat the integrated GPUs we measured.

On most hardware it runs faster than NVIDIA's own C++ engine, NeMo-Speech.cpp, with the same accuracy. Transcription stays local, and the model is pinned to one exact file verified by SHA-256.

If you don't want to write .NET code, `parakeet-runtime.exe` packages the same engine as a standalone app with a browser UI, a command line, and a local HTTP API, with no .NET installation required.

## Compared with NeMo-Speech.cpp

Both engines ran the same model (`parakeet-tdt-0.6b-v3.q8_0.gguf`) on the same hardware against FLEURS Polish and English clips of 20 seconds or less (desktop and laptop used different clip sets). "1 min of audio" is how long ParakeetRuntime.Net takes to transcribe a minute of those clips, and "Wrong words in 100" is the word error rate. Lower is better for both.

| Machine | Device | 1 min of audio | vs. NeMo (same device) | Wrong words in 100 | Model load |
| --- | --- | --- | --- | --- | --- |
| Desktop | GPU (RTX 5070 Ti) | **0.15–0.16 s** | 2.1–2.3× faster | 5–6 (same as NeMo) | 1.9 s (NeMo: 0.6–1.6 s) |
| Desktop | CPU (Core Ultra 7 270K Plus) | **0.88–0.90 s** | 4.2–4.3× faster | 5–6 (same as NeMo) | 1.0 s (NeMo: 0.6–1.6 s) |
| Desktop | Integrated GPU | **1.8 s** | 2.2× faster | 5–6 (same as NeMo) | 2.3 s (NeMo: 0.6–1.6 s) |
| Laptop | GPU (MX450) | **1.0 s** | 1.7–1.8× faster | 6–7 (same as NeMo) | 0.6 s (NeMo: 1.1–1.6 s) |
| Laptop | CPU (Core i7-11850H) | **3.2–3.3 s** | 3.7× faster | 6–7 (same as NeMo) | 0.4 s (NeMo: 1.1–1.6 s) |
| Laptop | Integrated GPU (UHD) | **5.2–5.5 s** | 1.0–1.1× faster | 6–7 (same as NeMo) | 0.4–0.5 s (NeMo: 1.1–1.6 s) |

Desktop numbers were measured on 2026-11-22 and 2026-11-23 on the engine before it moved into this repository. Laptop numbers were measured on 2026-11-27 at [`22fb689`](https://github.com/kkolodziejczak/ParakeetRuntime.Net/commit/22fb689), against NeMo's run on the same laptop and clips on 2026-11-25. The desktop's numbers predate the 2026-11-26 and 2026-11-27 speed-ups. An integrated GPU runs only when you name it; the CPU beats it. Every run, with its command and full numbers, is in [benchmarks.md](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/benchmarks.md).

## Quick start

In a .NET project targeting `net10.0-windows`, install the package and run the snippet at the top:

```powershell
dotnet add package ParakeetRuntime.Net --prerelease
```

To use the standalone app instead, download `ParakeetRuntime.Net-<version>-win-x64.zip` from [Releases](https://github.com/kkolodziejczak/ParakeetRuntime.Net/releases/latest), unzip it, and run it from PowerShell:

```powershell
.\parakeet-runtime.exe                        # opens the browser UI; drop a WAV file on it
.\parakeet-runtime.exe transcribe hello.wav   # prints the text
.\parakeet-runtime.exe record                 # records the microphone until Enter, prints the text

# While the app (or serve) is running, you can also call the local HTTP API:
curl.exe -F file=@hello.wav http://127.0.0.1:5092/v1/audio/transcriptions
```

You need 64-bit Windows 10 or 11. The GPU path requires Direct3D 12 and 900 MiB of video memory; without that, the engine runs on the CPU.

## Documentation

- [Using the library](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/library.md) explains loading, CPU and GPU options, microphone recording, WAV formats, and long audio.
- [Using the app](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/app.md) walks through the browser UI, CLI commands, the HTTP API, and its error codes.
- [The speech model](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/speech-model.md) details model downloads, offline installs, and verifying release checksums and signatures.
- [How it works](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/how-it-works.md) outlines the pipeline and links to the internal engine docs.

Also see [benchmarks](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/benchmarks.md), [contributing](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/CONTRIBUTING.md), [support](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/SUPPORT.md), [security](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/SECURITY.md), and the [code of conduct](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/CODE_OF_CONDUCT.md).

## License

[MIT](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/LICENSE). The engine is a C# port of parts of NVIDIA's NeMo-Speech.cpp (Apache-2.0; see [THIRD-PARTY-NOTICES.md](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/THIRD-PARTY-NOTICES.md)). The speech model is © NVIDIA under CC BY 4.0. This project is not affiliated with or endorsed by NVIDIA.
