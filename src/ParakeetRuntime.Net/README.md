# ParakeetRuntime.Net

Offline speech-to-text for .NET on Windows, in pure C#. It runs NVIDIA's Parakeet TDT 0.6B v3 inside your process on the GPU or CPU across 25 European languages, so audio stays on the machine and you don't have to ship Python, CUDA, or native binaries.

Add the package to a project targeting `net10.0-windows` (without `-windows`, the build warns that the API is Windows-only):

```powershell
dotnet add package ParakeetRuntime.Net --prerelease
```

```csharp
using ParakeetRuntime.Net;

var model = SpeechModel.Default();
await model.EnsureInstalledAsync();                       // downloads and checks the model once (714 MB)
using var engine = await SpeechEngine.LoadAsync(model);   // uses the GPU if one is usable, else the CPU
string text = await engine.TranscribeWavAsync("hello.wav");
```

## Using the engine

We don't bundle the weights in the package because the file is 714 MB. `SpeechModel.Default()` stores it in `%LOCALAPPDATA%\ParakeetRuntime\models`, and `EnsureInstalledAsync` downloads it once and verifies its exact size and SHA-256 before keeping it. It is safe to call on every start; pass an `IProgress<double>` if you want download progress. To store the model elsewhere, construct `new SpeechModel(folder, SpeechModel.Parakeet)`. On an offline machine, copy the GGUF file over and call `InstallFromFileAsync(path, progress, cancellationToken)` to run the same size and hash check.

Load `SpeechEngine` once and reuse that instance for the life of your app. Loading reads the 714 MB weights file and sets up the CPU or GPU backend, so you don't want to do it per call. One engine transcribes one clip at a time; concurrent calls wait their turn, and a `CancellationToken` cancels a waiting call or stops a long clip between pieces (a piece that has started finishes). Calling `Dispose` waits for any running call to finish and frees GPU and managed memory right away.

By default, `SpeechEngine.LoadAsync` picks a usable discrete Direct3D 12 GPU through the DLLs Windows already ships (tested on NVIDIA and Intel) and falls back to the CPU if none qualifies. It skips integrated GPUs, because the CPU path beat them on both machines we measured; name one with `GpuNameContains` to use it anyway. Pass `new SpeechEngineOptions(UseGpu: false)` to stay on the CPU, or `new SpeechEngineOptions(GpuNameContains: "RTX")` to match a specific adapter by name, still falling back to the CPU if none matches. Check `engine.Backend` and `engine.IsUsingGpu` to see where the model landed.

`TranscribeWavAsync` takes a file path or a `Stream` (non-seekable streams work and are left open) and accepts WAV audio up to 10 minutes long: PCM at 8, 16, 24, or 32 bits, or IEEE float at 32 or 64 bits, 1 to 8 channels, at 8 to 192 kHz. Anything else throws `InvalidDataException`, so convert MP3 or other formats before passing them in. If you already have 16 kHz mono float samples in the range -1 to 1, `TranscribeAsync(float[])` takes them directly with no length limit.

To capture from a microphone instead of reading a file, call `Microphone.StartRecording()` for Windows' default mic, or pick one from `Microphone.Devices()`. It records into memory as 16 kHz mono samples and stops at 10 minutes by default unless you pass another `longest` `TimeSpan`. Call `Stop()` on the recording to get the `float[]` for `TranscribeAsync`, and dispose the recording when you're done so its buffers are cleared.

Audio longer than 20 seconds is cut at its quietest points into pieces of 20 seconds or less, and the piece transcripts are joined with a space. We split long audio because the model starts dropping whole sentences when given long inputs in one shot. The model detects the language on its own across the 25 European languages of Parakeet TDT 0.6B v3 and writes punctuation and capitalization, so there is no language setting. Silent audio can occasionally come back as a stray word such as `"Yeah."`.

## Speed

Compared with NVIDIA's C++ engine, NeMo-Speech.cpp, running the same `parakeet-tdt-0.6b-v3.q8_0.gguf` model on FLEURS Polish and English clips of 20 seconds or less (desktop and laptop used different clip sets). Word error rate matches NeMo-Speech.cpp on both sets: 5–6 wrong words in 100 on the desktop, 6–7 on the laptop. Lower times are faster.

| Machine and device | 1 min of audio | vs. NeMo (same device) |
| --- | --- | --- |
| Desktop GPU (RTX 5070 Ti) | **0.15–0.16 s** | 2.1–2.3× faster |
| Desktop CPU (Core Ultra 7 270K Plus) | **0.88–0.90 s** | 4.2–4.3× faster |
| Desktop integrated GPU | **1.8 s** | 2.2× faster |
| Laptop GPU (MX450) | **1.5–1.6 s** | 1.1–1.2× faster |
| Laptop CPU (Core i7-11850H) | **3.4–3.6 s** | 3.4–3.5× faster |
| Laptop integrated GPU (UHD) | **5.8–5.9 s** | About the same |

Desktop numbers were measured on 2026-11-22 and 2026-11-23 before the engine moved into this repository. Laptop numbers were measured on 2026-11-25 at [`1078f4e`](https://github.com/kkolodziejczak/ParakeetRuntime.Net/commit/1078f4e). Both predate the 2026-11-26 speed-ups. Model load times and every raw run are in [benchmarks.md](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/benchmarks.md).

## Requirements and docs

You need .NET 10 on 64-bit Windows 10 or 11 with an x86-64 CPU. GPU execution requires Direct3D 12 and a Windows-reported video memory budget of at least 900 MiB; without that, the engine runs on the CPU.

The rest of the docs are on GitHub: [Using the library](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/library.md) covers all options and edge cases, [The speech model](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/speech-model.md) has the pinned URL, SHA-256, and offline install steps, and [How it works](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/how-it-works.md) explains the audio pipeline and shaders.

If you'd rather run a standalone tool than write .NET code, `parakeet-runtime.exe` on [GitHub Releases](https://github.com/kkolodziejczak/ParakeetRuntime.Net/releases/latest) bundles the same engine with a browser UI, CLI, and local HTTP API, with no .NET installation needed (see [Using the app](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/docs/app.md)).

## License

[MIT](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/LICENSE). The engine is a C# port of parts of NVIDIA's NeMo-Speech.cpp (Apache-2.0; see [THIRD-PARTY-NOTICES.md](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/THIRD-PARTY-NOTICES.md)). The speech model is © NVIDIA under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This project is not affiliated with or endorsed by NVIDIA.
