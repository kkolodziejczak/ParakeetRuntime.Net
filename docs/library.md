# Using the library

Add the package to a project targeting `net10.0-windows` (without `-windows`, the build warns that the API is Windows-only):

```powershell
dotnet add package ParakeetRuntime.Net --prerelease
```

Install the model once, load `SpeechEngine` once, and hold onto that instance for the life of your app. Loading reads the 714 MB weights file and prepares the engine for your hardware, so you don't want to do it per call.

```csharp
using ParakeetRuntime.Net;

var model = SpeechModel.Default();                      // %LOCALAPPDATA%\ParakeetRuntime\models
await model.EnsureInstalledAsync(new Progress<double>(p => Console.Write($"\r{p:P0}")));
using var engine = await SpeechEngine.LoadAsync(model);

string text = await engine.TranscribeWavAsync("hello.wav");
```

You can also capture from a microphone into memory and pass those samples straight into the same engine:

```csharp
using var recording = Microphone.StartRecording();      // Windows' default microphone; Microphone.Devices() lists all of them
Console.ReadLine();
float[] samples = recording.Stop();                     // 16 kHz mono
string spoken = await engine.TranscribeAsync(samples);
```

## Engine options and lifecycle

`TranscribeWavAsync` takes either a file path or a `Stream`. Non-seekable streams work and are left open. If you already have raw PCM in memory, `TranscribeAsync(float[])` takes 16 kHz mono samples in the range -1 to 1, of any length.

By default, `SpeechEngine.LoadAsync` picks a usable discrete GPU and falls back to the CPU if there isn't one. It skips integrated GPUs, because on both machines we measured the CPU path beat them (see [benchmarks.md](benchmarks.md)); name one with `GpuNameContains`, for example `"UHD"`, to use it anyway. Pass `new SpeechEngineOptions(UseGpu: false)` to stay on the CPU, or `new SpeechEngineOptions(GpuNameContains: "RTX")` to prefer a GPU whose name contains `"RTX"` (still falling back to the CPU if no matching GPU is usable). Check `engine.Backend` to see where the model landed and `engine.IsUsingGpu` to test whether a GPU is active.

To keep the model outside `%LOCALAPPDATA%\ParakeetRuntime\models`, construct `new SpeechModel(folder, SpeechModel.Parakeet)`. On an offline machine, call `model.InstallFromFileAsync(path, progress, cancellationToken)` on a file you downloaded yourself; it verifies the SHA-256 before marking the model installed. See [the speech model](speech-model.md) for the file URL and hash.

One `SpeechEngine` runs one transcription at a time, and concurrent calls wait their turn. A `CancellationToken` cancels a waiting call, or stops a long transcription between pieces (a piece that has already started finishes). Calling `Dispose` waits for any running transcription to finish, releases the model and its GPU resources, and returns the managed memory right away.

## What to expect

`TranscribeWavAsync` accepts WAV files up to 10 minutes long: PCM at 8, 16, 24, or 32 bits, or IEEE float at 32 or 64 bits, with 1 to 8 channels at 8 to 192 kHz. Other formats throw `InvalidDataException` explaining why, so convert MP3 or other containers before passing them in.

Audio longer than 20 seconds is cut at its quietest points into pieces of 20 seconds or less, and the piece transcripts are joined with a space. We split long audio because the model starts dropping whole sentences when given long inputs in one shot. Silent audio can also come back as a stray word such as `"Yeah."`.

The model handles the 25 European languages of Parakeet TDT 0.6B v3, including punctuation and capitalization. It detects the language on its own, so there is no language setting.

The library runs on 64-bit Windows 10 or 11 on an x86-64 CPU. GPU execution needs Direct3D 12 and a Windows-reported video memory budget of at least 900 MiB; without that, the engine runs on the CPU. For speed and accuracy numbers, see the [comparison with NeMo-Speech.cpp](../README.md#compared-with-nemo-speechcpp) and the full measurements in [benchmarks.md](benchmarks.md).
