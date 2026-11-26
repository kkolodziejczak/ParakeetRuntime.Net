# Using the library

```powershell
dotnet add package ParakeetRuntime.Net
```

Install the model once, load it once, and keep the engine for the life of your app. Target
`net10.0-windows`, or the build warns that the API is Windows-only.

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

## Options

- **Your own audio:** `TranscribeAsync(float[])` takes 16 kHz mono samples in the range -1 to 1, of any
  length.
- **CPU or GPU:** `SpeechEngine.LoadAsync(model, new SpeechEngineOptions(UseGpu: false))` stays on the
  processor; `GpuNameContains: "RTX"` prefers the graphics card whose name contains "RTX", and falls back to
  the processor when no such card is usable. `engine.Backend` says where the model runs, and
  `engine.IsUsingGpu` whether that is a GPU.
- **The model folder and offline installs:** `new SpeechModel(folder, SpeechModel.Parakeet)` keeps the model
  elsewhere. `model.InstallFromFileAsync(path, progress, cancellationToken)` installs a copy you downloaded
  yourself, after checking its SHA-256. See [the speech model](speech-model.md).
- One engine transcribes one call at a time; other calls wait their turn. A cancellation token stops a
  waiting call, and a long one between its pieces.
- `Dispose` releases the model and its GPU resources and returns its memory right away.

## What to expect

- **Audio:** WAV files of up to 10 minutes: PCM at 8, 16, 24 or 32 bits or float at 32 or 64 bits,
  1 to 8 channels, 8 to 192 kHz. Other formats are refused with an `InvalidDataException` that says why;
  convert them first.
- **Long audio** is cut at its quietest points into pieces of 20 seconds or less, because the model drops
  whole sentences from long audio. The texts of the pieces are joined with a space.
- **Silence** can come back as a stray word, for example "Yeah.".
- **Languages:** the 25 European languages of Parakeet TDT 0.6B v3, with punctuation and capitals. The
  language is detected by the model; there is no setting.
- **Where it runs:** a graphics card or integrated graphics with Direct3D 12 support and a video memory
  budget of at least 900 MiB, as Windows reports it. Without one, the engine runs on the processor. Windows
  10 or 11, 64-bit, on an x86-64 processor.
- **Speed and accuracy:** see the comparison in the
  [README](https://github.com/kkolodziejczak/ParakeetRuntime.Net#compared-with-nemo-speechcpp) and every
  measurement in [benchmarks.md](benchmarks.md).
