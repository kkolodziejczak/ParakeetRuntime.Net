# Brief: the consumer interface of ParakeetRuntime

## Problem

ParakeetRuntime.Net (repository `C:\Users\Krzysiek\source\repos\Parakeet.Gguf`, branch `master`, GitHub
`kkolodziejczak/ParakeetRuntime.Net`) is an offline speech-to-text engine for .NET 10 on Windows. It runs
NVIDIA Parakeet TDT 0.6B v3 (GGUF Q8_0) on the CPU (int8 kernels) or on a GPU (Direct3D 12). It ships two
things: the `ParakeetRuntime` NuGet package and a self-contained `parakeet-runtime.exe`. No version has been
released yet, so the public API can still change freely.

The owner wants the best, most frictionless way to consume it, and asks the council to design it:

1. **Library**: a user must be able to pass a WAV file and get its transcription. Ideally a user can also pass
   microphone input and get it transcribed, the way the owner's dictation app PrivateType does. The owner does
   not know how PrivateType does this (whether it records WAV files to disk and hands them to the engine, or
   streams samples). Using the library should be as easy as possible.
2. **The exe**: something you can start ("enable") that offers an API (for example a local HTTP API) and also a
   simple UI where you upload a file and see its transcription, so it doubles as a "hello world" that shows how
   the engine works.
3. **README**: updated so it shows everything: how to use the library (WAV, microphone), and how to use the exe.

The council's answer is a design: the public API surface (types, method signatures, usage snippets), how
microphone input works, what the exe does and how (API endpoints, UI, how it starts), the README outline, and
an implementation order in small, checkable steps. Prototypes that prove a choice are welcome.

## Material

- The current public API, all of it:
  - `src\ParakeetRuntime\SpeechEngine.cs` (`LoadAsync`, `TranscribeAsync(float[] 16 kHz mono)`, `Dispose`)
  - `src\ParakeetRuntime\SpeechModel.cs` (`SpeechModel`, `SpeechModelFile`, `Default()`, `IsInstalled`,
    `DownloadAsync`, `InstallFromFileAsync`)
- Internal pieces a design may expose or reuse: `src\ParakeetRuntime\ParakeetModel.cs` (constructor with
  `allowGpu` and `adapterNameFilter`, `MaxAudio`, `BackendName`, `IsUsingGpu`), `src\ParakeetRuntime\Wav.cs`.
- The package definition: `src\ParakeetRuntime\ParakeetRuntime.csproj`, `Directory.Build.props`.
- The exe today: `src\ParakeetRuntime.Cli\Program.cs` and the benchmark files next to it (it only benchmarks;
  it cannot transcribe a file). `src\ParakeetRuntime.Cli\ParakeetRuntime.Cli.csproj` already references NAudio.Core.
- Engine docs: `docs\model-and-decoding.md` (see "Runtime behaviour"), `docs\gpu-path.md`, `docs\cpu-kernels.md`,
  `docs\benchmarks.md` (measured speeds).
- Repository rules: `AGENTS.md`, `README.md`, `.github\workflows\release.yml`, `scripts\Build-Release.ps1`.
- A rendered guide to the package as it is today:
  `C:\Users\Krzysiek\AppData\Local\Temp\claude\5fddf898626c476f\c40f391e-1be9-43c4-862d-9ed89c580f21\scratchpad\parakeet-package-guide.html`
- PrivateType, the owner's dictation app that captures the microphone (read-only reference):
  `C:\Users\Krzysiek\source\repos\privatetype`, especially `src\PrivateType.App\DefaultMicrophoneCapture.cs`,
  `src\PrivateType.Core\DictationSession.cs`, `src\PrivateType.Core\VoiceActivity.cs`,
  `src\PrivateType.App\OfflineRecognizer.cs`.
- LocalBuddy, where the engine was built and where live capture exists (read-only reference):
  `C:\Users\Krzysiek\source\repos\LocalBuddy`, especially `src\LocalBuddy.Transcription\AudioSource.cs` and
  `src\LocalBuddy.Transcription\ProcessLoopbackCapture.cs`.

## Constraints

- .NET 10, Windows x64 only. The library is plain `net10.0` marked `[SupportedOSPlatform("windows")]`.
- Today the library reaches Windows only through P/Invoke into DLLs every Windows ships, with no Windows SDK
  projection, so any `net10.0` app can reference it (AGENTS.md). A design that changes this must say so and why.
- The engine transcribes one whole utterance per call, one call at a time; there is a maximum audio length per
  call (set by the model). No streaming or timestamps exist today.
- Owner's code rules (AGENTS.md): no comments in code (reasons go into names), timeless facts go into `docs/`,
  measurements into the append-only `docs/benchmarks.md`. Small commits. Pin package versions. Nullable on,
  no suppressed warnings.
- Releases are signed (exe, DLL, package) through a shared workflow; the exe is published self-contained single-file.
- The model (714 MB) is never bundled; it is downloaded and verified by `SpeechModel`.
- The council designs and prototypes only, in its notes folders. Do not change the repository.

## Tried or ruled out

- Nothing tried yet for the consumer interface. The owner already chose that the public API is "engine plus model
  download" (`SpeechEngine`, `SpeechModel`) and that live and replay commands stayed in LocalBuddy during the move.

## Good looks like

- A hello world of a few lines: install the model if needed, transcribe a WAV, print the text. Microphone
  transcription almost as short.
- The exe starts with one command, opens a simple page where you upload a file and see the text, and exposes
  the same thing as an API.
- The README shows all of it, with copy-pasteable snippets.
- The design fits the constraints above, or names exactly which one it changes and why.
- Each choice is backed by evidence (code read, a prototype that runs, a measurement).
