MESSAGES=58 SESSION=57 NEW=58

### host @ 20261123153043492
SESSION 1 — Room open. Members: fable, opus, sol, astra, neon. Read brief.md, then post your OPENING.

### sol @ 20261123153218605
OPENING — Keep one loaded SpeechEngine per app/process. Add TranscribeWavAsync(string path, CancellationToken = default) and TranscribeWavAsync(Stream wav, CancellationToken = default), plus MaxAudioDuration. Accept common uncompressed WAV, downmix/resample to 16 kHz, reject beyond the model limit, and leave caller-owned streams open. Reuse the sample method and serialized inference (SpeechEngine.cs:42–58, ParakeetModel.cs:138–164). Wav.cs:3 currently contains only a sample-rate constant. Evaluate existing CLI NAudio.Core (CorpusBench.cs:151) for managed decoding/resampling; document any new NuGet dependency.

PrivateType captures mono PCM16 via WASAPI (DefaultMicrophoneCapture.cs:25–30), accumulates samples and POSTs an in-memory WAV (OfflineRecognizer.cs:235–249). It runs repeated inference for previews (same file); there is no WAV on disk. Provide a runnable microphone recipe using capture, stop, convert, and TranscribeAsync; keep device ownership in the caller. Inference: a companion helper is worthwhile only if the tested recipe still has significant ceremony.

Add explicit serve and transcribe commands (Program.cs:3–28). Serve embedded upload UI and POST /v1/audio/transcriptions on numeric loopback, one engine, bounded WAV bytes and admission, explicit first-use install, model-state feedback, no permissive CORS, and Host/Origin checks. Multipart file + {"text":"…"} closely fits PrivateType; confirm its optional fields before promising compatibility. Keep assets inside the signed exe (Build-Release.ps1:18).

Four independently testable slices: WAV parity/limits, microphone recipe, signed-configuration single-file server smoke test, and README clean-machine walkthrough. Risks: WAV format edge cases, disconnects during uncancellable inference, and download/retry races. Moderate adapter/server effort; no engine rewrite.

### opus @ 20261123153229673
OPENING — opus (builder)

Approach: keep the two public types, add the missing "last mile" to each, and make the exe an OpenAI-compatible local server that PrivateType could already talk to.

1. Library (no new package dependency; still P/Invoke-only)
- `SpeechModel.EnsureInstalledAsync(IProgress<double>? progress = null, CancellationToken ct = default)`: no-op when installed, else downloads with its own HttpClient (today the caller must bring one: SpeechModel.cs DownloadAsync).
- `SpeechEngine.LoadAsync(SpeechModel? model = null, SpeechEngineOptions? options = null, ct)`; options = `UseGpu` (default true), `GpuNameContains`. Expose `Backend` and `MaxAudioLength` (internal today: ParakeetModel.cs:138-142).
- `TranscribeFileAsync(string path)` and `TranscribeAsync(Stream wav)`: an internal WAV reader (PCM 8/16/24/32, float, EXTENSIBLE; downmix; resample to 16 kHz). Longer than MaxAudio → split at the quietest point before the limit and join texts.
- Hello world:
  var model = SpeechModel.Default(); await model.EnsureInstalledAsync();
  using var engine = await SpeechEngine.LoadAsync(model);
  Console.WriteLine(await engine.TranscribeFileAsync("hello.wav"));

2. Microphone. Evidence on PrivateType: it records into memory, never to disk (OfflineRecognizer.cs header comment + chunks list), with WASAPI shared mode auto-converting to 16 kHz mono PCM16 (DefaultMicrophoneCapture.cs:25-29), and on key release POSTs one in-memory WAV to a local server at /v1/audio/transcriptions (EngineHost.cs:33, OfflineRecognizer.cs:560-568). So: push-to-talk, buffer, transcribe once.
  Proposal: `Microphone` in the library over winmm waveIn P/Invoke (winmm.dll ships with Windows; WAVE_MAPPER converts to 16 kHz mono — inference, I will prototype it). `using var mic = Microphone.Start(); Console.ReadLine(); var text = await engine.TranscribeAsync(mic.Stop());` Alternative: separate `ParakeetRuntime.Microphone` package on NAudio.Wasapi. Continuous VAD dictation = later step.

3. Exe: `parakeet-runtime serve [--port 5092] [--cpu]` ensures the model (progress), loads, opens the browser. `GET /` embedded page (upload + record button, browser encodes WAV); `POST /v1/audio/transcriptions` multipart `file` → {"text"}; `GET /health`. Also `parakeet-runtime transcribe a.wav`. Host choice Kestrel vs HttpListener: I will measure single-file size.

Risks: resampler quality affects WER; chunk joins can cut words; winmm is legacy.
Builder plan: P1 WAV reader+resample+split on a copy of the library vs NAudio reference text; P2 waveIn 16 kHz capture; P3 server size Kestrel vs HttpListener + PrivateType-format request. Effort: ~1–2 days to implement all.

### astra @ 20261123153233552
OPENING — Keep model acquisition explicit with await model.EnsureInstalledAsync(), then using var engine = await SpeechEngine.LoadAsync(model); await engine.TranscribeWavAsync(path). Add the same WAV overload taking caller-owned Stream, read from current position, never close it, support nonseekable upload bodies. Bound encoded bytes and decoded 16kHz samples independently; reject excess rather than truncating. Wav.cs:5 contains only SampleRate. SpeechEngine.cs:42 already gates inference, but initial decode before admission on concurrent HTTP requests could multiply memory; use immediate single-operation admission before reading the body, holding it until inference really completes, even after disconnect. SpeechEngine.cs:58 runs to completion.

Use one in-memory stop-and-transcribe microphone recording session, auto-stop at model cap with observable completion; cancellation discards. PrivateType’s DefaultMicrophoneCapture.cs:28,87 collects 16kHz mono PCM; OfflineRecognizer.cs:238 constructs WAV in memory at inference boundaries. Keep VAD/prefix rewriting out of the core. Require actual microphone start/stop/drain verification on Windows.

Default exe or serve opens embedded upload page after binding literal loopback, prints URL, supports --no-browser; same-origin JS and curl send audio/wav bytes, return {"text":"..."}. Expose model state and explicit first-use install. README in order: try exe, curl, NuGet WAV, push-to-talk, contracts/troubleshooting. Small slices: install/WAV, file CLI, capture, HTTP/UI, published artifact smoke. Inference: moderate effort; input conversion and recorder lifetime dominate uncertainty.

### sol @ 20261123153305307
CLAIM — Two adapter-level checks need to precede implementation.

1. Existing NAudio.Core 2.2.1 is useful for WDL resampling, but WaveFileReader requires seekable input (WaveFileChunkReader.cs:48–61), and ToSampleProvider accepts only direct PCM or IEEE float tags (SampleProviderConverters.cs:14–50). Extensible WAV is returned as WaveFormatExtraData (WaveFormat.cs:188–193), so handle its validated PCM/float subtype explicitly or document exclusion. Define stream positioning/ownership, duration/sample bounds and malformed/truncated data handling before relying on the reader. https://github.com/naudio/NAudio/tree/v2.2.1/NAudio.Core
2. ASP.NET Core's ordinary form buffering spills uploads over 64 KiB to disk. To promise no audio files, use raw audio/wav or MultipartReader with a strict byte cap and one admitted upload. If pairing multipart with a seekable WAV reader, explicitly buffer bounded bytes in memory. https://learn.microsoft.com/en-us/aspnet/core/mvc/models/file-uploads

Separately, library UI lifecycle needs a focused check: SpeechEngine.cs:44–54 captures SynchronizationContext while holding the semaphore that Dispose waits on (64–76). Closing a UI on that context during inference can deadlock. Library awaits under the gate should use ConfigureAwait(false), with deterministic testing of UI-context disposal.

### astra @ 20261123153310320
BUILD — Sol’s multipart choice fits the reference client better than my raw body proposal. OfflineRecognizer.cs:241–251 sends file, response_format=json (preview) or verbose_json (final), and automatic_punctuation=true. ParseTranscript:264–278 requires only string text and tolerates missing words. RecoverMissingEndAsync:198 gates recovery on a non-null last word end. So multipart WAV and text can serve PrivateType’s basic transcription at wire level, but omitted timings disable its tail recovery. Do not promise full OpenAI or PrivateType compatibility without a launched-client integration test (PrivateType additionally manages its own process). Recommend multipart only, common {text} response; supporting PrivateType-specific extras is optional, name supported semantics explicitly. Reject long audio for v1: ParakeetModel.cs:163 already does; silence splitting needs corpus evidence on word boundaries and punctuation before shipping.

### fable @ 20261123153323336
OPENING — fable

**Approach: one package, no new dependencies, three additions to the library; the exe becomes "serve + page" with an OpenAI-compatible endpoint.**

Library (all 16 kHz mono float underneath, one cutter shared by file and microphone):
```csharp
var model = SpeechModel.Default();
await model.EnsureInstalledAsync();                         // new: no-op when installed, owns its HttpClient
using var engine = await SpeechEngine.LoadAsync(model);     // + optional SpeechEngineOptions { Device }, engine.Backend
Console.WriteLine(await engine.TranscribeFileAsync("hello.wav"));   // new: any WAV rate/channels/length

await foreach (var utterance in engine.TranscribeLiveAsync(Microphone.ListenAsync()))   // new
    Console.WriteLine(utterance.Text);
```
- `TranscribeFileAsync`: managed RIFF reader (PCM 8/16/24/32, float, EXTENSIBLE), downmix, windowed-sinc resample, then cut at pauses into pieces under the limit and join. Today a long clip throws (ParakeetModel.cs:163).
- `TranscribeLiveAsync(IAsyncEnumerable<float[]>)` yields `Utterance(Start, Duration, Text)`; the cutter is LocalBuddy's UtteranceCutter.cs (pause 700 ms, max 20 s). Any source plugs in (loopback, network).
- `Microphone`: WASAPI through raw vtable calls into ole32/mmdevapi, the way D3D12Native.cs:253-274 already calls COM. Shared mode + AUTOCONVERTPCM lets Windows deliver 16 kHz mono, as PrivateType does (DefaultMicrophoneCapture.cs:25-29). Keeps the "only DLLs every Windows ships" rule. Push-to-talk: `Microphone.StartRecording()` → `Stop()` returns `float[]`.

Fact for the owner: PrivateType does not write WAV files. It buffers PCM in memory while the key is held and POSTs an in-memory WAV to `http://127.0.0.1:{port}/v1/audio/transcriptions` (OfflineRecognizer.cs:11-12, 241-249; EngineHost.cs:33).

Exe:
- `parakeet-runtime` with no arguments (double-click): installs the model with progress, serves on 127.0.0.1, opens the page. `parakeet-runtime <file.wav>` prints text. `serve [--port] [--device] [--no-browser]`, `listen` (microphone). Benchmarks keep their flags.
- `POST /v1/audio/transcriptions` (multipart `file`, `response_format` json|text|verbose_json with utterance segments): curl, the OpenAI SDKs and PrivateType work unchanged (inference: PrivateType tolerates missing `words`, OfflineRecognizer.cs:198). `GET /health` gives version, backend, model.
- Page: one embedded HTML file. The browser decodes any audio format and resamples to 16 kHz WAV (Web Audio), and can record the microphone (localhost is a secure context), so the server stays WAV-only. It shows text, audio length, time taken, and the curl/C# call it just made.
- Kestrel (ASP.NET Core minimal API) in the Cli project only, not the library.

Trade-offs and risks:
- Hand-written WASAPI and resampler are code we own; the alternative is NAudio as a package dependency or a second package.
- Kestrel grows the 75.7 MB exe (artifacts/release/0.0.0); HttpListener avoids that but needs a hand-written multipart parser. Needs a measurement.
- Cutting at pauses may cost accuracy at cuts; needs a WER check on long60/long120.
- No-argument behaviour changes from "print usage" (Program.cs:18-28).

Effort (inference): library 3-4 days, exe 2 days, README half a day.

### astra @ 20261123153353447
CLAIM — A managed reusable WASAPI wrapper can keep the existing plain net10.0 target and Windows SDK projection constraint. Inspect ~/.nuget/packages/naudio.wasapi/2.2.1/naudio.wasapi.nuspec and lib/: netstandard2.0 asset, only dependency NAudio.Core 2.2.1, itself netstandard2.0 with no dependency. Both MIT. PrivateType already exercises that API (DefaultMicrophoneCapture.cs:25–31). Package decision is managed dependency cost vs maintenance of COM or waveIn correctness, not a necessary Windows TFM or second signed package. Favor a tiny public microphone recording helper in the existing package using the pinned narrow WASAPI component if warranted; a built and hardware-verified no-dependency implementation could change that preference. Already hosted app capture can pass its existing float[] directly. Defer continuous VAD/utterance timestamps to their own design pending quality and backpressure measurements.

### sol @ 20261123153426921
CLAIM — Keep the v1 file method's existing whole-utterance semantics and expose the exact model limit. Evaluate long-form segmentation as a separate, evidence-gated policy. LocalBuddy’s UtteranceCutter.cs:13–17 uses a 20 s live target, an RMS threshold and minimum voiced frames. Push adds only complete 30 ms frames (25–35), and Flush never consumes pending samples (38–43), so directly reusing it loses the final fractional frame and can suppress a short valid utterance. Those policies may suit live captions, but they should not silently change ordinary WAV output. Published measurements do justify testing future segmentation: docs/benchmarks.md:443–459 reports substantial English sentence loss on longer FLEURS concatenations. Above-limit rejection and measurable short-file parity give the first release a precise contract while leaving space for a better long-form policy.

RE PrivateType, EngineHost.cs:169–170 launches an existing engine with its own flags, and OfflineRecognizer.cs:118–121/156–161 can leave a preview in flight when final dictation starts. Wire-format overlap is useful, but a one-operation server can reject an overlapping final request. Avoid describing current PrivateType as drop-in compatible.

### astra @ 20261123153457037
BUILD — Sol’s segmentation concern is measurable. I read the installed pinned 713975456-byte GGUF tensor directory: encoder.pos_enc.pe dimensions (1024, 9999). ParakeetModel.cs:75,135–138 gives MaxAudio = ((9999+1)/2)*8/100−1 = 399 seconds (6 min 39 s). Thus long60/long120 exercise *below*-limit quality, never an over-limit split; use an over-399s synthetic or licensed case for that acceptance check, and keep a separate test for actual speech quality. Generic code should get the runtime property. The LocalBuddy cutter’s 35 s model-bound comment does not apply to this model. The v1 cap accommodates most push-to-talk recordings. At 48kHz stereo PCM16 a 399s WAV has 76,608,000 data bytes, so size support and cap must be chosen together and published (e.g. 128 MiB input or a deliberately lower documented transport bound).

### fable @ 20261123153458027
ROLES — openings are in from fable, opus, sol, astra (neon: post yours when you arrive).

- **opus, builder**: (1) WAV reader + downmix + resampler, compared by transcript against NAudio's WDL on 44.1/48 kHz stereo copies of corpus clips; (2) server host: single-file exe size and start time, Kestrel vs HttpListener, with a multipart upload that never touches disk (sol's point). Skip the waveIn prototype unless my WASAPI probe below is rebutted.
- **fable, co-builder**: microphone capture, and the long-audio measurement (whole call vs cut at pauses, WER on long60/long120).
- **sol, consultant**: sources for the server contract (Host/Origin checks, body limits, admission, disconnects) and the UI-thread Dispose deadlock; say exactly which lines of each prototype break your rules.
- **astra, verifier**: rerun each measurement posted here from its script and report match/mismatch; check every "PrivateType works with it" claim against PrivateType's code.
- **neon, tester**: test each prototype against the brief's "Good looks like": count hello-world lines, compile the README snippets, try the wrong inputs a new user would pass (mp3, 48 kHz stereo, 10-minute file, no microphone).

Corpus for measurements: `%USERPROFILE%\.councils\local-parakeet-engine\fable\corpus\<lang>\{wav16,long60,long120}` with `dev.tsv`. The model is installed. Build against the already-built `src\ParakeetRuntime\bin\Release\net10.0\ParakeetRuntime.dll`; an assembly named `parakeet-runtime` sees its internals (ParakeetRuntime.csproj InternalsVisibleTo), so nothing in the repository is touched.

### fable @ 20261123153458225
CLAIM — two measurements, both rerunnable from `fable\proto` (`dotnet build -c Release`, then `dotnet bin\Release\net10.0\parakeet-runtime.dll <command>`).

1. `mic 3`: raw-vtable WASAPI capture, 95 lines, no package (WasapiProbe.cs). Output: "captured 48127 samples = 3,01 s of 16 kHz mono float in 301 packets over 3 s wall". Windows did the conversion (AUTOCONVERTPCM, float32 mono requested directly), so capture needs no resampler and no NAudio. Against astra's NAudio.Wasapi preference: that adds two package dependencies for what is 95 lines here. Against opus's waveIn: not needed. Not yet covered: device list, unplug during capture, "no microphone" error text.

2. `max`: "MaxAudio=399 s". So "reject beyond the model limit" (sol, astra) accepts a 60 s or 2 min WAV and transcribes it whole. docs/benchmarks.md:457-459 measured that: English WER 5.3% at ≤ 20 s, about 31% at 60 s, about 43% at 2 min, "model drops sentences after pauses"; Polish 5.7% → 8.8% → 9.9%. PrivateType works around the same loss (OfflineRecognizer.cs:18-19). So the whole-utterance file contract is the one that silently returns bad text; the limit that matters for quality is near 20 s, not 399 s.

My position: `TranscribeFileAsync` must cut at pauses. Sol is right that LocalBuddy's cutter drops the tail fraction (UtteranceCutter.cs:38-43), so it gets rewritten, not copied.

WORKING — measuring long60 and long120 English, whole call vs cut at pauses (≤ 20 s pieces), WER against dev.tsv.

### sol @ 20261123153511721
CLAIM — The pinned public model’s duration cap is exactly 399 s (6,384,000 normalized samples). I read the installed GGUF header with sol/Get-PinnedModelDuration.ps1. Output: bytes=713975456, encoder_width=1024, position_tensor=1024x9999, rows=9999. Applying ParakeetModel.cs:138 gives 399, independently of the audio device. The LocalBuddy cutter’s old “about 35 s” comment does not describe this pinned model.

This supports making above-limit splitting a follow-up to v1, with the separate WER question for 60–120 s audio still open. The server must advertise independent encoded-byte and audio-duration limits. A 399 s stereo 48 kHz float WAV is 153,216,000 bytes plus header, while a mono 16 kHz PCM16 WAV at the same duration is 12,768,000 bytes plus header. Pick the byte cap knowingly and make both failures actionable.

### neon @ 20261123153538230
OPENING — Zero-friction consumer API, in-memory mic capture, and OpenAI-compatible local server + UI

## 1. Key Findings from the Codebase

- **How PrivateType captures the mic (resolving the brief's question):** It does **not** write WAV files to disk (`OfflineRecognizer.cs:11-12`). `DefaultMicrophoneCapture.cs:25-29` uses WASAPI shared mode with `AUTOCONVERTPCM` (`WaveFormat(16_000, 16, 1)`) so Windows resamples to 16 kHz mono PCM16 in 100 ms chunks. `OfflineRecognizer.cs:42-43, 59-67, 234-260` buffers those chunks in memory (zeroing them on dispose), wraps them in a 44-byte in-memory RIFF header (`OfflineRecognizer.cs:284-315`), and POSTs them to a local HTTP `/v1/audio/transcriptions` server on release. `LocalBuddy` (`AudioSource.cs:34, 50-51`, `UtteranceCutter.cs:11-79`) instead downmixes/resamples `WasapiCapture` to `float[]` and cuts at pauses (≤20 s) to call `SpeechEngine` directly.
- **Engine internals already ready to surface:** `ParakeetModel.cs:37, 138-142` already has `allowGpu`, `adapterNameFilter`, `MaxAudio`, `IsUsingGpu`, and `BackendName`—none are exposed on `SpeechEngine.cs:7-83`.
- **Constraint to preserve:** `ParakeetRuntime.csproj:1-29` and `AGENTS.md:20-23` keep the library plain `net10.0` with `[SupportedOSPlatform("windows")]`, no Windows SDK projection, and only inbox Windows DLL P/Invokes (`d3d12.dll`, `dxgi.dll`, `d3dcompiler_47.dll`, `kernel32.dll`).

## 2. Proposed Design

### A. Library Public API (`ParakeetRuntime`)

Keep the surface tiny (extending `SpeechModel` + `SpeechEngine` + 2 new types: `SpeechEngineOptions`, `MicrophoneRecording`):

```csharp
// SpeechModel additions (keep existing DownloadAsync(HttpClient, ...) for DI/tests)
public Task EnsureInstalledAsync(IProgress<double>? progress = null, CancellationToken cancellationToken = default);

// SpeechEngineOptions
public sealed record SpeechEngineOptions(bool AllowGpu = true, string? GpuAdapterFilter = null);

// SpeechEngine additions
public static Task<SpeechEngine> LoadAsync(SpeechModel? speechModel = null, SpeechEngineOptions? options = null, CancellationToken cancellationToken = default);
public bool IsUsingGpu { get; }
public string BackendName { get; }
public TimeSpan MaxAudio { get; }
public Task<string> TranscribeAsync(ReadOnlyMemory<float> samples16KhzMono, CancellationToken cancellationToken = default); // plus existing float[]
public Task<string> TranscribeFileAsync(string wavPath, CancellationToken cancellationToken = default);
public Task<string> TranscribeWavAsync(Stream wavStream, CancellationToken cancellationToken = default);
public MicrophoneRecording StartMicrophone(string? deviceId = null); // or MicrophoneRecorder.Start()

// MicrophoneRecording : IAsyncDisposable
public TimeSpan Duration { get; }
public float PeakLevel { get; } // for UI meters like PrivateType
public Task<float[]> StopAsync(CancellationToken cancellationToken = default);
public Task<string> StopAndTranscribeAsync(CancellationToken cancellationToken = default);
```

- **Hello world (WAV, 4 lines):**
  ```csharp
  var model = SpeechModel.Default();
  await model.EnsureInstalledAsync();
  using var engine = await SpeechEngine.LoadAsync(model);
  Console.WriteLine(await engine.TranscribeFileAsync("hello.wav"));
  ```
- **Hello world (Mic, push-to-talk like PrivateType):**
  ```csharp
  await using var rec = engine.StartMicrophone();
  Console.ReadLine();
  Console.WriteLine(await rec.StopAndTranscribeAsync());
  ```
- **Built-in WAV decoding (zero new deps):** Pure C# RIFF/WAVE reader in `ParakeetRuntime` (`Wav.cs` is currently 6 lines) supporting PCM 8/16/24/32-bit, IEEE float32, `WAVE_FORMAT_EXTENSIBLE`, N→1 mono downmix, and resampling to 16 kHz. Keeps `ParakeetRuntime` dependency-free beyond `System.IO.Abstractions`.
- **Built-in microphone capture (zero new deps vs NAudio):** Implement in-memory capture via inbox Windows DLLs—either `winmm.dll` (`waveInOpen` with `WAVE_MAPPER` requesting 16 kHz mono PCM16, ~80 lines, no COM/MTA issues on WPF/WinForms STA threads) or `mmdevapi.dll` WASAPI COM vtable (`AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | SRC_DEFAULT_QUALITY`, matching `D3D12Native.cs` and `ProcessLoopbackCapture.cs:154-169`). Inference: `winmm` is dramatically simpler and STA-safe, while WASAPI gives modern device IDs and better resampling quality; we should prototype both and compare. Audio stays strictly in memory and is zeroed on dispose.
- **Open question for council:** What should happen when a WAV or mic recording exceeds `engine.MaxAudio`? Auto-split on quiet frames vs throw `ArgumentException` (with optional opt-in chunking)?

### B. The Exe (`parakeet-runtime.exe`)

Add three consumer commands alongside existing benchmarks (`Program.cs:1-28`):
1. `parakeet-runtime transcribe <file.wav> [--device auto|cpu|gpu|gpu:<name>] [--model <path>]` — one-shot CLI transcription to stdout.
2. `parakeet-runtime download [--model-dir <path>]` — downloads & verifies `SpeechModel.Parakeet` with progress.
3. `parakeet-runtime serve [--port 8080] [--host 127.0.0.1] [--device ...] [--no-browser]` — starts the local HTTP server + opens the browser to `http://127.0.0.1:<port>/`. If the model isn't installed, downloads it with progress (CLI + UI status).
   - **Endpoints** (intentionally matching `CorpusBench.cs:254, 288` & `OfflineRecognizer.cs:241-254` / OpenAI Whisper API so existing clients and `curl` work out of the box):
     - `GET /` — embedded single-file HTML/CSS/JS UI (100% offline): drag-and-drop WAV upload, browser microphone record button (Web Audio → WAV → same API), live status (backend, RTF, elapsed ms), copy button.
     - `GET /ready` (and `GET /health`) — `200` with `{ "status": "ready", "backend": "...", "maxAudioSeconds": ... }` or `503` while downloading/loading.
     - `POST /v1/audio/transcriptions` — `multipart/form-data` (`file`, optional `response_format=json|text`) returning `{"text": "..."}` (plus `duration_seconds`, `elapsed_ms`, `backend` or plain text); also accept raw `audio/wav` body for simple `curl --data-binary`.
   - **HTTP host choice:** `Microsoft.NET.Sdk.Web` (Kestrel) vs `HttpListener` vs minimal `TcpListener`. Inference: `HttpListener` can hit Windows URLACL issues on `127.0.0.1` vs `localhost` without elevation, whereas Kestrel works cleanly on `127.0.0.1:0` or `:8080` and parses multipart natively, at the cost of single-file exe size. We should measure the exact single-file publish size delta before deciding.
   - **Security:** Loopback-only (`127.0.0.1`) by default, strict request body cap (e.g., 100 MB), reject non-loopback `Host` headers / cross-origin writes by default.

### C. README Outline

1. One-paragraph intro + badges.
2. **Quick start: try it with `parakeet-runtime.exe`** (`serve` UI + `curl` `/v1/audio/transcriptions` + `transcribe clip.wav`).
3. **Quick start: use the NuGet package** (`dotnet add package ParakeetRuntime`, 4-line WAV snippet, 4-line mic snippet, short note on `SpeechEngineOptions`, raw `float[]`, and offline `InstallFromFileAsync`).
4. **Downloads**, **Performance table** (dated, from `docs/benchmarks.md`), **The speech model**, **Verify a download**, **Requirements**, **Help**, **License** (keeping absolute GitHub URLs for nuget.org).

### D. Implementation Order

1. `SpeechModel.EnsureInstalledAsync`, `SpeechEngineOptions`, `BackendName`/`IsUsingGpu`/`MaxAudio` + unit tests.
2. Pure-C# WAV decoder & resampler + `TranscribeFileAsync`/`TranscribeWavAsync` + unit tests (formats, channels, rates, bad headers).
3. In-memory microphone capture (`StartMicrophone` / `MicrophoneRecording`) + unit tests for buffering/level + device-gated test.
4. `parakeet-runtime` `download`, `transcribe`, `serve` (API + embedded UI) + HTTP integration tests.
5. `README.md` + `docs/` updates + full verification (`dotnet test`, `Build-Release.ps1`, `Test-ReleaseArtifacts.ps1`, transcript parity & speed check).

## 3. Trade-offs, Risks, Effort

- **Trade-offs:** Keeping `ParakeetRuntime` free of NAudio avoids transitive dependencies and keeps the NuGet package ultra-clean, at the cost of writing ~200–300 lines of WAV + resampler + Win32/WASAPI capture code (which we must test rigorously, especially resampling quality vs WER).
- **Risks:** (1) Resampling quality affecting WER on 44.1/48 kHz WAVs (testable against FLEURS resampled clips); (2) `HttpListener` URLACL vs Kestrel single-file exe size (measurable in 2 minutes); (3) Single-file publish + Certum signing compatibility (`Test-ReleaseArtifacts.ps1:34` expects exact 4 files in the zip folder—embedded HTML resource preserves this).
- **Effort:** Medium (~3–4 focused commits/PRs across the 5 steps).

### astra @ 20261123153612363
CONCEDE — Fable’s mic and max reruns: max matched exactly (399 s, CPU AvxVnniInt8). mic 3 under sandbox returned HRESULT 0x80070005 at WasapiProbe.cs:52, a concrete denied-access case. With approved normal desktop access it captured 48,287 samples, 302 packets, 3.02 s; RMS/peak zero, so timing and format matched, intelligibility unverified. I accept native WASAPI as an implementation option, contingent on dedicated capture-thread COM lifetime, checked HRESULT, balanced CoUninitialize, cleanup after partial initialization, reliable stop/drain and user-facing access-denied handling. Probe lines 39 and 46–80 only demonstrate happy-path device access.

PrivateType compatibility audit: EngineHost.cs:50 expects engine/bin/nemo-speech.exe; :169–170 launches serve --host ... --threads 1 --no-ui --asr-model ... --device ...; :196 runs --json doctor; :308–315 requires GET /ready. OfflineRecognizer.cs:198,241–278 confirms optional words and no recovery if missing. Existing PrivateType cannot use the proposed exe unchanged. Document similar push-to-talk/wire behavior only; do not commit this project to emulating process launch, health probes or timed words.

### sol @ 20261123153617371
CLAIM — Server contract I’ll check against the host prototypes.

Bind Kestrel with Listen(IPAddress.Loopback, port), read the actual address after StartAsync, then print/open only that URL. Replace Kestrel’s config loader with an empty configuration so ambient Kestrel__Endpoints/appsettings cannot add listeners. Microsoft documents that configured endpoints are cumulative with Listen: https://learn.microsoft.com/en-us/aspnet/core/fundamentals/servers/kestrel/endpoints?view=aspnetcore-10.0

Validate the exact numeric Host and port for every route. Before any POST work, require exact same-origin Origin when present, reject null/foreign origins, and require a custom header such as X-Parakeet-Request: 1 for browser/API calls. Keep CORS disabled. The custom header guards against simple cross-origin multipart submissions, including requests missing Origin; ordinary curl adds one flag. This is protection against web pages, not authentication against other local processes. https://fetch.spec.whatwg.org/#cors-safelisted-request-header

Before reading a body, take an immediate single-operation lease (503 busy), impose total body bytes plus MultipartReader per-section/header/count caps and an upload deadline. Accept exactly one named file; ignore its filename. Return a stable error code distinguishing byte limit (413), malformed WAV (400), unsupported input (415), and decoded duration (422). Cancel upload/pre-inference work on disconnect (RequestAborted), but await started inference to completion before releasing the lease; SpeechEngine.cs:44–58 deliberately cannot interrupt it. No disk upload buffering, transcript logging, or response caching. https://learn.microsoft.com/en-us/dotnet/api/microsoft.aspnetcore.webutilities.multipartreader.bodylengthlimit?view=aspnetcore-10.0

### opus @ 20261123153720727
CLAIM — P1 measured (RTX 5070 Ti GPU, pinned Q8_0 model, FLEURS en_us, 2026-11-23). Rerun: opus\proto\p1: `dotnet build -c Release; dotnet run -c Release --no-build -- 30`. Output: opus\proto\p1\p1-run1.txt.

1. Own WAV reader + Kaiser windowed-sinc resampler (opus\proto\lib\WavAudio.cs, ~130 lines, no package) equals NAudio WDL by WER on 30 wav16 clips (594 words):
- 16k mono ours 4.04% = NAudio 4.04%
- 48k stereo float32: ours 3.70% = NAudio WDL 3.70%
- 44.1k stereo pcm16 ours 3.70%; 8k mono ours 4.04%
Cost: mine is naive (180 ms per ~10 s clip at 48k vs WDL 7 ms); a polyphase table fixes that, not a design issue. So no NAudio needed in the library for WAV.

2. Long audio (long120, 124–129 s, all under the 399 s limit), WER:
| clip | whole call | quiet-split 20s | quiet-split 30s | blind-split 30s |
| 00 | 65.2% | 4.8% | 4.4% | 11.0% |
| 01 | 28.0% | 3.7% | 2.2% | 13.3% |
| 02 | 14.8% | 6.8% | 3.2% | 1.3% |
| 03 | 49.4% | 9.7% | 14.3% | 13.9% |
Quiet split = cut at the lowest-energy 100 ms window in the last 25% before the limit (opus\proto\lib\LongAudio.cs, 40 lines). This confirms fable: whole-call on 2-minute audio is badly wrong, so "accept up to 399 s and transcribe whole" (sol, astra) silently returns bad text. REBUT sol/astra 20261123153426921/153310320: v1 TranscribeFileAsync should cut at pauses; the cutter is small and evidenced. Clip 03 shows the naive cutter is not final (a VAD-aware cut beats lowest-energy); I'll try a pause-length cutter next.

WORKING — server prototypes: Kestrel vs HttpListener single-file size, start time, multipart never on disk.

### sol @ 20261123153733867
CLAIM — Deterministic disposal reproduction verified against the current SpeechEngine.cs source. Run sol/Test-Dispose.ps1. It compiles the unedited source with an inference stub held until the test releases it, and repeats with ConfigureAwait(false) at both awaits on lines 44 and 49. Original output: “worker finished; UI continuation queued; Dispose blocked”; disposal finishes only after the test manually dispatches the UI callback. Fixed output: “Dispose completed; queued UI callbacks=0” with no pump. Both builds had zero warnings. In a single-threaded GUI, synchronous Dispose on that blocked context cannot dispatch the callback itself. Include those await changes with the first consumer-library slice and retain a deterministic bounded regression. If the WAV API adds awaits while holding this gate, apply the same rule. The probe demonstrates scheduling, independent of recognizer speed or hardware. https://devblogs.microsoft.com/dotnet/configureawait-faq/

### fable @ 20261123153807368
CLAIM — measured: cutting long audio at its quietest point is not a risk to accuracy, it is the fix. Script: `fable\proto\run-long.ps1 -Language en_us|pl_pl` (LongAudio.cs; RTX 5070 Ti, FLEURS concatenations, WER counted as CorpusBench does; outputs in `fable\proto\results-*.txt`, transcripts in `out-*.tsv`).

| Corpus | whole call (today) | quietest point, ≤ 20 s | quietest, ≤ 30 s | every 700 ms pause, hard cut at 20 s |
| --- | --- | --- | --- | --- |
| EN 60 s | 32.24% | **6.19%** | 6.69% | 12.07% |
| EN 2 min | 43.38% | **5.99%** | 7.19% | 13.77% |
| PL 60 s | 8.86% | **6.02%** | 6.63% | 10.53% |
| PL 2 min | 9.92% | **6.14%** | 6.70% | 10.48% |

Short-clip baseline is 5.25–5.76% (benchmarks.md:415-417). "Whole" reproduces benchmarks.md:455-459.

What follows:
1. Against sol and astra ("reject above the limit, keep whole-utterance semantics for v1"): the limit is 399 s, so that contract returns 32% wrong words for a one-minute English WAV without any error. A file method that does not cut is the unsafe one.
2. The cut policy: greedy, quietest 300 ms window in the second half of each 20 s span; no loudness threshold, nothing dropped, every sample transcribed. 30 lines (LongAudio.cs CutAtQuietestPoint).
3. Against my own opening: LocalBuddy's policy (fixed 0.01 threshold, hard cut at 20 s) is twice as bad. The live cutter needs the same quietest-point fallback and must not ship as a copy.
4. The raw `TranscribeAsync(float[])` should apply the same cutting above 20 s, or its documentation must state the loss.

### sol @ 20261123153810273
CONCEDE — Opus’s four long120 comparisons changed my position. Whole-call WER of 14.8–65.2% versus 3.7–9.7% with 20 s quiet splitting on those same clips justifies a measured long-file policy in v1. Keep TranscribeAsync(float[]) as one explicit utterance and preserve exact normalized short-file parity. TranscribeWavAsync can segment long files internally and return joined text. Describe 399 s only as the native inference ceiling and document an independent WAV/HTTP resource limit. The acceptance check should cover no sample gaps, contiguous speech with no quiet window, a non-frame-aligned tail, cancellation between segments, and FLEURS en_us plus pl_pl quality. Avoid advertising the quiet-cut audio offsets as word timings.

I also accept an opt-in bounded WASAPI recording helper after the measured probe. Keep native capture lifetime on its own thread, and make its terminal result observable at stop, cap, or device error. In WasapiProbe.cs, line 39 needs checked COM initialization and same-thread uninitialization; lines 46–55 need partial cleanup; lines 70–75 need a finally for every acquired audio packet. These are implementation checks, not a rejection of the prototype. https://learn.microsoft.com/en-us/windows/win32/api/audioclient/nf-audioclient-iaudiocaptureclient-getbuffer

### astra @ 20261123153826680
CONCEDE — Opus P1 replayed from compiled p1.dll 30 with freshly regenerated input variants in astra/opus-p1. All 6 aggregate short WERs and all 4 long120 × 2 split limits × 3 methods matched p1-run1.txt to printed precision. E.g. long120-00 whole 65.20%, quiet 20s 4.76%, quiet 30s 4.40%; long120-03 whole 49.42%, quiet 20s 9.65%. I now support segmenting long file inputs by default, retaining exact short-file parity and the low-level whole-utterance float[] method. Do not advertise 399 s as a max file duration where segmentation is active. Preserve every sample, bound decode/upload memory independently, and check another language plus continuous speech before freezing cut parameters. Capture stopping and long microphone uploads should use the same file-quality policy where practical.

48k stereo float ours vs WDL both 22/594 wrong, ~164 vs 7 ms/clip; ours included generating variants in my fresh cache, though not in every previously warmed run. Equal aggregate WER does not prove identical transcripts. Sol’s own pinned-model script independently reran exactly too.

### fable @ 20261123153850367
POINTS — after exchange 1 (all five posted).

1. `SpeechModel.EnsureInstalledAsync(progress = null, ct = default)`. **agreed** (all).
2. `LoadAsync(model, options)` with GPU on/off + adapter filter; engine exposes backend name. **agreed** in substance (fable, opus, neon; sol/astra silent). Names open.
3. WAV reader + downmix + resampler written in the library, no NAudio. opus measured equal WER to NAudio WDL. **open**: sol, astra to accept or rebut. Stream overload beside the path overload: **agreed**.
4. Long audio in the file method: cut at the quietest point, ≤ 20 s, joined text. **agreed** (fable, opus measured; sol, astra conceded). Conditions from sol/astra accepted: no sample dropped, short-file parity, tests for no-quiet-window speech, odd tail, cancellation between pieces, EN + PL.
5. Raw `TranscribeAsync(float[])`: stays one whole utterance (sol, astra) vs also cuts above 20 s (fable). **open**.
6. Microphone capture in the library, native WASAPI on its own thread: fable, astra, sol (opt-in, with lifetime conditions) vs winmm (opus, neon as option). **open** for opus, neon.
7. Continuous microphone transcription in v1 (fable) vs push-to-talk only (opus, astra, neon). **open**.
8. No-argument exe starts the server and page (fable, astra) vs explicit `serve` only (sol, opus, neon). `download` command (neon). **open**.
9. `POST /v1/audio/transcriptions`, multipart `file` → `{"text"}`. **agreed**. Raw `audio/wav` body too (neon) vs multipart only (astra, sol): **open**.
10. "PrivateType works unchanged" (fable). **dropped**: astra's audit (EngineHost.cs:50, 169-170, 308-315).
11. Required custom request header and 503-when-busy (sol) vs Origin check and queueing. **open**.
12. Kestrel vs HttpListener. **open**, waits for opus's measurement.
13. `ConfigureAwait(false)` under the engine's gate. **agreed** (sol reproduced it).
14. Page converts non-WAV audio in the browser (fable). **open**, no replies.
15. README order: exe first (astra, neon) vs library first. **open**.

### fable @ 20261123153850558
CONCEDE — two of my opening positions, and one REBUT.

Point 10: astra's audit convinced me (PrivateType starts `nemo-speech.exe serve --no-ui …` itself and needs `GET /ready`). The README says "same request shape as OpenAI's transcription API", nothing about PrivateType.

Point 7: my own table convinced me. The live policy I proposed (cut at every pause, hard cut at 20 s) scored 12–14% against 6% for the offline cut. Continuous transcription is a separate design with its own measurement; v1 ships push-to-talk, which is what the brief names ("the way PrivateType does"). So v1 microphone = start, stop, get text.

REBUT sol 20261123153617371, point 11, the required `X-Parakeet-Request` header. Evidence against:
- It breaks the reason for choosing this route: `openai` SDK clients with `base_url=http://127.0.0.1:…/v1` and `curl -F file=@a.wav` stop working without an extra flag.
- The threat it adds protection for is a cross-site form POST. A page cannot read the answer (no CORS), so the damage is one transcription of the attacker's own audio. Browsers send `Origin` on every cross-origin POST, forms included (Fetch standard, "serializing a request origin"; Chrome and Firefox both do), and `Sec-Fetch-Site`. Rejecting a present, non-matching `Origin` covers it; non-browser clients send none.
Keep: literal loopback bind, exact Host check (DNS rebinding), no CORS, byte caps, no disk.

On 503-when-busy: the engine already queues (SpeechEngine.cs:44). I propose queue up to a small bound, then 503; QUESTION sol: what breaks with a bounded queue?

### sol @ 20261123153910529
CLAIM — Opus’s prototypes settle host portability, but these lines are still production gates:

• p3k/Program.cs:9–10 inherits the default Kestrel config loader and binds localhost. For a numeric IPv4-only contract use Listen(IPAddress.Loopback, port) and k.Configure(new ConfigurationBuilder().Build()); verify actual bound addresses after startup. ASP.NET Core v10.0.0 WebHost.cs:224–230 installs the inherited loader. https://github.com/dotnet/aspnetcore/blob/v10.0.0/src/DefaultBuilder/src/WebHost.cs#L224-L230
• p3k:16–21 has no exact Host/Origin/custom-header guard or early ready/busy admission. :22–37 correctly avoids form-to-disk buffering, but currently permits duplicate file sections, reads response_format to the total body cap, and has no upload deadline. :25–27/33 parses untrusted formats without mapping expected exceptions; test-servers-run1.txt confirms bad WAV returns 500. :41 correctly forwards disconnect, but add a lease around :21–42 and retain it through actual inference completion. :41 also calls the single-utterance API; the final endpoint should call the same long-file pipeline as CLI.
• p3h/Program.cs:12–15 provides no explicit socket-bind proof. :37–41 checks ContentLength64 but does not cap bytes actually read when length is unknown. :67–80 scans bare boundary text without validating multipart delimiter framing. Prefer Kestrel’s existing parser and request-abort plumbing.
• index.html:17–19 sends no custom header, assumes JSON even on early server rejection, and recomputes WAV on the client. For v1 upload the WAV bytes as selected and let one server/library conversion own the behavior. :10 only reads health once; missing/loading/failed model states need actionable UI.

### sol @ 20261123153947178
CONCEDE — Point 3: Opus’s measured DSP parity, plus astra’s independent replay, supports the dependency-free WAV path if production avoids the prototype’s measured resampling overhead and validates input. WavAudio.cs:16 ignores RIFF length, :27 allocates from the raw fmt size, :32–33 only checks two bytes of the extensible subtype, :40 clips truncated data, and :48 seeks on nonseekable streams. Add checked limits before allocations, full subtype GUID and frame alignment validation, and block reads/skips for nonseekable streams. Compare transcript deltas and conversion latency, not aggregate WER alone.

Point 11: A bounded queue works if admission is before any body parsing and canceled/disconnected waiters are removed. I accept one active upload/inference plus one header-only waiting request with a finite deadline, then 503 busy. Readiness remains independent. I accept standard multipart/curl without a special header on transcription if exact Host, foreign/null Origin rejection, and Sec-Fetch-Site rejection for every present value except same-origin are verified in supported browsers. A distinct install POST has a larger side effect; require the custom header there if install is page-driven.

Points 5/8/9/14/15: preserve and document raw single-utterance behavior; explicit serve is sufficient, but I accept no args aliasing serve with --help retaining usage; multipart only; WAV-only browser upload using the same library path. Start README with the page and one curl example, then WAV and push-to-talk NuGet snippets.

### astra @ 20261123154002862
CLAIM — Points 2/3: accept engine options + backend property and managed dependency-free WAV codec, subject to Sol’s validation and latency gates. Point 5: preserve existing exact float[] method contract. New consumer file/recording calls share segmentation; a caller supplying its own captured samples should have a clearly named segmented path if needed. Point 8: support no-args opening page, explicit serve/headless, preserve --help. Points 9/11/14/15: I agree with Sol’s latest positions. API docs should say “multipart WAV → text”; any claim particular SDKs work requires a client test. Native mic helper is enough for v1; no browser-side re-encoding.

Independent rerun of sol/Test-Dispose.ps1 matched exactly on .NET 10.0.302, zero warnings: worker finished with UI continuation queued and Dispose blocked in original; fixed completed before dispatch. Retain its test. Now replaying fable/run-long.ps1 for all EN+PL combinations, copied under Astra’s notes to avoid overwriting results.

### neon @ 20261123154007907
CLAIM — Tester edge-case run (`neon\tester\Program.cs`) against `opus\proto\lib\WavAudio.cs` and `fable\proto\WasapiProbe.cs`:

1. **WAV reader (`WavAudio.Read16KhzMono`)**:
   - MP3 (ID3 and `0xFF 0xFB`) and empty stream threw `InvalidDataException`; caller stream stayed open (`CanRead=True`).
   - **5 failures to fix in production WAV reader:**
     - `JUNK`/`LIST` chunk on a non-seekable stream (HTTP upload): `NotSupportedException` (`WavAudio.cs:48` uses `Seek`). Must drain/skip forward.
     - No `data` chunk: `EndOfStreamException` instead of `InvalidDataException`.
     - `rate=0`: `OutOfMemoryException` (`WavAudio.cs:88` divides by `0`).
     - `bits=0`: `DivideByZeroException` (`WavAudio.cs:41`).
     - Streaming `data` size `0xFFFFFFFF` on non-seekable stream: `ArgumentOutOfRangeException: count ('-2')` (int overflow at `WavAudio.cs:41-42`).
   - **Performance/memory on 10-min 48 kHz stereo PCM16 WAV:** `11,559 ms` and `586 MB` managed delta (11.6 s just to resample—10× slower than RTX inference). Must validate bounds before allocating and optimize/polyphase the resampler (or compare NAudio WDL speed).
   - Also check `WAVE_FORMAT_EXTENSIBLE`: `WavAudio.cs:32-33` reads only 2 bytes of the 16-byte SubFormat GUID (`fmt.Length >= 40` + full PCM/IEEE_FLOAT GUID check needed).

2. **WASAPI probe (`WasapiProbe.cs`) on an STA thread (WPF/WinForms):**
   - `CoInitializeEx(MTA)` returned `0x80010106` (`RPC_E_CHANGED_MODE`), yet capture still succeeded (`8607 samples = 0.54 s in 54 packets`).
   - Still requires: dedicated MTA background thread, `try/finally` COM `Release` on error (`WasapiProbe.cs:45-80` leaks on throw), and mapping `0x80070490` (`E_NOTFOUND`) to a clear no-microphone exception.

### astra @ 20261123154101480
BUILD — Independent EN and PL rerun of fable/proto/run-long.ps1 is complete. All 16 configurations reproduced the posted WER. Stronger check: SHA-256 for all 16 transcript TSVs in astra/long matched their namesakes in fable/proto byte-for-byte. These are 23+12 English and 22+11 Polish public FLEURS concatenations, on the installed Q8_0 and RTX 5070 Ti. Endorse the second-half quietest 300 ms/20 s default as measured policy, with no dropped samples. For a strict advertised 20.000 s bound, check exact samples: LongAudio.cs:55,68–72 floors to complete 30 ms frames before deciding to split; 320,159 samples (~20.010 s) can pass whole. A tiny regression at exactly 320000 and 320001 samples protects short-file parity and the contract.

### sol @ 20261123154137193
BUILD — A self-contained Kestrel bundle can pass the existing exact-file requirement. My sol/Test-KestrelBundle.ps1 publishes Microsoft.NET.Sdk + FrameworkReference Microsoft.AspNetCore.App, retaining the CLI’s plain SDK. The project excludes the IIS-only aspnetcorev2_inprocess.dll from ResolvedFileToPublish after ComputeResolvedFilesToPublishList. The Windows ASP.NET runtime pack lists that DLL as native, which is why ordinary single-file publish leaves it beside the exe. The probe produced exactly one 103,310,869-byte exe and ran it successfully: “Kestrel host built and route registered.” It intentionally opens no socket. Add this targeted exclusion and an actual bound-route smoke test to the consumer release slice; keep the release’s exact-file check. No change to the engine NuGet project or signing workflow is needed for this approach. https://learn.microsoft.com/en-us/dotnet/core/deploying/single-file/overview

### fable @ 20261123154156716
BUILD — on my probe, answering sol's, astra's and neon's lifetime conditions: `fable\proto\Microphone.cs` (327 lines, no package) is the v1 shape. Rerun: `dotnet bin\Release\net10.0\parakeet-runtime.dll record 3` and `record 3 1.5`.

```csharp
using var recording = Microphone.StartRecording();        // optional device, optional longest duration (default 10 min)
Console.ReadLine();
Console.WriteLine(await engine.TranscribeAsync(recording.Stop()));   // float[] 16 kHz mono
```
Also `Microphone.Devices()` → `MicrophoneDevice(Id, Name)`, and `recording.Duration`, `.Level` (for a meter), `.IsRecording`.

What it does and the output that shows it:
- Capture runs on its own thread, which initializes COM, releases every interface in `finally`, and uninitializes on the same thread. An STA caller (neon's case) never touches COM.
- Device list: "Microphone (2 - MAYA U5) [{0.0.1.00000000}.{c7cd…}]" and two more.
- `record 3`: "duration 3,00 s … stopped: 47967 samples".
- Cap (`record 3 1.5`): "duration 1,54 s … recording False" then Stop returns 24607 samples. Nothing is lost at the cap; the caller sees it through `IsRecording`.
- Unknown device: "InvalidOperationException: That microphone is no longer connected." `E_NOTFOUND` on the default device says "Windows has no microphone to record from."
- A failure during capture is kept and thrown by `Stop()`. Dispose without Stop discards and clears the audio.

Not verified: unplugging during capture, and real speech (the room was silent).

Side finding for the README: 1.5 s of near-silence transcribed as "Yeah." The model invents a word on silence, so a recording with no speech is not guaranteed to give empty text.

Against winmm (opus, neon): it would add a second capture technique where WASAPI is measured working, and winmm device names are cut at 31 characters (WAVEINCAPS.szPname is 32 chars).

### fable @ 20261123154156941
CLAIM — point 5: one rule for every entry point, "audio longer than 20 s is transcribed in pieces cut at its quietest points". Sol and astra want `TranscribeAsync(float[])` to stay whole and a second, differently named method to cut. Evidence against two methods:
- The microphone snippet, and any app with its own capture, calls the short obvious name with a 60 s dictation and gets the 32% path. The better behaviour would hide behind the longer name.
- Nothing is released (brief, lines 8-9), so there is no contract to preserve. The one existing caller, LocalBuddy, cuts at 20 s itself (UtteranceCutter.cs:17), so it never crosses the threshold.
- At or below 320,000 samples the call is unchanged, byte for byte (astra's exact-boundary test stays).
- No measurement shows whole-call above 20 s winning: 60 s and 2 min lose badly, and the 4 FLEURS clips of 20–23 s give whole 28.6% / 10.2% vs cut 17.1% / 7.4% (EN 1 clip, PL 3 clips; too few to prove a gain, enough to show no loss; `fable\proto\over20-*.txt`).
Benchmarks keep the whole-call path through the internal `ParakeetModel.Transcribe`.
QUESTION sol, astra: which caller needs a public whole-call above 20 s?

CONCEDE — point 14: no conversion in the browser; one conversion, in the library (sol, astra). Other formats, if ever, go into the library so the API and the page gain them together.

Accept: 9 multipart only; 11 as sol restated it (no custom header on transcription, exact Host, foreign/null Origin and non-same-origin Sec-Fetch-Site rejected, one running + one waiting, then 503); 15 page and curl first, then NuGet.

New point 16, QUESTION all: when `serve` finds no model, does it (a) download at once with a console line naming size, source and folder, or (b) start, and let the page offer a Download button?

### neon @ 20261123154157122
CONCEDE — Aligning on open points and reporting a release-gate finding on `opus\proto\pub\p3k*`:

- **Point 5 (raw `float[]`):** Concede to sol/astra. Keep `TranscribeAsync(float[])` single-utterance; document the ~20 s quality drop and 399 s `MaxAudio` ceiling. File and microphone paths use the ≤20 s quietest-point split.
- **Point 6 (mic):** Concede to fable's raw-vtable WASAPI (`WasapiProbe.cs` + my STA test), with dedicated MTA thread, balanced COM cleanup, and friendly exceptions for `0x80070490` (no mic) and `0x80070005` (denied, astra's hit).
- **Point 8 (no-args exe):** Concede to fable/astra/sol: no-args starts server + opens page (`--version` still works for `Test-ReleaseArtifacts.ps1:40`, `--help` prints usage), plus `serve`, `transcribe <wav>`, `download`.
- **Point 9/11/15:** Concede to multipart-only `POST /v1/audio/transcriptions`, sol's security/admission rules (`Host`/`Origin`/`Sec-Fetch-Site`, no custom header on transcribe, bounded 1+1 queue), and README order (exe + curl → NuGet WAV → NuGet mic).
- **Point 12 (Kestrel release gate!):** Inspected `opus\proto\pub\p3k2` (`98.85 MB`, compressed `47.94 MB` vs `p3h` `70.44`/`35.99 MB`). Both `p3k` and `p3k2` emit an unsigned `aspnetcorev2_inprocess.dll` (IIS module, 0.37 MB) beside `p3.exe`, which **fails `scripts\Test-ReleaseArtifacts.ps1:34-38`** (allows only 4 files). Prefer Kestrel (`FrameworkReference Microsoft.AspNetCore.App`) **only if** build strips `aspnetcorev2_inprocess.dll` from publish output; `EnableCompressionInSingleFile=true` also keeps the exe smaller than today's 75.7 MB.
- **Point 14 (UI):** Upload `.wav` files untouched. Note: browser `MediaRecorder` outputs WebM/Opus, not WAV—so either encode mic PCM to WAV in JS (mic-only) or drop browser mic in v1.

### astra @ 20261123154249403
BUILD — Verified packaging and actual HTTP startup separately. Sol’s Test-KestrelBundle.ps1 copied to astra/package republished exactly one file, 103,310,869-byte Test-KestrelBundle.exe, and ran it successfully. Astra/hosts.ps1 replayed Opus’s published p3h and p3k normally on loopback: HTTP first observed 170/206 ms; ready 2225/2135 ms; both pages 200, 2759 bytes; 48 kHz stereo float multipart with PrivateType’s extra fields returned 208 transcript characters and 17.54 s on both. No files remained in ASPNETCORE_TEMP after each request. This last check alone does not detect transient disk writes; the MultipartReader + MemoryStream path is the stronger evidence. Endorse Kestrel over maintaining a multipart parser, with production guards Sol identified. The model load dominates the observed start-time difference.

### opus @ 20261123154249811
CLAIM — P3 server host measured (same machine, 2026-11-23). Both prototypes in opus\proto\p3h (HttpListener) and p3k2 (Kestrel minimal API, plain Sdk + FrameworkReference); rerun `opus\proto\test-servers.sh`, output test-servers-run1.txt, startup.txt.

Single-file self-contained exe size (Build-Release.ps1:18 flags):
| variant | exe | extra files |
| baseline (lib + NAudio.Core, no server) | 74.0 MB | — |
| HttpListener | 73.9 MB | none |
| Kestrel | 103.6 MB (+29.6) | aspnetcorev2_inprocess.dll (+web.config etc. with Sdk.Web) |
| HttpListener, EnableCompressionInSingleFile | 37.7 MB | none |
| Kestrel, compressed | 50.3 MB | aspnetcorev2_inprocess.dll |
HTTP up: ~310 ms vs ~345 ms (both; noise). Model ready ~2.2 s either way (GPU).

Facts:
- HttpListener binds http://127.0.0.1:port/ and http://localhost:port/ as a NON-admin user with no URL ACL (checked `net session` fails, `netsh http show urlacl` empty); only `+` fails (Access denied). Neon's URLACL risk does not hold for loopback.
- Neither spills to disk: Kestrel via MultipartReader straight into a bounded MemoryStream, HttpListener reads the body into memory; temp folder had 0 files after 6.7 MB and 4 MB uploads (sol's point is satisfied by not calling ReadFormAsync).
- PrivateType-shaped request (file + response_format=verbose_json + automatic_punctuation) of a 48 kHz stereo float WAV returns {"text":…} on both, 60 ms inference.
- Page with drop/upload and browser-mic record (Web Audio → 16 kHz WAV) served from an embedded resource: opus\proto\index.html.

Position: HttpListener. It keeps the exe the same size and the release folder's file set unchanged (Test-ReleaseArtifacts), at the cost of a ~30-line multipart parser over a bounded in-memory body (p3h\Program.cs Multipart). Kestrel's gains (robust multipart, Host filtering middleware) don't pay for +30 MB and an extra file. Also: EnableCompressionInSingleFile halves the shipped exe today, independent of this choice.

### sol @ 20261123154251796
CONCEDE — Point 5: I found no established caller that needs a public whole-call path above 20 s. SessionMemory.cs:30 is a benchmark; if it needs the native call contract, keep it on the internal path. Given the unreleased API and Fable’s 20–23 s evidence, I accept one high-level segmented contract for WAV, microphone and 16 kHz samples, with short parity and cancellation between segments. Keep the documentation clear that 20 s is a segmentation policy rather than an input limit.

Point 16: choose (b). Start the page immediately, show the pinned model source, exact size and target folder, and offer Download. A headless user can run the explicit download command before serve. Let the process own one install and expose progress through GET status, so refreshing a tab does not restart it. Guard the install POST with exact origin + a custom same-origin header; it can use substantial network and disk.

Microphone follow-up: fable/proto/Microphone.cs:86–105 checks the cap outside packet draining, then drains again unconditionally, matching the 24,607 samples after a 24,000-sample cap request. Declare packet-granularity or enforce an exact cap while reading and releasing each full native packet. :184 throws outside the worker’s catch and can terminate the process. :34/:77 wait indefinitely; package async startup/stop or verify bounded behavior on a blocked or unplugged device.

### sol @ 20261123154326088
REBUT — Opus’s host-size result is real, but HttpListener is more than a 30-line parser decision. On Windows its numeric prefix registers an HTTP.sys routing category, not an app-owned loopback socket. Microsoft says an IP-bound weak-wildcard prefix matches any Host on that interface and should not be relied on to enforce security; IPv4 127.0.0.1 plus exact Host still needs independent binding evidence. HTTP.sys listen configuration is shared machine-wide. Kestrel gives the process an explicit IPAddress.Loopback socket and atomic port 0 startup. https://learn.microsoft.com/en-us/windows/win32/http/urlprefix-strings#host-specifier-categories

HttpListenerRequest has no equivalent to Kestrel’s RequestAborted for an admitted request after the body is read, so disconnect-aware queue removal and cancellation between segments would need extra plumbing. Multipart correctness, finite body reads, and shutdown are also production code. https://learn.microsoft.com/en-us/dotnet/api/system.net.httplistenerrequest?view=net-10.0

The extra-file objection is resolved by my single-EXE proof, independently rerun by astra. Choose Kestrel and keep the targeted IIS-module exclusion. Treat bundle compression as a separate release option, gated on cold and warm startup measurements, rather than making it a prerequisite for the consumer flow.

### opus @ 20261123154327077
CONCEDE — points 6 and 12, plus one finding and two positions.

Point 6: fable's Microphone.cs (device list, cap, E_NOTFOUND text, COM on its own thread) is measured working; winmm's 31-char device names (WAVEINCAPS.szPname) are a real defect. WASAPI it is.

Point 12: Kestrel. My size argument is weak against the 714 MB model download (+30 MB is ~4% of what a user fetches), sol removed the extra DLL (Test-KestrelBundle.ps1), and a hand-written multipart parser is security code we would own. Pair it with EnableCompressionInSingleFile: Kestrel compressed = 50.3 MB, smaller than today's 74 MB. Cost measured (opus\proto\memory-compression.txt, idle, no model): private memory 15.1 → 51.9 MB, startup 342 → 350 ms — noise next to the 1.1 GB loaded model.

Finding for sol's Host rule: neither host rejects a foreign Host by itself. `curl -H "Host: evil.example:5093" http://127.0.0.1:5093/health` → 200 on HttpListener bound to 127.0.0.1, and 200 on Kestrel. The exact-Host check must be our code, with a test.

Point 5: I side with fable. The agreed mic snippet is `engine.TranscribeAsync(recording.Stop())`; a 60 s dictation through it gets the whole-call path (my long120: 14.8–65.2% vs 3.7–9.7% cut). Keep the 320,000-sample boundary test so short calls are byte-identical.

Point 16: (a) with no button: `serve` binds and opens the page at once, starts the download immediately with one console line (size, source, folder), and /health reports progress, which the page shows. No install POST, so no side-effect endpoint to guard.

WORKING — polyphase resampler + sol/neon's WAV-reader validation fixes, re-measured.

### neon @ 20261123154343024
BUILD — Tested `fable\proto\Microphone.cs` from an STA thread (`neon\tester\Program.cs`), plus Point 5 & Point 16:

1. **`Microphone.cs` STA test results:**
   - Passed: `Devices()` (3 endpoints), `Start`+`Stop`+double-`Stop` (`4607` samples, `IsRecording=False`), `Dispose`→`Stop` (`ObjectDisposedException`).
   - **4 fixes needed for production:**
     - Unknown/malformed device ID threw `COMException (0x80070057)` (`E_INVALIDARG`), not `InvalidOperationException` (`Microphone.cs:312` only checks `E_NOTFOUND`). Map both, plus `0x80070005` (`E_ACCESSDENIED`) → `UnauthorizedAccessException`.
     - Cap overshoot: `longest=0.5s` returned `8,767` samples (+767 / 48 ms over `8,000`) because `Keep` doesn't trim and line 88 drains again after hitting cap.
     - `longest <= TimeSpan.Zero` accepted silently (validate with `ArgumentOutOfRangeException`).
     - Dispose `stopRequested`, move `CoInitializeEx` inside `try` at `Microphone.cs:184`, and mark `OnItsOwnComThread` thread background.

2. **Point 5 — CONCEDE to fable (`20261123154156941`):** The mic hello world calls `engine.TranscribeAsync(recording.Stop())`. Without >20 s quiet-cut in `TranscribeAsync(float[])`, a 45 s dictation silently hits 32%+ WER. ≤320,000 samples stays bit-identical, >20 s improves everywhere measured, and `CorpusBench` keeps internal `ParakeetModel.Transcribe`. One rule everywhere beats a trap on the shortest method.

3. **Point 16 (missing model on `serve`):** Prefer **(b)** — start immediately and open the page showing size/source/folder + a **Download model** button (with custom header per sol) and live progress bar; pair with `parakeet-runtime download` and `serve --download` for headless CLI. Avoids silent 714 MB surprise downloads on double-click.

### astra @ 20261123154343361
CONCEDE — Point 5: fable asks which caller requires whole inference above 20 s. I cannot identify one: this package is unreleased, the benchmark's direct ParakeetModel path already exists, and the owner wants microphone samples to work with the obvious method. I independently reran fable’s borderline cases: for a 23.4 s EN clip whole 10/35 errors vs cut 6/35, for 3 PL clips whole 11/108 vs cut 8/108. Coupled with the 68 long fixtures, I support uniform public sample/WAV/microphone segmentation, preserving exact single-call behavior at ≤320,000 samples. Keep MaxAudio as an internal native limit or label it explicitly if exposed, not a public input cap.

Point 16: support (b) immediate page and explicit download with progress, plus headless download command. This preserves offline operation after provisioning and avoids a double-click initiating 714 MB of traffic unexpectedly. For headless serve with a missing model, print the download command and URL promptly; for one-shot transcription, document the same prerequisite.

### fable @ 20261123154441812
POINTS — after exchange 2.

1. `EnsureInstalledAsync`. agreed.
2. Engine options + backend property. agreed (astra accepted). Exact names: in my draft, coming next.
3. Managed WAV reader + resampler, no NAudio. **agreed** (sol, astra conceded), gates: neon's 5 failures and sol's validation list fixed, resampler made fast (11.6 s for a 10-minute file today), transcript and latency compared.
4. Long files cut at the quietest point, ≤ 20 s. agreed. New evidence: cut length 10 / 15 / 20 / 25 / 30 s gives EN 2 min 6.82 / 5.78 / 5.99 / 6.55 / 7.19% and PL 2 min 7.53 / 6.10 / 6.14 / 6.42 / 6.70% (`fable\proto\results-threshold.txt`), so 15–20 s is the flat best and 20 s stays. astra reproduced my 16 runs byte for byte.
5. One segmented contract for WAV, microphone and raw samples: fable, opus, sol (conceded). Whole-call raw method: neon, astra (earlier). **open** until astra and neon answer.
6. Microphone: native WASAPI in the library. **agreed** (opus, neon conceded; sol, astra with gates: exact cap or declared packet granularity, no throw outside the worker's catch at Microphone.cs:184, bounded start/stop waits).
7. Continuous transcription. **dropped** from v1 (fable conceded).
8. No arguments = `serve`; `--help` prints usage; `transcribe`, `download`. **agreed** (sol, astra, neon, fable; opus silent).
9. Multipart only. **agreed**.
10. dropped.
11. Guards as sol restated; opus showed neither host checks Host by itself, so it is our code with a test. **agreed**.
12. Kestrel in the Cli project, IIS module excluded (sol's script, astra reran). **agreed** (opus conceded).
13. agreed. 14. **agreed**: page uploads WAV untouched, no browser microphone in v1. 15. **agreed**: page and curl first.
16. No model at `serve`: (a) download at once, progress in /health (opus, fable) vs (b) page button + guarded install POST (sol). **open**.
17. Single-file compression now (opus: 50.3 MB, +37 MB idle memory, +8 ms start) vs separate later step (sol). **open**.

### opus @ 20261123154452537
BUILD — WAV reader v2 answers sol's and neon's gates (opus\proto\lib\WavAudio.cs, ~200 lines; rerun `opus\proto\p2: dotnet build -c Release; dotnet run -c Release --no-build`; output p2-run1.txt).

Changes: polyphase resampler (gcd ratio, per-phase Kaiser taps precomputed), forward-only reads (skip chunks by reading, never Seek on non-seekable), full 16-byte extensible GUID check, block-align/rate/channel validation, fmt size cap, duration limit checked from the declared size before allocating and again while streaming.

neon's cases now: mp3 / empty / no data / rate 0 / bits 0 → InvalidDataException with a user-facing message; JUNK chunk on non-seekable → OK; data size 0xFFFFFFFF streamed → OK (reads to end); over the limit → ArgumentException both declared and streamed. Truncated mid-data → returns what is there (lenient on purpose: crashed recorders write such files; QUESTION sol: you prefer an error?).

10 min 48 kHz stereo PCM16: 0.79–1.05 s (neon measured 11.6 s on v1), NAudio WDL 0.56–0.58 s; 256 MB allocated in total, peak lower.

Transcripts, ours vs NAudio WDL on the same 30 FLEURS clips: 48k stereo float 30/30 identical, 44.1k stereo pcm16 30/30 identical, 8k mono 25/30 identical (upsampling filters differ; aggregate WER was equal in P1).

So point 3's conditions (validation, latency, transcript deltas) are met by a prototype, not only promised.

### sol @ 20261123154502263
CONCEDE — Point 17: Opus’s measured compression run answers my startup reservation. Include compression in the release artifact if the final combined single-EXE/Kestrel build repeats the size advantage and the no-installed-runtime, startup, model load, and signing checks pass. The 15.1→51.9 MB idle private-memory increase matters before model load, so record it with the startup numbers. The release can stay uncompressed if the combined artifact fails the check.

Point 16 remains (b) for me. A double-click should open the demo immediately, with one obvious Download button and visible status. An install job uses network and writes the verified model, independently of whether a browser is open. Astra and neon also posted (b) at 20261123154343361 and 20261123154343024. Their same messages accept uniform segmentation for point 5.

### opus @ 20261123154506988
CONCEDE — points 8, 16, 17.

8: agreed (no args = serve, --help, transcribe, download).
16: (b). Neon's and astra's evidence decides it: a double-click must not start 714 MB of traffic unasked. Page shows source, exact size, folder and a Download button; one process-owned install with progress in GET status; install POST guarded by exact Origin + custom header (sol); `download` command for headless.
17: separate later step (sol). It is orthogonal to the consumer API; my numbers (50.3 MB, +37 MB idle private memory, +8 ms warm start) go in as input to that step, cold start unmeasured.

WORKING — integrated prototype p4, so the README snippets compile and the exe flow runs end to end: public API on my library copy (EnsureInstalledAsync, LoadAsync + options, TranscribeFileAsync/Stream, one segmented contract with ≤320,000-sample parity, ConfigureAwait(false)), fable's Microphone.cs copied in, and a Kestrel `serve` with the agreed guards (exact Host, Origin/Sec-Fetch-Site, 1 running + 1 waiting then 503, 400/413/415/422 codes, model-status + guarded install). fable: I'll take names from your draft when it lands; until then neon's.

### fable @ 20261123154528576
BUILD — on everyone's positions: the full draft answer is in `fable\draft-design.md` (lettered sections A–E, so an objection can say "A4" or "C2"). It writes down the agreed points with exact signatures, and fills gaps nobody disputed. Choices in it that are mine and need your yes or a REBUT:

- A2: `SpeechEngineOptions(bool UseGpu = true, string? GpuNameContains = null)`, `engine.Backend`, `engine.IsUsingGpu`. `LoadAsync` keeps the model parameter required.
- A4: names `TranscribeWavAsync(string path)` and `TranscribeWavAsync(Stream wav)`. I drop my `TranscribeFileAsync`: the method takes only WAV, and the name should say so (sol, astra had it right).
- A5: `Microphone.Devices()`, `Microphone.StartRecording(device = null, longest = null)`, `MicrophoneRecording { Duration, Level, IsRecording, Stop(), Dispose() }`. Not on the engine (neon's `engine.StartMicrophone()`): recording needs no engine, and an app can record while the model loads.
- A7: no progress callback for long files in v1 (additive later).
- C: commands `serve` (also no arguments), `transcribe`, `record`, `download [--from file]`. `record` is new: 10 lines, and it is the tool for the manual real-speech check none of us could run.
- C1: one status route, `GET /health`, 200 when ready and 503 otherwise, with `status`, `progress`, `backend`. `response_format` json or text only.
- C2: byte cap 256 MiB by default.
- E: 11 steps, each with its check; the page mockup goes to the owner before UI code.

Still open: 5 (astra, neon), 16, 17. QUESTION astra, neon: your position on 5 after sol's concession, and on 16?

### sol @ 20261123154533753
REBUT — Opus’s v2 resampling and validation evidence settles the dependency choice. I’d reject a truncated finite WAV by default: WavAudio.cs:73 floors a misaligned declaration, and :82–94 returns a transcript after EOF even when bytes remain. The client sees a successful transcript with missing audio whether the cause was a crashed recorder or a broken transfer. Accept a documented unknown-length sentinel only if bounded and ending on a whole frame; decide this separately from finite truncation. Include declared RIFF bounds, checked wide addition for padding at :39, and a nonseekable fixture.

One allocation gate remains for admitted but uncommon rates: :64 accepts 383,999 Hz; :125–131 sees GCD 1, builds 16,000 phases × about 1,636 taps (roughly 100 MiB of float coefficients) before processing even a small clip. Bound filter-table size with a bounded phase grid or explicitly constrain the supported rate set. Test it independently of the encoded-byte and duration limits. Reject nonfinite float samples at the WAV boundary before they can poison quiet-cut and inference. This is a design review of the prototype; the new normal-input timings are valuable.

### fable @ 20261123154552074
CONCEDE — point 16: (b). Four of you hold it and the argument stands: a double-click must not start 714 MB of traffic unasked, and it matches the library, where the developer calls `EnsureInstalledAsync` explicitly. `fable\draft-design.md` C3 now says: page opens at once with status `no-model` (source, exact size, folder) and a Download button; `POST /model/download` guarded by own Origin + `X-Parakeet-Request: 1`; one process-owned install, progress in `/health`; headless uses `parakeet-runtime download`; `transcribe` and `record` without a model stop with that hint. No `serve --download`: one way to do it.

POINTS — after exchange 3.

1–4, 6, 8, 9, 11–15: agreed. 7, 10: dropped.
5. One segmented contract for samples, WAV and microphone; exactly 320,000 samples or fewer is one unchanged call; the 399 s native limit stays internal. **agreed** (all five; astra reran the 20–23 s clips).
16. **agreed**: (b), as above.
17. **agreed**: compression is a separate release step, kept only if the combined build passes size, cold/warm start, idle memory, model load and signing checks.
3. Gates now met in a prototype: opus's reader v2 passes neon's cases, 10-minute 48 kHz file in 0.8–1.0 s, transcripts identical to NAudio WDL on 30/30 clips at 48 and 44.1 kHz, 25/30 at 8 kHz.

Open, small:
18. A truncated WAV: return what is there (opus) or error (sol to answer).
19. The draft's own choices (A2, A4, A5 names; `record` command; `/health` shape; 256 MiB cap; step order E). Each member: "accept draft" or a REBUT naming the section.

When 18 and 19 close and opus's p4 and neon's snippet test report, I write the result and STOP.

### fable @ 20261123154604294
CLAIM — point 18: I side with sol. A WAV whose data chunk declares more bytes than the stream holds is an `InvalidDataException` that says how much is missing ("the file ends after 41 of 60 seconds of audio"). Reasons: a transcript that silently stops early looks like the engine dropping speech, which is the exact failure this design removes elsewhere; and "check outside data once, where it enters" is the owner's rule (writing-code.md, Design). The streaming sentinel (data size 0xFFFFFFFF, as ffmpeg writes to a pipe) reads to the end of the stream, bounded by the byte cap, ending on a whole frame. Sol's other three gates go into step 4 of the draft: bounded resampler table for odd rates (or a rate range of 8,000–192,000 Hz), non-finite float samples rejected, RIFF bounds checked with wide arithmetic.

QUESTION opus: do you hold "return what is there", and if so, for which user?

### sol @ 20261123154604682
BUILD — I accept A2/A4/A5/A7, the command list, and E. Three draft edits before sign-off:

• A3/A4 “any length/rate/channel count” overpromises. State the v1 supported WAV format/rate/channel envelope and bounded decoded duration; make the encoded HTTP cap independent. 256 MiB fits a ten-minute stereo 48 kHz float WAV, but only with one live request, no redundant full-size copy, and checked decode allocations. Keep the model’s native cap internal.
• C1/C2: add health state missing and error 422 for decoded duration. Keep Host exactly the printed numeric authority. Admitting localhost while only trusting the numeric origin makes a second URL appear supported until POST fails. For POST accept Sec-Fetch-Site only when absent (CLI) or same-origin (browser); none is unnecessary. “All tested” overstates the prototypes: mark guards and graceful cancellation as implementation gates.
• C3: Opus, astra, neon and I now prefer a Download button guarded by exact Origin + custom header. Add install POST with idempotent in-process scheduling, GET health polling, visible failed/retry state and an explicit CLI download command. No automatic download on double-click.

I’m fine putting compression in E11 as a separate optional slice, given Opus’s cold-start caveat.

### astra @ 20261123154620309
BUILD — P2 independently replayed: transcripts vs WDL exactly 30/30 (48 kHz), 30/30 (44.1 kHz), 25/30 (8 kHz), matching Opus. Ten-minute fixture produced 9,610,240 samples = 600.64 s (generator writes whole blocks), 746–881 ms vs WDL 520–551 ms, 256 MiB total allocation for ours. Nonseekable JUNK and unknown-length cases worked. Declared-length premature EOF returned a partial 52,309 samples; recommend strict rejection for finite declared data, only allowing intentional stream-to-EOF for unknown length with complete frames. A successful transcription of a truncated transfer is indistinguishable from a complete one to a caller.

New microphone rerun matched 47,967 samples at 3 s, 24,927 samples with 1.5 s cap (=927 extra samples), and actionable stale-device error. Accept proposed exact-cap fix and full native buffer release.

Point 17: agree with Sol’s conditional compression; final combined self-contained, signed artifact and startup gates determine enabling it. Re points 5/16, Astra and Neon both already supported uniform segmentation + explicit Download button in our preceding messages.

### neon @ 20261123154624072
BUILD — Re-ran `neon\tester` against `opus\proto\lib\WavAudio.cs` v2 (plus Point 17):

1. **`WavAudio.cs` v2 verified:**
   - All 5 earlier failures fixed: MP3, empty stream, no `data`, `rate=0`, `bits=0` all throw clear `InvalidDataException`; odd-length `JUNK` and `0xFFFFFFFF` streaming `data` on non-seekable streams pass (`1600` samples); 11-min declared WAV throws `ArgumentException` before allocating.
   - **10-min 48 kHz stereo PCM16:** `815 ms` (down from `11,559 ms` — 14× faster). Prime rate `44101 Hz` (1 s) took `99 ms`.
   - **2 one-line production fixes still needed:**
     - `WavAudio.cs:39`: `Skip(wav, size + (size & 1))` overflows `uint` when unknown chunk `size == 0xFFFFFFFF` (skipped 0 bytes and returned 2 samples instead of failing). Cast `(long)size + (size & 1)`.
     - `WavAudio.cs:76`: `new List<float>((int)declaredFrames)` trusts the header before reading—a 100-byte body declaring 10 min @ 48 kHz allocated **109 MB**. Cap initial capacity (e.g. `Math.Min(declaredFrames, format.SampleRate * 10L)`).
   - On truncated mid-data (`opus`'s question): lenient return on files is pragmatic, while HTTP uploads should fail if the client disconnected (`RequestAborted`).

2. **Point 17 (single-file compression):** Agree with enabling `EnableCompressionInSingleFile=true` as the final/separate step of the CLI slice, gated on cold+warm startup and `Test-ReleaseArtifacts.ps1` (satisfies both opus's 50 MB size win and sol's measurement gate).

### fable @ 20261123154643569
BUILD — on sol's three edits (20261123154604682): all taken into `fable\draft-design.md`.

- A4 now states the envelope instead of "any": uncompressed PCM 8/16/24/32-bit or float 32/64-bit (also EXTENSIBLE), 1–8 channels, 8,000–192,000 Hz, up to 2 hours of decoded audio. The 2 hours is this version's memory bound (460 MB of samples), named as such, not the model's limit; decoding in blocks straight into the cutter would lift it and is a listed follow-up. Truncated finite WAV = error saying how much is missing; 0xFFFFFFFF = read to the end; non-finite samples rejected.
- C1/C2: 422 for audio over 2 hours; Host must be exactly `127.0.0.1:<port>` (no `localhost` alias); POST accepts `Sec-Fetch-Site` only when absent or `same-origin`; the rules are labelled implementation gates, each with a test; a body is read only after admission, so one upload is in memory at a time; the 256 MiB cap is independent of the audio bound.
- C3 was already (b); status name is `no-model`, with `failed` + retry on the page.

QUESTION all: the one number in there that is my judgment, not a measurement, is "2 hours". Object with a better bound if you have one. Otherwise I need "accept draft" from opus, astra and neon (sol's is conditional on these edits; sol, confirm).

### astra @ 20261123154645525
CLAIM — Accept draft sections A–E including public names, record CLI, /health, separate 256 MiB transport cap, staged implementation and Sol’s clarifications (finite supported formats, resource limits, 422, strict numeric authority, controls still to verify). Point 18: reject truncated finite data at every entry point. C4 minor consistency: C1 returns only text so the page cannot display decoded audio length without metadata. Omit length for v1, or include an explicitly tested response duration; elapsed request time can be measured by the page. In Windows PowerShell 5.1 use curl.exe in README so the example avoids the Invoke-WebRequest alias. I accept the integrated design subject to the documented implementation gates.

### opus @ 20261123154706597
CONCEDE — point 18: strict. Astra's and sol's argument decides it: a truncated transfer and a complete file give the caller the same successful result, and the cut-at-pauses work exists to stop silently missing speech. Finite data that ends early → InvalidDataException naming how much is missing; 0xFFFFFFFF reads to the end on whole frames. Neon's two fixes (wide `(long)size + (size & 1)` skip, capped initial capacity) go into step E4.

Point 19: accept draft (A–E), with sol's and astra's edits.

BUILD — p4 first half: the README snippets compile and run against the public API only (opus\proto\hello, assembly without InternalsVisibleTo; lib copy has A1, A2 options, A3 cutter at 320,000 samples with fable's second-half quietest-300 ms policy, A4 stream/path, A6 ConfigureAwait(false), fable's Microphone.cs).
- `dotnet bin\Release\net10.0\hello.dll wav long120-00.wav`: the 4-line snippet, 127 s of English, 3.0 s wall including model load (GPU), clean joined text (output in my chat log; the method names there are TranscribeFileAsync, to rename per A4).
- `hello.dll mic 3`: StartRecording → Stop → TranscribeAsync: "[3,00 s]" with empty text (silent room). Real speech still untested by anyone.

WORKING — p4 second half: Kestrel `serve` with C1–C3 (health states incl. no-model, guarded POST /model/download, exact Host 127.0.0.1:port, Origin/Sec-Fetch-Site, 1 running + 1 waiting → 503, 400/413/415/422), with a test script that hits every gate.

### sol @ 20261123154710801
REBUT — I accept the edited draft except the 2-hour library cap if it is paired with the current whole-buffer decoder. The 460.8 MB figure counts only 16 kHz output. At the allowed 192 kHz, two hours represent 1,382,400,000 input frames and a single intermediate float array would be 5.53 GB before copying or resampling (WavAudio.cs:76,94,132). The HTTP cap cannot protect a file path or caller stream. Specify a conservative 10-minute v1 bound as a design choice, matching the tested class and microphone default, and require a peak-memory gate at the highest supported rate. If the council wants two hours in v1, block-stream the WAV normalization/cutter in that slice and prove its peak memory. The rest of the draft is accepted.

### neon @ 20261123154723621
CONCEDE — **Accept draft (`fable\draft-design.md`, sections A–E)**, including Point 18 (reject truncated finite WAVs; allow `0xFFFFFFFF` streaming WAVs to EOF on whole frames).

Hello-world check ("Good looks like"): WAV is 5 lines, microphone is 4 lines (`draft-design.md:75-88`). Clean, obvious, zero new dependencies.

Three quick tightenings for the final result (non-blocking):
1. **2-hour memory math (`A4`, line 65):** 2 hours is 460 MB *after* resampling to 16 kHz, but materializing a full 192 kHz mono array before resampling (`WavAudio.cs:76`) takes **5.5 GB** (`7200 × 192000 × 4 B`), and 48 kHz takes **1.38 GB**. Either note that step 4 resamples in bounded blocks (keeping peak memory ~460 MB at any rate) or cap v1 at 1 hour / 256 MiB encoded. Also include the two one-line fixes from my v2 test: `(long)size + (size & 1)` at `WavAudio.cs:39` and capped initial `List` capacity at `:76`.
2. **PowerShell `curl.exe` (astra's catch):** In `D.2` and `C4`, use `curl.exe -F file=@hello.wav ...` so Windows PowerShell 5.1 doesn't hit the `Invoke-WebRequest` alias.
3. **`C4` audio length (astra's catch):** `C1` returns `{"text":"…"}` only, so either add `"duration"` to the JSON reply or drop "audio length" from `C4` (page already times elapsed ms).

Ready for `STOP`.

### astra @ 20261123154825648
BUILD — Verified the p4 WAV example executes (exit 0, public consumer). Acceptance caveat: on long120-00 its 1,649-character text differs by one character from fable’s measured quietest-20 reference. The two cutters implement different metrics: fable/proto/LongAudio.cs:54–80 scores sums of ten 30-ms RMS values on a frame grid; opus/proto/lib/LongAudio.cs:21–37 scores the squared samples of the full window, on an exact 10.000-s offset. Both fit the prose. Preserve fable’s actual metric and tie behavior if citing its measurements, or rerun the final shipped cutter on that corpus before attaching those numbers to it. This is an implementation acceptance check, not an objection to the design.

### fable @ 20261123154852557
STOP — SESSION reached 50.

agreed: the design in sections A–E, accepted by all five. Library: `EnsureInstalledAsync`, `SpeechEngineOptions` + `Backend`, `TranscribeWavAsync(path | stream)`, `TranscribeAsync(samples)` of any length (above 320,000 samples it cuts at the quietest points, ≤ 20 s each: 32–43% → 6% word errors on long English), `Microphone.StartRecording()` → `Stop()` over native WASAPI, no new dependency, `ConfigureAwait(false)` under the gate. Exe: no arguments = `serve` on `127.0.0.1` with the page and multipart `POST /v1/audio/transcriptions` on Kestrel, exact Host + Origin + Sec-Fetch-Site checks, one running + one waiting, model downloaded only from the page's button or `download`; plus `transcribe`, `record`. README: page and curl first, then NuGet WAV and microphone. Eleven build steps, each with its check. Point 18 closed in the last exchange: a truncated WAV is an error (opus conceded).

open: the bound on decoded audio. I kept 2 hours and took the way out sol and neon both named (convert in bounded blocks, peak-memory check in step 4; a whole-file buffer at 192 kHz would be 5.5 GB); they had no turn left to confirm, and their lower bound is the fallback. Not built: the guarded server (opus's p4, second half) and the page.

dropped: continuous microphone transcription in v1; "PrivateType works unchanged".

result: C:\Users\Krzysiek\.councils\parakeet-consumer-api\result-session-1.md

### opus @ 20261123155927699
BUILD — p4 done: draft C runs end to end as ONE self-contained exe (103.7 MB, no aspnetcorev2_inprocess.dll; my 4-line target in opus\proto\p4\p4.csproj). Rerun: `opus\proto\test-p4.sh`; output test-p4-run1.txt.

C3, no model (`--model-folder <empty>`): /health 503 "no-model" with source, bytes, folder; transcription 503 not-ready; /model/download without header 403, with header + foreign Origin 403; console prints the `download` hint.
C1/C2, with model: HTTP up 627 ms, ready 2.2 s. Host localhost:5092 → 421, evil → 421; POST Origin evil → 403; Sec-Fetch-Site cross-site → 403; same-origin → 200. 48 kHz WAV → 200 {"text"}; response_format=text → plain; mp3 → 415; truncated → 400 "ends after 0,3 of 17,5 seconds"; no file → 400; two files → 400; raw audio/wav body → 415; 260 MiB → 413. Three 10-minute WAVs at once: #3 503 in 1 ms, #1 200 in 2.2 s, #2 200 in 3.9 s (waited its turn). 0 files in ASPNETCORE_TEMP. `transcribe x.wav` prints text; unknown command exit 2.
Lib: public API only (A1–A6), TranscribeWavAsync names, strict truncation, 1–8 ch, 8–192 kHz, non-finite rejected (opus\proto\lib).

Two findings for the draft:
1. A4 says InvalidDataException for everything outside the envelope, but C1 needs 415 vs 400. I used NotSupportedException for "not a WAV / unsupported encoding / rate / channels" and InvalidDataException for damaged files. Propose writing that split into A4.
2. Memory: private memory was 2.4 GB after two 10-minute 48 kHz stereo uploads (model ~1.1 GB). Upload bytes + 48 kHz mono floats + list growth stack up. Fine for v1 with one body at a time (my admission reads the body only after the engine's turn), but it makes fable's "decode in blocks" follow-up real; the 256 MiB cap is the bound.

Microphone with real speech: still untested.
