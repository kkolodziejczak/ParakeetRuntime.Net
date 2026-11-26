# ParakeetRuntime.Net agent guide

Repository-specific guidance. Inspect the current code and scripts before acting on it.

## Project shape

| Path | Purpose |
| --- | --- |
| `ParakeetRuntime.Net.slnx` | The solution. |
| `src/ParakeetRuntime.Net` | The engine library, packed as the `ParakeetRuntime.Net` NuGet package. Plain `net10.0`, marked Windows-only. |
| `src/ParakeetRuntime.Net.Cli` | `parakeet-runtime.exe`, published self-contained as one file: the user commands `transcribe`, `record`, `download` and `serve` (`ConsoleCommands.cs`; the HTTP server on Kestrel, loopback only, in `LocalServer.cs` and `ModelState.cs`; the page it serves at `/` is the embedded `Page.html`, whose script hash `Page.cs` puts in the Content-Security-Policy), and the benchmarks (`--corpus`, `--list-gpus`, `--cpu-bench`, `--gemm-bench`, `--session-memory`). |
| `tests/ParakeetRuntime.Net.Tests` | xUnit tests: kernels, GGUF reading, GPU kernels (skipped without a GPU), and the real model (skipped when it isn't installed). |
| `scripts/Run-Matrix.ps1` | The full speed matrix against NeMo-Speech.cpp, one run at a time, through `parakeet-runtime.exe --corpus`. |
| `scripts/Build-Release.ps1` | Builds the unsigned release files into `artifacts\release\<version>`. |
| `scripts/Test-ReleaseArtifacts.ps1` | Checks those files, and their signatures when given a thumbprint. |
| `Directory.Build.props` | Shared settings: target framework, version, embedded PDBs, package metadata. |

The library has no Windows SDK projection: the engine reaches Windows only through P/Invoke into
`d3d12.dll`, `dxgi.dll`, `d3dcompiler_47.dll`, `kernel32.dll` and `ole32.dll`, which every supported Windows
ships. Keep it that way, so that any `net10.0` application can reference the package.

## Where the engine came from

The engine was built in LocalBuddy and moved here on 2026-11-23 as a fresh copy of
`kkolodziejczak/LocalBuddy` at `19d2905` (branch `perf/engine-matrix`); its history stays there.
Its measurements are in LocalBuddy PR #3 and in that branch's commit messages. LocalBuddy will
switch to the `ParakeetRuntime.Net` package in its own repository. Its live and replay commands stay
there, because they use the app's audio capture.

- The public API is `SpeechEngine`, `SpeechEngineOptions`, `SpeechModel`, `Microphone`,
  `MicrophoneDevice` and `MicrophoneRecording`; everything else in the engine is internal.
  Design a change to it before making more types public.
- Keep the shaders as embedded resources and the attribution to NeMo-Speech.cpp. Port more of
  NeMo-Speech.cpp only with an update to [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
- `SpeechModel.Default()` keeps the model in `%LOCALAPPDATA%\ParakeetRuntime\models`. The tests
  that need the real model look for it there.

### Engine documentation

- [docs/model-and-decoding.md](docs/model-and-decoding.md): GGUF reading, Q8_0 blocks, the mel front end, encoder, TDT decoding, model install.
- [docs/cpu-kernels.md](docs/cpu-kernels.md): the int8 CPU paths, quantization, threads and the thread budget.
- [docs/gpu-path.md](docs/gpu-path.md): the Direct3D 12 host side: adapters, pipelines, buffers, the Q8 GEMM tiles, driver caveats.
- [docs/gpu-shaders.md](docs/gpu-shaders.md): what every shader entry point computes, its layouts and invariants.
- [docs/microphone.md](docs/microphone.md): WASAPI capture into memory, its thread, cap, timeouts and errors.
- [docs/benchmarks-howto.md](docs/benchmarks-howto.md) and [docs/benchmarks.md](docs/benchmarks.md): how to measure, and every measurement so far.

### Owner's rules for the engine code

- No comments in code: a reason goes into a name, and hardware-specific behaviour becomes a named
  method or constant. License and attribution headers stay.
- Facts that hold on every machine and don't change (for example the ggml Q8_0 quantize contract
  or the model format) go into `docs/`.
- Measured values go into `docs/benchmarks.md`, which is append-only and dated: date, machine,
  commit and numbers, never edited afterwards.

### Open follow-ups

- Tiered JIT: with default tiering, `Kernels.Quantize`'s AVX2 passes stay at instrumented Tier0 for
  a whole `--cpu-bench` run, which skews kernel A/B tests. A/B kernels with
  `DOTNET_TieredCompilation=0`. Whether an app's first utterances run Tier0 code is not measured.
- Next CPU step, which needs the owner's OK: the positions-table walk in the attention scores.
- Other platforms (decided 2026-11-24: v1 stays Windows-only): the cheap first step is CPU
  transcription and WAV on Linux x64. `ThreadBudget` calls `GetSystemCpuSetInformation` without an
  `OperatingSystem.IsWindows()` guard; guard it, drop the assembly's Windows-only mark for that path,
  and add a Linux CI job. The GPU (Direct3D 12) and microphone (WASAPI) stay Windows-only.
- Benchmarks: pin 4 P-cores (`--affinity C03` on the owner's Core Ultra 7 270K Plus) and run one at
  a time. The CPU has no AVX-512.

## Verification

```powershell
dotnet test ParakeetRuntime.Net.slnx --configuration Release
.\scripts\Build-Release.ps1 -Version 0.0.0
.\scripts\Test-ReleaseArtifacts.ps1 -Version 0.0.0
```

- CI (`.github/workflows/ci.yml`) runs all three on every pull request and push to `master`.
- Hosted runners have no GPU. GPU tests must skip cleanly there and run on a local machine.
- Every engine change compares transcripts against the reference output and states the speed
  before and after, measured on the same machine, model and audio.

## Data and claims

- Never commit models, recordings, or transcripts of real speech. Test audio is synthetic or from
  a public set (for example FLEURS) whose license allows it.
- Every speed or accuracy figure in a document names its machine, model, audio, and date.
  Re-measure and update [README.md](README.md) when the engine, the model, or the build changes.

## Documentation boundaries

- [README.md](README.md) is for users: what it is, downloads, the model, verification,
  requirements, help, license. It is also the NuGet package's readme, so its links are absolute.
- [CONTRIBUTING.md](CONTRIBUTING.md) is for contributors: setup, commands, pull requests.
- Release mechanics and repository internals stay in this file.

## Releasing (maintainer only)

1. Add `.github/release-notes/vX.Y.Z.md` (see the [README there](.github/release-notes/README.md))
   on `master`.
2. Push the tag `vX.Y.Z`. `release.yml` tests and builds, then waits for signing.
3. Run `gh release-signing approve -Tag vX.Y.Z` and type the SimplySign phone code. The signing
   job (in [gh-release-signing](https://github.com/kkolodziejczak/gh-release-signing)) signs
   `parakeet-runtime.exe`, `ParakeetRuntime.Net.dll` inside the package, and the package, all in one
   approval.
4. Review the draft release, then publish it. `publish-nuget.yml` pushes the signed package to
   nuget.org.

A manual run of `release.yml` from a branch builds unsigned test files and never signs or
releases.

### Starting the approval for the owner

<!-- Shared with every repository that signs through gh-release-signing; keep it identical to the
     copy in that repository's README ("Instructions for agents"). -->
When a run waits for signing, start the approval script for the owner instead of asking them to
run it. Open it in its own window, from this repository's folder, so the owner can answer it:

```powershell
Start-Process powershell.exe -WorkingDirectory (Get-Location) -ArgumentList '-NoExit', '-Command', 'gh release-signing approve -Tag vX.Y.Z'
```

- Pass `-RunId <id>` after a rerun, and `-Workflow <file>` when the calling workflow is not
  `release.yml`.
- If `gh extension list` does not show `release-signing`, run `Approve-ReleaseSigning.ps1` from a
  local clone of gh-release-signing instead (`-NoExit -ExecutionPolicy Bypass -File <path> …`),
  still with this repository's folder as the working directory.
- Never answer its prompts or type the code yourself. Tell the owner the window is open, then wait
  for the run to finish in the background and report its result.

### One-time setup

- The `code-signing` environment (owner as required reviewer; `v*.*.*` tags and `master`) needs
  the secrets `CERTUM_EMAIL` and `MAILBOX_TOKEN`, and this repository must be on that token's
  access list. See the gh-release-signing README.
- `publish-nuget.yml` pushes through nuget.org trusted publishing, so no API key is stored. On
  nuget.org, add a trusted publishing policy (Repository owner `kkolodziejczak`, Repository
  `ParakeetRuntime.Net`, Workflow file `publish-nuget.yml`, Environment empty). The workflow logs
  in as the nuget.org user `kkolodziejczak`, which owns that policy (set up on 2026-11-22).
- nuget.org accepts a signed package only after its certificate is registered on the account:
  export the public certificate (`.cer`) from a signed `parakeet-runtime.exe` and upload it in the
  nuget.org account settings. Repeat after each yearly certificate renewal.

## Conventions

- Pin package and GitHub Action versions (Actions by SHA with a version comment); dependabot
  updates the Actions weekly. The one deliberate exception is `gh-release-signing`'s `sign.yml@main`:
  the owner's yearly certificate renewal reaches every repository through that branch.
- Keep nullable reference types on and resolve warnings instead of suppressing them.
