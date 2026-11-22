# ParakeetRuntime.Net agent guide

Repository-specific guidance. Inspect the current code and scripts before acting on it.

## Project shape

| Path | Purpose |
| --- | --- |
| `ParakeetRuntime.Net.slnx` | The solution. |
| `src/ParakeetRuntime` | The engine library, packed as the `ParakeetRuntime` NuGet package. Plain `net10.0`, marked Windows-only. |
| `src/ParakeetRuntime.Cli` | `parakeet-runtime.exe`, published self-contained as one file. |
| `tests/ParakeetRuntime.Tests` | xUnit test project; empty until the engine and its tests move in. |
| `scripts/Build-Release.ps1` | Builds the unsigned release files into `artifacts\release\<version>`. |
| `scripts/Test-ReleaseArtifacts.ps1` | Checks those files, and their signatures when given a thumbprint. |
| `Directory.Build.props` | Shared settings: target framework, version, embedded PDBs, package metadata. |

The library has no Windows SDK projection: the engine reaches Windows only through P/Invoke into
`d3d12.dll`, `dxgi.dll`, `d3dcompiler_47.dll` and `kernel32.dll`, which every supported Windows
ships. Keep it that way, so that any `net10.0` application can reference the package.

## Moving the engine in

The engine lives in LocalBuddy at `src/LocalBuddy.Transcription/Parakeet/` (C# and HLSL
shaders), with its benchmark in `src/LocalBuddy.Spike/` and its tests in `tests/LocalBuddy.Tests/`
(`GgufTests`, `KernelsTests`, `D3D12KernelsTests`, `SpeechEngineTests`). Several LocalBuddy
branches still change it (`perf/engine-matrix`, `feat/optimize-long-audio-attention`,
`other/perf-long-utterance-gpu-scaling`). Ask the owner which state to move, and whether to keep
its Git history, before copying anything.

- Rename the namespace `LocalBuddy.Transcription.Parakeet` to `ParakeetRuntime`.
- Keep the shaders as embedded resources and the per-file note that the code is a port of
  NeMo-Speech.cpp. Port more of NeMo-Speech.cpp only with an update to
  [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
- The benchmark and live commands become subcommands of `parakeet-runtime.exe`.
- Design the public API before making types public: today everything in the engine is internal.

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
   `parakeet-runtime.exe`, `ParakeetRuntime.dll` inside the package, and the package, all in one
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

- The GitHub CLI (`gh`) is installed and signed in. Use it for everything on GitHub (runs,
  approvals, releases, secrets, `gh api`) instead of HTTP requests or fetching web pages.
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
- `publish-nuget.yml` needs the repository secret `NUGET_API_KEY`, an API key scoped to push
  `ParakeetRuntime`.
- nuget.org accepts a signed package only after its certificate is registered on the account:
  export the public certificate (`.cer`) from a signed `parakeet-runtime.exe` and upload it in the
  nuget.org account settings. Repeat after each yearly certificate renewal.

## Conventions

- Pin package and GitHub Action versions (Actions by SHA with a version comment); dependabot
  updates the Actions weekly. The one deliberate exception is `gh-release-signing`'s `sign.yml@main`:
  the owner's yearly certificate renewal reaches every repository through that branch.
- Keep nullable reference types on and resolve warnings instead of suppressing them.
