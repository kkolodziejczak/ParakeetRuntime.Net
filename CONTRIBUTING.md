# Contributing to ParakeetRuntime.Net

Thank you for helping improve ParakeetRuntime. Bug reports, measurements, documentation, tests and
code are welcome.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). By submitting a
contribution, you agree that it may be distributed under the repository's [MIT License](LICENSE).

## Before you start

- Search [existing issues](https://github.com/kkolodziejczak/ParakeetRuntime.Net/issues) first.
- For a security vulnerability, follow [SECURITY.md](SECURITY.md) instead of opening an issue.
- For a substantial change, open an issue first so its scope can be agreed before you build it.

## Development setup

Required: Windows 10 or 11 (x64), the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0),
Git and PowerShell. A Direct3D 12 graphics device is needed only to run the GPU tests.

```powershell
git clone https://github.com/kkolodziejczak/ParakeetRuntime.Net.git
Set-Location ParakeetRuntime.Net
dotnet test ParakeetRuntime.Net.slnx
```

Check that a release still builds and packs:

```powershell
.\scripts\Build-Release.ps1 -Version 0.0.0
.\scripts\Test-ReleaseArtifacts.ps1 -Version 0.0.0
```

## Making a change

1. Create a focused branch from `master`.
2. Reproduce the bug, or measure the behavior you intend to change.
3. Add or update tests that prove the behavior.
4. For engine changes, compare transcripts with the reference output and report the speed before
   and after on the same machine, model and audio, with the command you ran.
5. Update documentation when behavior, requirements or commands change.
6. Open a pull request and complete the template's checklist.

### Guidelines

- Keep nullable reference types enabled and resolve warnings instead of suppressing them.
- Pin package and GitHub Action versions; do not introduce floating versions.
- Never commit models, recordings, or transcripts of real speech. Use synthetic audio or a public
  set whose license allows it.
- The engine is a port of NeMo-Speech.cpp (Apache-2.0). If you port more of it, say so in
  [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) in the same pull request.
