# ParakeetRuntime.Net

<p align="center">
  Offline speech-to-text for .NET on Windows, running NVIDIA's Parakeet TDT models in GGUF format.
</p>

<p align="center">
  <a href="https://github.com/kkolodziejczak/ParakeetRuntime.Net/actions/workflows/ci.yml"><img src="https://github.com/kkolodziejczak/ParakeetRuntime.Net/actions/workflows/ci.yml/badge.svg" alt="Build and test status" /></a>
  <a href="https://github.com/kkolodziejczak/ParakeetRuntime.Net/releases/latest"><img src="https://img.shields.io/github/v/release/kkolodziejczak/ParakeetRuntime.Net" alt="Latest release" /></a>
  <a href="https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/LICENSE"><img src="https://img.shields.io/github/license/kkolodziejczak/ParakeetRuntime.Net" alt="MIT license" /></a>
</p>

ParakeetRuntime is a speech recognition engine written in C#. It loads a Parakeet TDT model from a
GGUF file and transcribes audio on the processor, with int8 kernels, or on the graphics card,
through Direct3D 12. Audio never leaves your computer, and no Python or separate native runtime
is needed.

> **Status: before the first release.** The engine moved here from
> [LocalBuddy](https://github.com/kkolodziejczak/LocalBuddy), where it was built and tuned. Its
> API (`SpeechEngine` and `SpeechModel`) may still change before version 1.0.

## Two ways to use it

| Download | Choose it when |
| --- | --- |
| `ParakeetRuntime-<version>-win-x64.zip` from [Releases](https://github.com/kkolodziejczak/ParakeetRuntime.Net/releases/latest) | You want `parakeet-runtime.exe`, a self-contained command-line tool. No .NET installation needed. |
| The `ParakeetRuntime` NuGet package | You want speech-to-text inside your own .NET 10 application. |

```powershell
dotnet add package ParakeetRuntime
```

## The speech model

The model is not included; download it once:
[`parakeet-tdt-0.6b-v3.q8_0.gguf`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.q8_0.gguf).
Its SHA-256 must be `e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e`. The model
is © NVIDIA, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and
recognizes 25 European languages.

Load only model files you trust: a GGUF file is input to the engine like any other file.

## Verify a download

Each release file has a `.sha256` file next to it. The hash must match exactly:

```powershell
Get-FileHash .\ParakeetRuntime-0.1.0-win-x64.zip -Algorithm SHA256
```

`parakeet-runtime.exe`, `ParakeetRuntime.dll` and the NuGet package are signed by
**Open Source Developer Krzysztof Kołodziejczak** (certificate issued by Certum). For the exe or
the DLL, right-click the file, choose **Properties → Digital Signatures**, and confirm that name.
For the package, run `dotnet nuget verify ParakeetRuntime.<version>.nupkg`. Windows SmartScreen
may still warn while the certificate builds reputation. Download only from this repository.

## Requirements

- Windows 10 or 11, 64-bit, on an x86-64 processor.
- For the GPU path, a graphics card or integrated graphics with Direct3D 12 support. Without
  one, the engine runs on the processor.
- For the NuGet package, .NET 10.

## Help and feedback

- [Get help or report a problem](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/SUPPORT.md)
- [Report a security vulnerability](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/SECURITY.md)
- [Contribute](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/CONTRIBUTING.md)
  and the [Code of Conduct](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/CODE_OF_CONDUCT.md)

## License

ParakeetRuntime is [MIT licensed](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/LICENSE).
Its engine is a C# port of parts of NVIDIA's NeMo-Speech.cpp (Apache-2.0); see
[THIRD-PARTY-NOTICES.md](https://github.com/kkolodziejczak/ParakeetRuntime.Net/blob/master/THIRD-PARTY-NOTICES.md).

ParakeetRuntime is an independent open-source project. It is not affiliated with, endorsed by, or
supported by NVIDIA.
