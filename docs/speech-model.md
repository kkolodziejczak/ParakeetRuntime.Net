# The speech model and release files

## The speech model

The model is not included. The app's page, `parakeet-runtime.exe download` and
`SpeechModel.EnsureInstalledAsync` download it once and check it:
[`parakeet-tdt-0.6b-v3.q8_0.gguf`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.q8_0.gguf)
(714 MB). Its SHA-256 must be `e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e`. The model
is © NVIDIA, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

`SpeechModel.Default()` keeps it in `%LOCALAPPDATA%\ParakeetRuntime\models`.

On a computer without internet access, copy the file over and install it with
`.\parakeet-runtime.exe download --from <file>`, or `SpeechModel.InstallFromFileAsync`. Both check it the
same way.

Load only model files you trust: a GGUF file is input to the engine like any other file.

## Verify a download

Each release file has a `.sha256` file next to it. The hash must match exactly:

```powershell
Get-FileHash .\ParakeetRuntime.Net-0.1.0-win-x64.zip -Algorithm SHA256
```

`parakeet-runtime.exe`, `ParakeetRuntime.Net.dll` and the NuGet package are signed by
**Open Source Developer Krzysztof Kołodziejczak** (certificate issued by Certum). For the exe or
the DLL, right-click the file, choose **Properties → Digital Signatures**, and confirm that name.
For the package, run `dotnet nuget verify ParakeetRuntime.Net.<version>.nupkg`. Windows SmartScreen
may still warn while the certificate builds reputation. Download only from this repository.
