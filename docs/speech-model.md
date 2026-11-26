# The speech model and release files

## The speech model

The speech model is 714 MB, so we don't bundle it with the library or the app. Instead, the browser UI, `parakeet-runtime.exe download`, and `SpeechModel.EnsureInstalledAsync` fetch it once on demand and check its size and SHA-256 before keeping it.

The pinned file is [`parakeet-tdt-0.6b-v3.q8_0.gguf`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.q8_0.gguf) (714 MB), with SHA-256 `e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e`. The model is © NVIDIA under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). By default, `SpeechModel.Default()` stores it in `%LOCALAPPDATA%\ParakeetRuntime\models`.

On a machine without internet access, copy the GGUF file over and install it with `.\parakeet-runtime.exe download --from <file>` or `SpeechModel.InstallFromFileAsync`. Both verify the file's size and SHA-256 the same way before marking it installed.

Load only model files you trust: a GGUF file is input to the engine like any other binary file.

## Verify a download

Every release file has a matching `.sha256` file next to it. Hash the download in PowerShell and make sure it matches:

```powershell
Get-FileHash .\ParakeetRuntime.Net-<version>-win-x64.zip -Algorithm SHA256
```

`parakeet-runtime.exe`, `ParakeetRuntime.Net.dll`, and the NuGet package are signed by **Open Source Developer Krzysztof Kołodziejczak** (certificate issued by Certum). For the exe or DLL, right-click the file, open **Properties → Digital Signatures**, and confirm that signer name. For the NuGet package, run `dotnet nuget verify ParakeetRuntime.Net.<version>.nupkg`.

Windows SmartScreen may still warn while the certificate builds reputation. Download release files only from this repository.
