# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| Latest release | Yes |
| `master` | Yes |
| Older releases | No |

## Report a vulnerability

Please do not open a public issue for a suspected vulnerability.

1. Use GitHub's **Report a vulnerability** option on the repository's Security tab when it is
   available.
2. Otherwise, contact the maintainer privately through the options on
   [their GitHub profile](https://github.com/kkolodziejczak).
3. Include the affected version or commit, the impact, reproduction steps or a proof of concept,
   and any suggested fix. A crafted model or audio file that triggers the problem is welcome.
   Do not send private recordings.

The maintainer will assess the report, ask for details if needed, and coordinate a fix and its
disclosure according to severity. Please allow a reasonable time to investigate before publishing
details.

## Security considerations for users

- A GGUF model file is input that the engine parses and reads weights from. Load only model files
  from sources you trust, and check their SHA-256 against the published value.
- Download `parakeet-runtime.exe` and the package only from this repository's
  [Releases](https://github.com/kkolodziejczak/ParakeetRuntime.Net/releases) or the `ParakeetRuntime`
  package on nuget.org. Releases are signed by **Open Source Developer Krzysztof Kołodziejczak**
  (Certum); do not run a file whose signature is missing or names someone else.
- Transcription runs locally and opens no network connections. The only network call is
  `SpeechModel.DownloadAsync`, which fetches the pinned model from Hugging Face when an application
  calls it, through the `HttpClient` that application passes in.

This policy covers ParakeetRuntime. Report problems in .NET, Windows, graphics drivers, or the
NVIDIA models to their own vendors.
