# Release notes

The release workflow refuses a tag `vX.Y.Z` until `vX.Y.Z.md` exists here, and uses it as the
draft release's text. Write it for users and include:

- what changed, with measured speed changes (machine, model, audio length);
- the downloads: `ParakeetRuntime-X.Y.Z-win-x64.zip` (the `parakeet-runtime.exe` tool) and the
  `ParakeetRuntime` NuGet package, each with its `.sha256` file;
- that `parakeet-runtime.exe`, `ParakeetRuntime.dll` and the package are signed by
  **Open Source Developer Krzysztof Kołodziejczak** (Certum), and that SmartScreen may still warn
  while the certificate builds reputation.
