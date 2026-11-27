# Release acceptance

Before a version is tagged, the owner runs the manual checks that tests can't cover: the microphone, the exe's
commands, the HTTP server, the page in a browser, the device choice and the package in a new app.
`Start-Acceptance.ps1` walks through them:

```powershell
.\scripts\acceptance\Start-Acceptance.ps1 -Version 0.6.0
```

It builds the release files for that version, makes its test files, runs every command, and asks for a
verdict after each step: Enter for pass, `f` for fail (then type or dictate what you saw), `s` to skip, `r` to
run the step again. The results go to `artifacts\acceptance\<version>\results-<time>.md`. `-From 3.1` starts at
a step, `-SkipBuild` reuses the release files already built, and `-PrepareOnly` builds and stops.

The wizard moves the models folder aside for the steps that need no model, and puts it back after each of
them, also on Ctrl+C. It asks before deleting a test download of the model.

## Audio

`audio\` holds eight clips from the FLEURS dev set: six English (`en_us`) and two Polish (`pl_pl`), 16 kHz mono
16-bit. `audio\clips.tsv` gives each clip's original FLEURS file name and its reference text, which the wizard
prints next to what the engine writes. The 6-minute file and the bad files are made at run time in
`artifacts\acceptance\<version>\`.

FLEURS is © Google, released under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
([dataset](https://huggingface.co/datasets/google/fleurs); Conneau et al., "FLEURS: Few-shot Learning Evaluation of
Universal Representations of Speech", 2022). The clips are unchanged copies.
