# Manual acceptance session before the first NuGet release

The owner signs off the whole consumer API in one session at the end (decided 2026-11-24), after
steps 1–10 and before the release tag. Each step adds here what only a person at the machine can check.
Run against the release build (`.\scripts\Build-Release.ps1 -Version 0.0.0`), not a debug build.

## Step 6 · Microphone

1. `parakeet-runtime record`, speak two sentences in English, press Enter: the text matches what was said.
2. The same in Polish.
3. `record`, then unplug the microphone (or turn it off in Settings > System > Sound) while recording,
   then press Enter: the message says "The microphone stopped working: The microphone was unplugged or turned off."
4. Settings > Privacy & security > Microphone > "Let desktop apps access your microphone" off, then `record`:
   the message names that setting. Turn it back on.
5. No microphone at all (all disabled in Sound settings), then `record`: "Windows has no microphone to record from."
6. `record`, wait silently 3 s, press Enter: note what the model writes for silence (council saw "Yeah.").

## Step 7 · Exe commands

1. `parakeet-runtime --help` lists `transcribe`, `record`, `download`.
2. `transcribe` on a WAV of your own speech (phone recording exported to WAV): the text is right.
3. `transcribe` on a 5–10 minute WAV (a podcast excerpt): no sentences missing at the 20 s cuts.
4. `transcribe` on an MP3 renamed to `.wav`: a clear "not a WAV file" message, exit code 1.
5. On a second machine, or after renaming `%LOCALAPPDATA%\ParakeetRuntime\models` away: `transcribe` says to run
   `parakeet-runtime download`; `download` shows progress and installs (714 MB); rename the folder back.
6. `download --from <copy of the .gguf>` on a machine with no network installs from the copy.
7. Ctrl+C during `download`: it stops, and no `.part` file is left behind after the next run.
8. Polish characters print correctly in Windows Terminal and when redirected: `transcribe pl.wav > out.txt`.

## Step 8 · HTTP server

1. `parakeet-runtime serve`, then from another terminal
   `curl.exe -F file=@speech.wav http://127.0.0.1:5092/v1/audio/transcriptions`: the JSON text and duration are right.
2. The same with `-F response_format=text`: only the text.
3. Windows Firewall asks nothing (the server listens on 127.0.0.1 only); from another computer on the network the
   port does not answer.
4. Rename the models folder away, `serve`, then `curl.exe http://127.0.0.1:5092/health`: `no-model` with the URL,
   size and folder; the console names `parakeet-runtime download`. Rename it back.
5. Ctrl+C stops the server with "Stopped.", and the port is free again at once.
6. Any HTTP client you use (an OpenAI SDK pointed at `http://127.0.0.1:5092/v1`, Postman): the transcription route
   answers.
