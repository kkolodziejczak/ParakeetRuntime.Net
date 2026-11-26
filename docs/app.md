# Using the app

`parakeet-runtime.exe` comes in `ParakeetRuntime.Net-<version>-win-x64.zip` on
[Releases](https://github.com/kkolodziejczak/ParakeetRuntime.Net/releases/latest). It needs no .NET
installation.

## Try it in a minute

1. Unzip the release and double-click `parakeet-runtime.exe`. It opens a page in your browser at
   `http://127.0.0.1:5092/`.
2. The first time, the page offers to download the speech model (714 MB). Nothing downloads until you
   click **Download model**.
3. Drop a WAV file on the page. Its text appears with the time it took, a **Copy text** button and the
   `curl` command that does the same.

## Commands

From a terminal in the unzipped folder:

```powershell
.\parakeet-runtime.exe download                 # installs the model once
.\parakeet-runtime.exe transcribe hello.wav     # prints the text
.\parakeet-runtime.exe record                   # records the default microphone until Enter (at most 10 minutes), prints the text
.\parakeet-runtime.exe serve --no-browser       # the page and the HTTP API, without opening the browser
curl.exe -F file=@hello.wav http://127.0.0.1:5092/v1/audio/transcriptions
```

Write `curl.exe`, not `curl`: in Windows PowerShell 5.1, `curl` is another command. `transcribe`, `record`
and `serve` take `--device cpu`, `gpu` or `gpu:<part of the adapter's name>`; without it, a usable GPU is
chosen, else the processor. `.\parakeet-runtime.exe --help` lists everything.

The audio formats, long audio and languages are the library's; see
[what to expect](library.md#what-to-expect).

## HTTP API

`parakeet-runtime serve` (or the exe with no arguments) answers on `http://127.0.0.1:5092` only, never on
the network. `--port` changes the port.

| Route | What it does |
| --- | --- |
| `GET /` | The page. |
| `GET /health` | `{"status","progress","version","backend","model","error"}`. `status` is `no-model`, `downloading`, `loading`, `ready` or `failed`. HTTP 200 when ready, 503 otherwise. |
| `POST /v1/audio/transcriptions` | `multipart/form-data` with one `file` part holding a WAV, and optionally `response_format=text`. Answers `{"text":"…","duration":17.54}` (seconds of audio), or the plain text. Other fields are ignored, so OpenAI-style clients work. |

A transcription request that fails answers `{"error":{"code","message"}}`:

| Status | When |
| --- | --- |
| 400 | The request or the WAV is malformed. |
| 403 | The request came from another web page. |
| 408 | The request body did not arrive within one minute of its turn starting. |
| 413 | The request body is over 256 MiB. |
| 415 | The request is not `multipart/form-data`, the file is not a WAV, or its encoding is not supported. |
| 421 | The request used another address than `127.0.0.1:<port>`, for example `localhost`. |
| 422 | The audio is longer than 10 minutes. |
| 503 | The model is not ready, or one transcription runs and another already waits. |

Requests must use the address the exe prints (`127.0.0.1:<port>`) and must not come from another web
page, so websites you visit can't use the server. Nothing about the audio or its text is logged or written
to disk.
