# Using the app

Download `ParakeetRuntime.Net-<version>-win-x64.zip` from [Releases](https://github.com/kkolodziejczak/ParakeetRuntime.Net/releases/latest) and unzip it. `parakeet-runtime.exe` is self-contained and needs no .NET installation.

## Try it in a minute

1. Double-click `parakeet-runtime.exe`. It starts a local server and opens `http://127.0.0.1:5092/` in your browser.
2. On the first run, the page offers to download the speech model (714 MB). Nothing downloads until you click **Download model**.
3. Drop a WAV file onto the page, or click **Choose file**. You get the transcript, the audio duration and transcription time, a **Copy text** button, and the matching `curl.exe` command.

## Command line

From a terminal in the unzipped folder:

```powershell
.\parakeet-runtime.exe download                 # downloads and checks the model once
.\parakeet-runtime.exe transcribe hello.wav     # prints the text
.\parakeet-runtime.exe record                   # records the default microphone until Enter (at most 10 minutes), prints the text
.\parakeet-runtime.exe serve --no-browser       # starts the browser UI and HTTP API without opening a browser
```

While `serve` (or the double-clicked app) is running, you can send WAV files to the local HTTP API:

```powershell
curl.exe -F file=@hello.wav http://127.0.0.1:5092/v1/audio/transcriptions
```

In Windows PowerShell 5.1, `curl` is an alias for `Invoke-WebRequest`, so type `curl.exe`.

`transcribe`, `record`, and `serve` accept `--device cpu`, `--device gpu`, or `--device gpu:<part of the adapter's name>`. Leave `--device` off and the app picks a usable GPU, falling back to the CPU. Transcript text goes to standard output; status and error messages go to standard error. Run `.\parakeet-runtime.exe --help` to list every command and flag.

WAV format limits, long-audio splitting, and language detection are the same as in the library; see [What to expect](library.md#what-to-expect).

## HTTP API

`parakeet-runtime.exe serve` (or running the exe with no arguments) listens on `http://127.0.0.1:5092` only and never binds to the network. Pass `--port` to change the port.

| Route | What it does |
| --- | --- |
| `GET /` | Serves the browser UI. |
| `GET /health` | Returns `{"status","progress","version","backend","model","error"}`. `status` is `no-model`, `downloading`, `loading`, `ready`, or `failed`. Responds HTTP 200 when `ready`, 503 otherwise. |
| `POST /v1/audio/transcriptions` | Accepts `multipart/form-data` with one `file` part holding a WAV and optional `response_format=text`. Returns `{"text":"…","duration":17.54}` (seconds of audio), or plain text when requested. Other fields are ignored so OpenAI-style clients work. |

Failed requests return `{"error":{"code","message"}}`:

| Status | When |
| --- | --- |
| 400 | The request or WAV file is malformed. |
| 403 | The request came from another web page. |
| 408 | The request body did not arrive within one minute of its turn starting. |
| 413 | The request body is over 256 MiB. |
| 415 | The request is not `multipart/form-data`, the file is not a WAV, or its encoding is not supported. |
| 421 | The request used an address other than `127.0.0.1:<port>` (for example, `localhost`). |
| 422 | The audio is longer than 10 minutes. |
| 503 | The model is not ready, or one transcription is running and another is already waiting. |

Requests have to use the exact address the app prints (`127.0.0.1:<port>`, not `localhost`) and cannot come from another web page. Both checks exist so websites you visit in your browser can't call your local server. Nothing about the audio or its transcript is logged or written to disk.
