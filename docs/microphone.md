# Microphone capture

`Microphone.StartRecording()` records into memory and `MicrophoneRecording.Stop()` returns 16 kHz mono
samples for `SpeechEngine.TranscribeAsync`, which cuts audio longer than 20 s (see
[model-and-decoding.md](model-and-decoding.md#long-audio)). Nothing is written to disk. It is
push-to-talk: one recording, then one transcription.

## How it works

- WASAPI in shared mode, reached through raw COM vtable calls on `ole32.dll`, as the GPU path
  reaches Direct3D 12. There is no package dependency.
- The client asks Windows for 32-bit float, 1 channel, 16,000 Hz, with
  `AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY`, so Windows converts
  from the microphone's own format. The buffer holds 1 second.
- Each recording has its own background thread, which initialises COM (multithreaded), opens the
  device, and drains packets every 20 ms. All COM calls for a recording happen on that thread.
  `Microphone.Devices()` runs on a short-lived thread of its own, so a caller's COM apartment
  (for example a UI thread's STA) does not matter.
- A packet flagged silent is kept as zeros.
- `Level` is the RMS of the latest packet, at most 1. `IsRecording` turns false as soon as capture
  ends, before the device has finished closing.

## Rules

| Rule | How |
| --- | --- |
| The cap is exact | `longest` is converted to samples (rounded down, without overflow for any `TimeSpan`), and the packet that crosses it is cut to fit. The thread then ends. |
| No exception leaves the capture thread | Failures are kept and rethrown: from `StartRecording` when opening fails, from `Stop` as `IOException` when the device fails later. |
| Start and stop are bounded | `StartRecording` waits at most 5 s for the device to open, `Stop` at most 5 s for the thread to end; both then throw `IOException`. `Dispose` waits at most 5 s and drops the audio either way; a second `Dispose` returns at once. |
| Audio does not linger | `Dispose` zeroes every kept packet. A device that opens after `StartRecording` gave up keeps nothing: its packets are zeroed as they arrive. |

## Errors

| Situation | Exception and message |
| --- | --- |
| No microphone | `IOException`: "Windows has no microphone to record from." |
| The chosen microphone is gone | `IOException`: "That microphone is no longer connected." |
| Privacy settings block desktop apps | `UnauthorizedAccessException`, naming Settings > Privacy & security > Microphone |
| Unplugged while recording | `IsRecording` turns false; `Stop` throws `IOException`: "The microphone stopped working: The microphone was unplugged or turned off." |
| Any other WASAPI failure, including from releasing a packet | `IOException` naming the step and the HRESULT |

## Tests

`MicrophoneRecordingTests` cover the rules above with a fake packet source, on any machine.
`MicrophoneTests` use the real device and skip when Windows lists no microphone or cannot list
them (as on a machine without the audio service), so a broken device list also shows as skips. Not covered by any automated test: real speech, the privacy block, and unplugging
during a recording.
