---
name: consumer-api-plan
description: "The 10-step consumer API plan (WAV, cutter, microphone, exe, server, page, README) from council parakeet-consumer-api; owner's decisions that differ from the council design"
metadata:
  node_type: memory
  type: project
  originSessionId: c40f391e-1be9-43c4-862d-9ed89c580f21
  modified: 2026-11-24T23:02:40.809Z
---

Design: `C:\Users\Krzysiek\.councils\parakeet-consumer-api\result-session-1.md`, section E lists
steps 1–10, done one commit per step on `master`, each followed by a Codex pair review.

Owner decisions that override the council text (2026-11-23/24):
- WAV input is limited to 10 minutes (`WavAudio.LongestAudio`), not 2 hours.
- One `InvalidDataException` whose message names the case, not separate exception types.
- v1 stays Windows-only; Linux x64 CPU+WAV is a listed follow-up in AGENTS.md.
- No mid-way manual testing; see [[manual-acceptance-at-the-end]].
- Disconnect/cancel rule (2026-11-25): live, in-memory transcriptions (HTTP reply, record,
  TranscribeAsync) stop between 20 s pieces when the client is gone; a transcription whose result
  goes to a file must finish every piece. No current path writes to a file yet, so any future
  `--out`/batch feature must not pass the client's token into the piece loop.

**Why:** the owner wants a NuGet release of a consumer-friendly library and exe.
**How to apply:** check `git log` for which steps are done before starting the next; step 9 (the page)
needs an image mockup approved before any UI code.
