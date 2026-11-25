# Handoff: consumer API, step 9 onward

Written 2026-11-25 on Krzysiek-PC so the work can continue on another machine.
`master` at `2dc9420` holds steps 1–8. This branch only adds this `handoff/` folder. Delete the branch once the
work has moved, and never merge it into `master`.

## Where the work stopped

- **Steps 1–8 are done.** Each one is a commit on `master`, followed by a commit that fixes what its Codex pair
  review found. Step 8 is the `serve` API, and all 230 tests passed.
- **Step 9 (the page)** waits for the owner to approve the mockup "Page 1 · v1"
  (`council/mockups/page-1-v1.png`). No UI code has been written.
- **Undrawn states,** proposed alongside the mockup and part of the same approval. They reuse components the board
  already draws:

  | State (from `/health`) | How it looks |
  | --- | --- |
  | `loading` | Grey status pill "Loading model…"; drop zone greyed, as in #7 |
  | `failed` | Red status pill "Failed"; the error text in the red box from #9, plus the Download button from #7 when the model is missing |
  | 503 busy / 422 too long | The red message box from #9 with the server's own message |
  | Transcribing | The drop zone reads "Transcribing <file>…" until the answer arrives |

- **The board's placeholders:** "0.1.0" becomes the real build version, the file name and transcript are sample
  content, and the timing shown is a GPU timing.

## What's next

1. **Show the mockup.** Show `council/mockups/page-1-v1.png` to the owner and get approval or callout-numbered
   changes, for example "Page 1, #6: drop the curl box". Use a new image revision for every change.
2. **Build the page (step 9).** It is one embedded HTML file with no external assets, served at `GET /` by the
   same server.
   - It calls the API with its own Origin and adds `X-Parakeet-Request: 1` for the download.
   - `parakeet-runtime` with no arguments, and `serve`, open the page in the browser; `--no-browser` turns that off.
   - Check every state at desktop and narrow widths, keyboard use of the drop zone and buttons, and contrast.
   - Add a "Step 9 · Page" section to `manual-acceptance.md`.
   - Get a Codex pair review, then commit to `master`.
3. **Step 10.** Update the README and docs following the council's outline (section D in
   `council/result-session-1.md`), and re-measure the speed figures into `docs/benchmarks.md`.
4. **Step 11 (a separate release step).** Try single-file compression and keep it only if the conditions in
   section E, step 11 hold.
5. **Manual acceptance.** The owner runs the whole of `council/manual-acceptance.md` against the release build in
   one session, then signs off the first NuGet release.

## Owner decisions that override the council text

These are also in `claude-memory/consumer-api-plan.md`.

- WAV input is limited to 10 minutes (`WavAudio.LongestAudio`), not 2 hours.
- Bad input raises one `InvalidDataException` whose message names the case, not separate exception types.
- v1 is Windows-only. Linux x64 CPU+WAV is a listed follow-up in `AGENTS.md`.
- No manual testing midway: collect every manual check into `manual-acceptance.md` for the final session.
- When a client disconnects, live in-memory transcriptions (HTTP reply, `record`, `TranscribeAsync`) stop between
  20 s pieces. A transcription whose result goes to a file must finish every piece. No current path writes to a
  file, so a future `--out` or batch feature must not pass the client's token into the piece loop.

## Restore on the new machine

The original files lived outside the repository. Copy them back before continuing:

- `handoff/council/*` → `%USERPROFILE%\.councils\parakeet-consumer-api\`. The council members' `proto/` folders
  were build outputs, about 1.8 GB, and were left behind.
- `handoff/claude-memory/*.md` → the Claude Code memory folder of this repository's project, under
  `%USERPROFILE%\.claude\projects\<checkout path with separators turned into dashes>\memory\`.
- Change `C:\Users\Krzysiek\…` in those files to the new machine's paths.
