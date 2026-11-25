---
name: manual-acceptance-at-the-end
description: "Owner wants no mid-way manual tests; one big manual testing session before NuGet sign-off, checklist kept in the council folder"
metadata:
  node_type: memory
  type: feedback
  originSessionId: c40f391e-1be9-43c4-862d-9ed89c580f21
  modified: 2026-11-24T22:12:06.275Z
---

Don't ask the owner to try things manually step by step. Collect every manual check (real speech,
unplugging, privacy settings, real files) into one acceptance session they run at the end, before
signing off and pushing the NuGet version. The checklist is
`C:\Users\Krzysiek\.councils\parakeet-consumer-api\manual-acceptance.md`; each step of the consumer
API plan appends its own section.

**Why:** said on 2026-11-24 ("no need for testing please prepare one huge testing session at the end
before I sign off the whole thing before pushing the nuget version").

**How to apply:** when a step has something only a person can verify, add it to that file and list it
under "Not verified" in the report; don't offer mid-way manual test options. Related:
[[consumer-api-plan]].
