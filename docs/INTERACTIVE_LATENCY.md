# Interactive answer latency

The interactive API uses a bounded, faster agent profile. The latest
[mandatory parallel Nemotron routing](NEMOTRON_ORCHESTRATION.md) builds on this work;
algorithms remain optional. This document's two live timings measured the earlier
optional-specialist profile. No cached answer or change to original drawing pixels,
model weights or geometry rules is involved.

## Observed bottleneck

Two earlier stair questions on the same 10200 x 6600 dormitory sheet took 246.969 and
355.812 seconds. Algorithm calls consumed 89.125 and 212.469 seconds; the longer call
failed to supply evidence. Successful Gemini attempts consumed about 153 and 132
seconds respectively. Nemotron calls took about 1-2 seconds. Both answers counted four
stairs. The longer run was not primarily a Nemotron or frontend delay.

Previously the algorithm job cache used an ephemeral upload/document ID and held a
global lock while preparing the full ML/geometry bundle. Reopening a cached drawing
created a new ID, lost reuse, and could repeat the entire CPU pipeline.

## Implementation

- `algorithm_jobs.py` uses canonical RGB hash, dimensions and a recipe fingerprint
  (model/calibration hashes, implementation files, device and executable paths).
  Re-uploading identical pixels shares one future. A versioned disk index restores
  completed jobs only after checking source/model/profile identity and artifact hashes.
  Old unversioned jobs are not implicitly trusted. Results contain no reference labels.
- One CPU builder plus one waiting job bounds work. The lock only protects registration,
  not computation. At most eight retained futures; failed jobs have a cooldown. Different
  pages do not wait behind a global lock. Interactive tools wait up to 0.25 seconds for
  a future after fingerprint preparation, then return `pending`/`busy` and continue with
  clean crops. Pending evidence is never reported as zero objects or as delivered clues.
  The agent may revisit once later if useful; ready jobs retain the original catalog.
- Gemini Carbon uses `LOW` thinking for normal interactive turns. Output remains8192
  tokens, with16384 for token repair. Unknown models retain their existing thinking
  settings. Google documents thinking level as the latency control and cautions that
  lowering output limits can truncate reasoning/answers:
  [official thinking guide](https://aistudio.google.com/docs/thinking?generate-content=true),
  observed2026-11-14.
- Normal transport allows two attempts,35seconds per attempt,70seconds per turn,
  and10 total network requests. The interactive agent has a120second admission budget,
  reserving35seconds and at least one network request for finalization. Existing
  bounded retry, Retry-After, cancellation, token recovery, citation validation and
  local measurement recovery remain. This is not a guarantee that every upload,
  local operation or server queue completes within120seconds.
- The measured profile used15second Nemotron socket timeouts. The newer parallel
  inspection profile allows30seconds clipped to the remaining batch budget, more
  room-level calls, and one explicit failed-tile retry; see the routing document.
- Independent detail crops are encouraged in one model turn. A `reflect` followed by
  `submit_answer` can share a response when the self-check's next action is answer.
  Reflection is optional metadata: its formatting failure is recorded but cannot
  block a valid answer based on already supplied evidence. New crops, algorithms,
  specialist output and measurement results must still return before an answer can
  use them. Final answers always pass the original host validation.
- The live UI explicitly labels pending analysis and continues showing crop inspection.
  Late background evidence does not retroactively become evidence for a finished answer.

`DrawingAgent` accepts an explicit `AgentBudget`; its default remains600/120 for
existing offline callers. The web `ask` integration selects120/35. Algorithm preparation
is question-triggered, not automatic at upload. Background jobs retain their original
worker timeouts and finish locally to make later questions reusable. Cache ownership
is single-process; do not run multiple servers with the same cache root.

## Verification and limits

Two explicitly approved live runs used the unchanged dormitory sheet with the question:
"How many stairs are shown on this floor plan? Show their locations."

| Condition | End-to-end seconds | Returned answer | Citations | Algorithm wait |
| --- | ---: | --- | ---: | ---: |
| Fresh algorithm cache |72.922|4 stairs|4|0.688 +0.359s|
| Repeat after background completion, new upload ID |43.687|4 stairs|4|0.032s|

The first run recovered from a roughly35second provider server error; it used source
crops without algorithm output. The second received the cached algorithm catalog.
Both chose not to call Nemotron. They matched the prior count and cited the three
exterior stairs plus the interior stair; the four first-run source snippets were also
visually inspected. These are two development examples, not a general accuracy score,
latency percentile or guarantee. The original question wording was not retained in the
older run records, so this is a comparable stair-count task, not a controlled same-prompt
benchmark. The new timing includes agent setup; old result timings excluded some setup.

The repeat run spent an extra turn correcting an overlong reflection entry. The final
patch removes that formatting-only delay and passes regression tests; no third live
request was made and no additional speedup is claimed for that patch.

A separate entirely local cache check returned pending in1.938seconds including
initial fingerprint work, completed the CPU pipeline in another120.469seconds,
reused it in0.031seconds and restored it after a manager restart in0.203seconds.
The local cache test overlapped some live testing, so CPU contention may affect timing.
Neither result caching nor this optimization improves weak detector predictions.

Validation:331 Python backend/feedback/tool tests, TypeScript/build/Ruff and four focused
browser scenarios pass. Tests cover pending jobs, bounded queues, single-flight work,
failure reuse, disk restoration, model/source/size invalidation, corrupted artifacts,
finalization reserves, same-turn reflection, specialist limits and truthful live UI.
Private source drawings, cache artifacts, provider traces and screenshots remain ignored.
