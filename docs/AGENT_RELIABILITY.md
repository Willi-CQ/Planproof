# Agent interruption recovery

The 50-case development run recorded 15 adaptive failures in Gemini generation
or response handling. Only two were explicit read timeouts; seven were server
errors, four other transport errors and two generic processing errors. The old
traces do not contain HTTP statuses or finish reasons, so those causes cannot be
retroactively reconstructed.

## Implemented behavior

`backend/gemini_transport.py` owns model-call recovery. The SDK's internal retries
are disabled so all attempts share one budget and appear in telemetry.

- At most three HTTP attempts per normal turn, two in final-answer recovery and
  twelve total per question. The existing eight normal agent turns remain; one
  final-answer recovery turn can follow an interrupted turn. A distinct, single
  token-repair turn may follow `MAX_TOKENS`, including during outage recovery.
  Both recovery phases share the twelve-request and remaining-time budgets.
- A cancellable async request limits each generation attempt to 90 seconds;
  queueing, retry delays and attempts share a 240-second turn budget and the
  remaining question budget. Each attempt opens a fresh connection and closes
  both SDK clients. It does not leave a timed-out inference thread running.
- Retry only connection/network failures, timeouts and HTTP 408, 429, 500, 502,
  503 or 504. Exponential backoff starts at one second with jitter, capped at
  eight seconds unless the provider specifies a longer `Retry-After`. Numeric
  and HTTP-date headers are supported. Never retry earlier than that header or
  sleep beyond the remaining deadline.
- Two concurrent Gemini attempts per key/model per server process. Four
  consecutive transport/server failures impose a 20-second cooldown; provider
  `Retry-After` also applies across requests. During cooldown the API reports
  unavailable with a retry delay. This gate is not distributed across workers.
- Authentication, permission, invalid request and blocked-generation responses
  fail without automatic retries. No alternate model, key or endpoint is chosen.
- `MAX_TOKENS`, malformed tool calls and empty/incomplete responses are classified
  separately, with finish reasons retained. Partial responses cannot execute
  tools or become final answers.

The agent reserves the last 120 seconds of its 600-second admission budget for
finalization. After a recoverable interruption it can rebuild a fresh transcript
from the question, original page, registered crops and already observed tool
results. Only `submit_answer` is available in this recovery turn: algorithms and
Nemotron calls are not replayed in ordinary outage recovery. The same answer schema, citation identities and
coordinate checks still apply. Invalid recovery answers yield explicit abstention.
A recovered answer is not independently verified merely because recovery worked.

### Token exhaustion and bounded specialist delegation

One failed live trace consumed 8,189 thinking tokens and only three candidate
tokens under an 8,192-output-token cap. `MAX_TOKENS` reports a generation limit;
it does not establish exhausted API credits or an invalid key.

For the verified `gemini-4-carbon` model, ordinary finalization uses `LOW` thinking
with the existing 8,192 output limit. The distinct token-repair phase uses `LOW`
and a 16,384 output allowance. A live capability check confirmed that combination
is accepted and the model advertises a 32,768 output limit. Normal routing keeps
its previous settings. Unknown model IDs retain their existing settings; do not
apply an unverified model profile automatically. Numeric `thinking_budget` is
not sent. See Google's [thinking configuration](https://ai.google.dev/gemini-api/docs/thinking).

Token repair rebuilds a compact evidence transcript (12,000 JSON characters for
tool observations), with explicit omissions and unknown coverage. It preserves
the original question, units, available native text, full-page image and all
registered crops. Reflection and incomplete drafts are excluded from evidence.
Specialist observations remain atomic with their citations; truncated algorithm
candidate lists report additional omissions. No missing evidence becomes zero.

When the mode and user permissions allow Nemotron, the host can delegate up to
two local inspections before finalization. These are subquestions about the
original question, using existing, previously uninspected crops only. They do
not start another unrestricted Gemini agent or rerun algorithms. Each crop is
at most 75% of the page, uses the original canonical frame, and overlaps another
selected crop by at most 20% of the smaller area. These scheduling rules do not
prove complete coverage or that physical objects are disjoint. Gemini must
resolve overlap, scope and ambiguity; it must never blindly sum local counts.

Delegation consumes the existing three-specialist/twelve-tool budgets. Each call
uses request-local settings: 20-second network timeout, zero retries, 2,048 output
tokens. It is admitted only with at least 115 seconds remaining to reserve time
for finalization. The synchronous specialist timeout is an I/O timeout, not a
hard cancellation guarantee. No delegation occurs without an available Gemini
request or during its cooldown. Direct mode never delegates. Repeated token
exhaustion terminates with explicit unavailability; no partial answer is accepted.

Local algorithm and specialist operations retain their existing own timeouts.
The 600-second value is **not** a hard end-to-end deadline for the whole synchronous
endpoint: in-flight tools and cleanup can finish after admission closes. Gemini
I/O is cancellable, but client cancellation cannot guarantee provider-side billing
or computation stops immediately. Browser disconnect does not cancel the entire
host workflow.

## Traces and frontend integration

Existing answer fields remain compatible. The new `reliability` object contains:

- `provider_attempts`: attempt number, normal/recovery/token_repair phase, duration, timeout,
  safe error class/category, HTTP status or finish reason where available, retry
  delay, output allowance, thinking profile and provider-reported usage. No
  exception bodies, headers or credentials.
- `recovery_used`, `recovery_succeeded`: whether finalization was attempted and
  produced a validated non-abstaining answer.
- `token_repair_used`, `token_repair_succeeded`, `delegated_recovery_calls`,
  `token_repair_notes`: bounded token recovery and skipped-delegation reasons.
- `last_error`, `retryable`, `retry_after_seconds`: failure details and retry
  guidance. A successful answer may retain the error it recovered from.

An exhausted outage remains `status: unavailable` with `answer: null` inside the
answer object. Do not turn this into zero or keep a previous answer/citation on
screen. The local UI explains the interruption/cooldown and identifies recovered
answers. There is no automatic browser resubmission that could duplicate charges.

Separate atomic `.progress.json` files preserve observations while a question is
running. They are marked completed when the final trace is written. They are
diagnostic checkpoints, not an implemented restart/resume API. Never consume a
partial checkpoint as a scored final result. Token accounting prefers per-attempt
usage, including incomplete responses, and does not double-count successful turns.

## Verification

`backend/test_gemini_transport.py` injects transport and HTTP failures, validates
retry limits, terminal errors, Retry-After, cooldown/concurrency, deadline/queue
exhaustion, actual async cancellation and client cleanup, invalid final citations,
blocked/partial responses and evidence-preserving finalization. A real SDK test
uses a mock HTTP transport to prove there are no hidden extra requests.
`tools/test_question_agent_ui.mjs` tests outage/recovery rendering with explicitly
simulated responses; it does not claim real inference correctness.

For live development probes, use `tools/run_resilience_probe.py` with a private
input inventory and a fresh output directory. It exercises actual upload and ask
endpoints sequentially and never opens evaluator labels or retries completed
requests. Existing 50-case results and source data remain frozen. A small
time-separated rerun cannot isolate provider variability from code changes.

### Completed development checks, 14 November 2026

The earlier outage-only revision passed 174 Python tests and eight browser checks. Four
previously observed questions were run through the old and repaired live HTTP
pipelines, including actual algorithm and specialist calls when requested:

| Condition | Returned answers | Correct | Execution failures | Summed HTTP time |
| --- | ---: | ---: | ---: | ---: |
| Original transport | 1/4 | 1/4 | 3/4 | 1,113.6s |
| Recovery implementation | 3/4 | 2/4 | 1/4 | 959.3s |

One natural interruption involved two HTTP 504 responses and a local timeout.
Finalization reused the two crops and Nemotron feedback and produced an answer,
but the count disagreed with the frozen reference. This verifies recovery, not
correctness. Another question exhausted recovery with `MAX_TOKENS` after two
504s and a local timeout; the partial answer was rejected. The other new success
did not need retry, so provider variability cannot be separated from code changes.
All original 50-case scores remain unchanged.

A separate, explicitly injected first-turn connection failure followed by real
Gemini finalization succeeded in 13 seconds on a simple control. It is not evidence
of a natural outage. The four-case run used the first recovery revision; final
budget-edge, checkpoint and blocked-response refinements passed regression tests,
and the final code is active locally. Raw labels, drawings and probe traces stay
in ignored private artifacts. No broad accuracy or availability guarantee follows
from these small development checks.

### Token-repair development check

The token-repair revision passes 211 Python tests, nine browser checks and Ruff.
The local app was restarted with the tested implementation. A controlled replay used the
actual failed door-count drawing, its registered crops and previously observed
tool evidence, with a synthetic `MAX_TOKENS` trigger. It retained the remaining
171-second budget and seven already consumed Gemini requests. Evaluator answers
were not loaded during inference. This is not a fresh end-to-end benchmark.

One new crop-local Nemotron inspection and real Gemini finalization completed in
20.2 seconds. Gemini returned a complete cited answer using 897 candidate tokens;
thinking usage was not separately reported. The count was **13 versus the frozen
reference's 10**, so execution recovery succeeded but semantic accuracy did not.
The model included adjacent partial units and acknowledged clipped symbols;
resolving scope and recognition belongs to the separate hard-drawing review.
The original failed trace and original 50-case scores are preserved unchanged.

## Sources checked 14 November 2026

Google recommends bounded exponential backoff with jitter for transient errors,
and distinguishes those from client/authentication errors:
[Gemini troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting).
SDK client lifecycle and async generation:
[Google Gen AI Python SDK](https://github.com/googleapis/python-genai).
Implementation was checked against the locally installed `google-genai 2.34.0`.
The transport omits `temperature`; current Google documentation states Gemini 4
ignores custom sampling values and warns they may become invalid. Model, tools,
image encoding and evidence validation remain unchanged. The token-repair profile
above is the only increased output allowance.
