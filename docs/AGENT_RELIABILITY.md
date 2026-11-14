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
  final-answer recovery turn can follow an interrupted turn.
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
Nemotron calls are not replayed. The same answer schema, citation identities and
coordinate checks still apply. Invalid recovery answers yield explicit abstention.
A recovered answer is not independently verified merely because recovery worked.

Local algorithm and specialist operations retain their existing own timeouts.
The 600-second value is **not** a hard end-to-end deadline for the whole synchronous
endpoint: in-flight tools and cleanup can finish after admission closes. Gemini
I/O is cancellable, but client cancellation cannot guarantee provider-side billing
or computation stops immediately. Browser disconnect does not cancel the entire
host workflow.

## Traces and frontend integration

Existing answer fields remain compatible. The new `reliability` object contains:

- `provider_attempts`: attempt number, normal/recovery phase, duration, timeout,
  safe error class/category, HTTP status or finish reason where available, retry
  delay and provider-reported usage. No exception bodies, headers or credentials.
- `recovery_used`, `recovery_succeeded`: whether finalization was attempted and
  produced a validated non-abstaining answer.
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

The final local source passed 174 Python tests and eight browser checks. Four
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

## Sources checked 14 November 2026

Google recommends bounded exponential backoff with jitter for transient errors,
and distinguishes those from client/authentication errors:
[Gemini troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting).
SDK client lifecycle and async generation:
[Google Gen AI Python SDK](https://github.com/googleapis/python-genai).
Implementation was checked against the locally installed `google-genai 2.34.0`.
The transport omits `temperature`; current Google documentation states Gemini 4
ignores custom sampling values and warns they may become invalid. Model, tools,
image encoding, token limit and evidence validation remain otherwise unchanged.
