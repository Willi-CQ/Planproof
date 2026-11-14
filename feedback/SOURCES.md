# Sources and access observations

Observed 14 November 2026, approximately 10:05–10:30 UTC. These establish source
behavior and local preparation, not authenticated model access.

| Source | Status and use | Limit |
| --- | --- | --- |
| Owner-supplied `https://containers.datacrunch.io/nemotron-3-5-super-vl-b300/v1`, model `nemotron-3.5-super-vl` | **Reported** on 14 November. Adapter uses `max_tokens` and final `message.content`; it does not substitute `reasoning_content` for answers. | Full key is missing. Tool/image support, limits, service availability, allocation and billing remain unverified. The OpenCode example is not an access test. |
| [Verda OpenAI compatibility](https://docs.verda.com/inference/openai-compatibility/) | **Confirmed documentation:** OpenAI-compatible inference. | General inference docs use another URL; the adapter retains the owner's specific deployment. Optional JSON-schema/reasoning controls are not assumed. |
| [Verda API credentials](https://docs.verda.com/welcome-to-verda/api-credentials/) | **Confirmed documentation:** inference keys differ from Cloud API credentials. | No provisioning or account changes were performed. |
| [Junction Platform](https://hackjunction.app/hackathons/junction-2026) | **Confirmed public page:** accessed 14 November, 10:05 UTC; event dated 13–15 November 2026. | No authenticated Stoa partner brief/participant guide exposed. Owner-provided challenge wording remains reported; current participant/submission rules remain an external publication gate. |
| [Junction event site](https://2026.hackjunction.com/) | **Confirmed public organizer material:** accessed 14 November. | Does not verify the Stoa judging rubric or this team's partner access. |
| [Owner-supplied Stoa archive](https://fwjydgxpttajiqpfmqtj.supabase.co/storage/v1/object/public/stoa-junction-2026/junction-dataset.zip) | **Confirmed artifact:** HTTP 206 range access. Selected source bytes, original README and three source notices fetched and hashed; ZIP CRC32 verified. README defines real question fields and exact-count/2%-quantity scoring, alternate-format basenames and licenses. | No official text scorer provided. IFC floor drawings are not assumed equivalent to whole models. Questions/drawings remain local. |
| Existing private `stoa-client-eval-reference.jsonl` in the publication checkout | **Confirmed local input:** 101,635 actual rows and actual schema. Selection uses question identity, not answers. | Final partition never read; no training or tuning use. Source/pretraining overlap unknown. |
| Local `planproof_evidence.pipeline.get_region` | **Confirmed source inspection:** clean crop, SHA256, affine crop-to-canonical transform, IDs and optional `planproof-agent-context-v1` context. | Geometry is on another active local branch. This package consumes the artifact contract without overwriting that worktree or importing its dependencies. |

The ignored local pack manifest and readiness receipt contain exact hashes,
source notices and render checks. No live accuracy result exists. Self-review and
model citations remain advisory until checked against evidence or reference labels.
