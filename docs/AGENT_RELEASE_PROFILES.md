# Agent production profiles

The deployment includes the latest scope/counting implementation while retaining
the strongest measured agent as its default. No frontend, model weights or drawing
processing changes are needed.

| Profile | Behavior | Evidence |
| --- | --- | --- |
| `qualified-v6` (default) | Qualified prompts, tool schemas, counting and measurement behavior, plus authentication failure handling | Historical fixed50: 39/50; original holdout: 20/30; additional confirmation: 14/20 |
| `scope-count-candidate` (opt-in) | Adds task scope propagation, localized context observations, cited scope exclusions, window/area guidance | Local contract tests; accuracy comparison was interrupted by provider authentication failures |

Configure `PLANPROOF_AGENT_PROFILE` in the server release's nonsecret
`deployment.env`. Compose defaults explicitly to `qualified-v6`. Do not turn on
the candidate solely because it is newer: freeze it and compare it on reference
questions before promoting it. It has no measured score yet. Unknown profiles
fail configuration rather than silently changing behavior.

Both profiles stop queued specialist requests and further coordinator turns after
an HTTP 401/403 rejection, retain sanitized diagnostics and abstain. Temporary
503 errors remain local to the affected tile. Required specialist evidence is
never bypassed. This is reliability handling, not a claim of new accuracy.

`GET /api/v1/agent` and saved question results identify the active profile.
The status endpoint reports configuration only, not provider availability.
Existing result fields, source coordinates, citations and saved histories remain
compatible with the frozen frontend.

Release guards compare two complete instruction hashes and 48 tool configurations
against frozen V6 revision `9909cb3`. The JSON fixture contains only hashes and
configuration choices, no drawings or evaluator answers. Behavioral tests cover
default counting, rejected opt-in fields, recovery context and candidate-enabled
scope/counting. These tests prove contract preservation, not model accuracy.

Deploy only committed source to the existing worker, verify the image and active
profile, and preserve the prior source/image and saved histories for rollback.
The production restart requires an idle queue and clears only the in-memory page
cache; browser drawings may need re-selection. See [deployment](VERCEL_DEPLOYMENT.md).
