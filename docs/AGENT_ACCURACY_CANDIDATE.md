# Scope and count-evidence candidate

## Production integration

This source is included in main behind `PLANPROOF_AGENT_PROFILE=scope-count-candidate`.
The production default is `qualified-v6`, preserving the measured V6 prompts,
tool schemas and counting behavior. Authentication failure handling is active in
both profiles. See [release profiles](AGENT_RELEASE_PROFILES.md). The candidate's
recognition improvements remain unqualified; its inclusion is not an accuracy claim.

This backend candidate is based on the qualified V6 agent. The product UI, detector
weights, drawing pixels, count scoring and area tolerance are unchanged. It is not
qualified for activation: the live diagnostic comparison was interrupted by a
Nemotron authentication failure in both the frozen control and candidate.

## Behavior

The coordinator distinguishes an explicit whole-sheet request from a named or
visually selected apartment/room. An optional `task_scope` on `inspect_regions`
records the target, interpretation basis, exclusions and counting unit without
adding a tool call. Workers receive that interpretation as advisory data. Recovery
retains it. A changed scope invalidates cached context inspections. The scope does
not itself filter objects or prove apartment membership; a local crop need not
contain the global apartment label to identify a physical symbol.

For door questions, a context inspection still answers the focused question. It
also requests structured locations for doors it actually identifies. Valid objects
enter the shared inventory; prose numbers, missing object lists and invalid boxes
do not. A context reply never establishes complete counting coverage or overwrites
a full count record. Duplicates and disagreement retain the existing conservative
handling. Objects outside successfully counted regions require a targeted count
inspection before a complete total is accepted. This addresses the observed
failure where several doors were described but only one entered the inventory.

For a selected unit or room, `count_basis.exclusions` can identify an observed
object outside the target. Each exclusion needs a concrete reason and valid,
previously inspected citations covering the object. Unknown/duplicate IDs,
unrelated citations and whole-sheet/unresolved scope are rejected. Raw observations
and excluded objects remain recorded; a failed draft restores the whole inventory.
Exclusions never make incomplete count coverage complete. This is an auditable
model scope judgment, not independent proof of apartment membership.

Window guidance separates openings, window units and panes; it does not impose a
universal convention or manufacture extra objects. Area guidance prioritizes the
room list, missing measurements, holes and final aggregation. These prompt changes
have no measured accuracy gain yet. This increment does not introduce a generalized
window inventory or new area geometry algorithm.

## Authentication and evaluation controls

Specialist HTTP 401/403 statuses survive error sanitization without retaining
provider bodies or credentials. A rejected credential stops queued workers and
further coordinator turns in that run. Temporary 503 failures do not cancel valid
neighboring inspections. Mandatory specialist evidence is not bypassed.

The benchmark runner accepts `--admission-deadline-utc` with an explicit timezone.
The cutoff is bound into the immutable trial receipt and checked before upload and
before paid admission. In-flight runs retain their original records. A separate,
private server watchdog bounded this round to the owner's one-hour limit; it
excludes production containers.

## Validation and release status

- Full local suite: 528 tests passed, including the cited-exclusion increment.
- Lint passes for changed code. Frontend source is unchanged from the filming release.
- Synthetic contracts exercise scope propagation/correction, unchanged context
  questions, source-coordinate and unit checks, duplicate preservation, refusal to
  promote prose counts, uncovered-object abstention, cited exclusions and their
  rejection/reset behavior, authentication fail-fast and transient-error isolation.
  They do not measure recognition accuracy.
- Diagnostic selection: all 11 failed questions from the original 50 plus five
  previously correct controls. 5 control and 6 scope-only attempts completed before
  admissions were stopped. All returned abstentions after specialist unavailability.
- Read-only availability checks from the isolated control, live product and local
  configuration returned HTTP 401. The owner reported no replacement key available.
  No comparison score or improvement claim is justified by these outage runs.
- The complete count-handoff candidate has not been sent to the providers. No fresh
  holdout inference or automatic production promotion occurred.

Resume only after authorized credentials work. Preserve failed attempts, freeze the
candidate revision/image, create new trial directories, and run matched comparisons.
Do not overwrite/relabel outage runs or present the previously inspected held-outs
as untouched validation. Private drawings, evaluator references, logs, API keys and
the personal knowledge graph remain outside GitHub.
