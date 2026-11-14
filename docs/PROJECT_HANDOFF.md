# Consolidated project handoff

The completed ML, deterministic evidence and Nemotron feedback work is collected
on `main`. Package boundaries and existing behavior are preserved by this merge.
The root README remains empty at the owner's request.

| Component | Entry point |
| --- | --- |
| Training, inference, calibration and export | [ML package](../ml/README.md) |
| Frozen weights, training checkpoints and restore checks | [Baseline v1](../models/baselines/v1/README.md) |
| Pixel extraction, geometry and ML comparisons | [Deterministic package](../deterministic/README.md) |
| ML-assisted room boundaries and local context | [Room rules and measured limits](deterministic-evidence/ROOM_BOUNDARIES.md) |
| Visual and agent self-feedback | [Nemotron package](../feedback/README.md) |
| Question comparison and artifact preservation | [Comparison contract](../feedback/COMPARISON.md) |

## Source preservation

Merge ancestry retains the deterministic handoff, element-classification work,
integrated Evidence Lab, ML training pipeline, frozen baseline v1 and Nemotron
feedback tooling. The immutable baseline payload and checksums are unchanged.
The `ml/trained-baseline-v1` release branch remains available for existing restore
instructions. Superseded implementation branches can be removed after main's
remote commit is verified; teammates' unmerged branches remain separate.

Datasets, evaluation drawings/labels, generated dashboards and run archives stay
local. Only the explicitly approved model baseline is included through Git LFS.
Private instructions, knowledge files and real credentials are excluded; the
empty `.env.example` describes the credential variable to set locally.

## Validation and next connection

Run each package's existing tests in its documented environment. Dependency
versions differ between packages, so separate environments remain supported.
`python ml/scripts/verify_baseline.py` verifies all frozen model payloads.

The question-aware Gemini coordinator is now implemented in this integration
branch. It chooses direct vision, clean crops, Nemotron inspection/review and
optional ML/geometry tools. See [QUESTION_AGENT.md](QUESTION_AGENT.md) for the
architecture, local launch, API, execution budgets and three-mode comparison.
The feedback adapter also accepts the newer room-context schema. Live QA scores
for this coordinator must be measured separately from earlier Nemotron pilots.

Room rules retain imported ML geometry by default, with deterministic boundaries
as alternatives. Door rules remain weak and must not suppress stronger ML
opening evidence. Component development scores are documented in their guides;
they do not establish final Stoa question-answering accuracy.
