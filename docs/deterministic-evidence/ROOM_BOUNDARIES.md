# Room boundaries and local question context

This optional postprocessor uses frozen ML wall/room proposals and the existing
Track 1 pixels to prepare smaller search regions. It does not generate BIM, run
another ML model, call an LLM, or assert that a region is a verified physical room.
The original `evidence.json`, predictions, drawing and labels remain unchanged.

## Rules

1. Rasterize the selected ML wall polygons in the registered working frame.
   Snap boundaries only within `min(2 * stroke, .25 * wall thickness)` using the
   existing validity/area/hole guards. Paired outlines may fill a wall body only
   when at least 90% of their band is near the ML wall proposal.
2. Search opposing skeleton terminals, not arbitrary Hough line ends. Require
   facing axes within 10 degrees, lateral offset at most half wall thickness,
   comparable thicknesses, supported endcaps, and low intervening wall occupancy.
   Search lengths are .5–12 local thicknesses. These are configurable image
   heuristics, not construction standards or metric measurements.
3. Also scan cross-sections along ML wall axes. A narrow hollow span between two
   solid, ML-supported flanks may be an opening even when thin frame lines keep
   the skeleton connected. Require at least .65 flank fill and at most .35 span
   fill. Furniture lines alone do not establish a wall gap.
4. Within each supported gap, compare rotated local templates. A door needs an
   attached leaf and curved swing; test 60/75/90/105-degree swings and full/half
   spans. A window needs 2–4 sustained parallel stroke bands with supported
   endcaps. A clear passage needs low central ink and no strong arc. Conflicting
   or incomplete symbols remain unknown. A missing door symbol can look like a
   walkthrough; access is never verified by this rule.
5. Insert separately typed **virtual separators** across candidate gaps for
   free-space partitioning. They never become physical-wall observations.
   Image-border-connected free space supplies possible exterior context. If the
   wall reaches the image edge, exterior classification is withheld.
6. Classify wall sections by sampling both sides: two interior sides suggest an
   inner wall; one interior and one exterior side suggest an outer wall. Mixed,
   unsupported or cropped cases remain unknown. These roles depend on ML-assisted
   partitions and are not independent confirmation of the model.
7. Keep each ML room's identity and multipart geometry. A bounded component is
   offered as an alternative only when seed/component IoU is at least .80 and area ratio is
   .8–1.2. Multiple seeds sharing open space remain separate ML hypotheses with a
   shared-space warning; the larger enclosure is stored separately. No invented
   divider or automatic merged room replaces them. Primary geometry retains the
   ML seed by default. `apply_refinements: true` enables the experimental replacement
   explicitly; it is not the default because development area errors regressed.
8. Export boundary segments as wall-supported, virtual or unresolved, with source
   dependencies, opening evidence, possible adjacency and competing ML hints.
   Confidence-like support ratios are geometric measurements, not probabilities.

## Host API and CLI

```python
from planproof_evidence import build_room_boundaries, get_room_context

bundle = build_room_boundaries("job/evidence.json", "job/rooms", model_source="main")
room = bundle["pages"][0]["rooms"][0]
get_room_context("job/rooms/room-boundaries.json", room["id"], "job/crops/room-1")
```

```powershell
planproof-evidence rooms job/evidence.json --out job/rooms --model main
planproof-evidence room-crop job/rooms/room-boundaries.json --room-id REGION_ID --out job/crops/room-1
```

The bundle records the source evidence SHA, consumed artifact hashes, implementation
hash, rule parameters and coordinate transform. Modified/unregistered source
artifacts are rejected. `--rules rules.json` accepts `RoomRules` fields.

The room crop contains `clean.png`, a separate `room-overlay.png`, `region.json`
and `agent-context.json`. Context includes the room, related walls/openings,
separators, neighbors and explicit limitations. `source_evidence_context` also
retains the ordinary crop's localized original ML and independent observations;
unattached ML doors/windows are not lost when deterministic rules miss them.
Margin is the larger of two wall
thicknesses and 5% of the room's long side; attached door swings expand it. Source
pixels are preserved. A host should register room IDs and expose only those IDs
to Nemotron/Gemini, then verify answers against the returned clean crop.

## Development feedback loop

The fixed simple case is P05-resplan and the complex case P01-cubicasa5k. Run
inference first, then the separate label evaluator; inspect both cases visually
and all ten cases numerically before retaining a change:

```powershell
.venv/Scripts/python tools/run_room_boundaries.py --pack .artifacts/reference10-integrated-v1
.venv/Scripts/python tools/build_evidence_lab.py --pack .artifacts/reference10-integrated-v1
node tools/test_evidence_lab.mjs .artifacts/reference10-integrated-v1
.venv/Scripts/python tools/review_room_feedback.py --pack .artifacts/reference10-integrated-v1 --out .artifacts/room-rules-feedback/NEW_ITERATION
```

`ROOM_BOUNDARY_REPORT.json` compares refined room F1 with the frozen ML baseline,
including unresolved fallbacks. Eight plans have room annotations; two CAD plans
remain N/A. The original failure report is never overwritten. Rule outputs and
the report are local artifacts, not committed data. The Evidence Lab's **Room
boundaries** view provides role/type toggles, region selection, clean crops,
per-plan scores and inspectable rule evidence.

The first experiment was rejected: unrestricted components and splitting multipart
fallbacks reduced pooled room F1 from .6845 to .5476. The resulting safeguard
preserves ML instances and treats partition enclosures as separate hypotheses.
The second experiment preserved room F1 but worsened matched-room area errors
on several plans (e.g. P06 .0339 to .0689). Therefore the default keeps the ML
geometry and exports supported alternatives separately. Visual and numerical
receipts, original sources and checksums are preserved for each local iteration.
This is an engineering feedback loop, not an executed Nemotron self-review.

The final conservative mode preserves room F1 on all eight annotated plans:
pooled .6845 (64 TP / 46 FP / 13 FN), simple P05 .7500, complex P01 .7429.
Eighteen supported boundary alternatives are available across the ten drawings.
Direct geometry checks compare every retained primary region to its imported ML
parent. Small residual area-score differences versus raw frozen predictions can
remain from the existing import normalization/raster representation; the feedback
receipt reports them rather than silently rounding them away.

Opening strips are separately scored at box IoU .5: windows 27 TP / 35 FP /
83 FN, F1 .3140; doors 0 TP / 10 FP / 108 FN, F1 .0. These rules therefore do
**not** replace the stronger ML opening detector. There are 58 recorded ML/rule
class conflicts. Window performance exceeds the earlier independent rule engine
on this cohort (.0819), but this new stage uses ML wall proposals and is not an
independent confirmation. Door candidate generation and symbol attachment remain
weak on real plans despite passing controlled rotated-symbol tests.

## Limits

This repeatedly inspected development cohort cannot establish unseen Stoa answer
accuracy. Window frames and simplified door slots may be indistinguishable;
opening classes remain hypotheses even when pixels and ML agree. Inner/outer wall
and walkthrough accuracy is N/A without suitable annotations. An unchanged room
F1 demonstrates preservation at that threshold, not improved recognition or area
accuracy. Room categories, stairs, metric dimensions and exact physical counts
are not added by this postprocessor. Missing ML wall coverage cannot be repaired
by promoting every drawing stroke to a wall.
