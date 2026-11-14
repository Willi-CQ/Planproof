# Evidence Lab result review

Observed 14 November 2026. Independent read-only review of the local
`reference10-integrated-v1` pack, its `failure-report.json`,
`failure-table.csv`, per-case evidence and source/classical-overlay images.
No inference, training, classifier tuning or label edits were performed.

## Measured result

These are the same ten selected development drawings, not a representative final
test or a combined question-answering score. Room matching uses instance-mask
IoU; door/window/stair matching uses box IoU, each at 0.5. Wall IoU is the mean
over eight annotated drawings. Object F1 pools TP/FP/FN over annotated cases.

| Class | Main ML | Deterministic | Metric |
| --- | ---: | ---: | --- |
| Walls | 0.7356 | 0.5336 | Mean mask IoU |
| Rooms | 0.6845 | 0.4480 | F1 at IoU 0.5 |
| Doors | 0.4270 | 0.0252 | F1 at IoU 0.5 |
| Windows | 0.5890 | 0.0819 | F1 at IoU 0.5 |
| Stairs | 0.0952 | 0.0000 | F1 at IoU 0.5 |

The deterministic pipeline is presently useful as inspectable structural
evidence, especially on clean filled-wall drawings. It is not a dependable
general-purpose semantic detector. Main ML is stronger overall but still
overproduces room/window proposals and barely detects stairs on this cohort.

## 1. Outlined walls and non-building strokes are the main structural weakness

**Observed:** deterministic mean wall IoU is 0.2271 on the four CubiCasa plans
versus 0.8401 on the four clean ResPlan renders. Only one of the 40 labeled
CubiCasa rooms is matched; 27 of 37 ResPlan rooms are matched. On P01, wall IoU
is 0.1979, no room is recovered, and all 41 window proposals are false positives
under this evaluator. Its overlay visibly labels section arrows and small text
strokes as walls, and several outlined internal wall segments as windows.

**Code evidence:** `pixels.vectorize_pixels` only fills paired outlines when
their interior ink support is at least 0.55. Hollow wall bands remain diagnostics
rather than accepted filled walls. Room extraction uses enclosed free space of
the accepted wall mask. These choices explain why a filled-wall render is a much
better fit than hollow, hatched or fragmented outlines.

**Next experiment:** separate wall-band construction from symbol/dimension/text
strokes using boundary continuity, thickness consistency and junction context.
Keep uncertain closure hypotheses separate. Compare source-stratified wall IoU
and room instance F1; do not optimize solely for the already strong ResPlan view.

## 2. Doors and windows need representation-aware rules

**Observed:** deterministic doors are 2 TP / 49 FP / 106 FN; windows are
12 TP / 171 FP / 98 FN. All 37 ResPlan doors are missed despite good wall masks.
P05 has 6 room matches and wall IoU 0.8230, yet misses all 10 labeled doors.
P02 generates 54 window proposals but only one match. A near-correct total count
can therefore hide poor localization; show TP, FP and FN, not just count error.

**Code evidence:** door availability is explicitly `swing_symbols_only`; window
recognition uses parallel strokes/bands. The ResPlan source view does not provide
the conventional swing arcs this detector expects. P01's parallel wall outlines
are visibly confused with glazing.

**Hypothesis to test:** distinguish opening gaps, door leaves/swings and glazing
as separate evidence types before classification. For windows, require local
wall attachment, bounded glazing extent and wall-versus-symbol continuity checks.
For doors, add bounded gap/leaf evidence rather than relaxing arc thresholds
globally. Existing raw Track1 shape candidates alone do not establish a class.

## 3. Stair recall is absent; zero proposals is not a successful result

**Observed:** deterministic stairs are 0 TP / 0 FP / 8 FN on annotated cases;
main ML is 1 TP / 12 FP / 7 FN. P09 visibly contains a large stair flight, but
the deterministic overlay has no stair candidate. P05/P06 source renders do not
visibly show conventional repeated stair treads, although each has a stair label.
Their exact render/annotation relationship deserves review before assuming all
misses can be fixed by a tread recognizer.

**Code evidence:** `pixels.repeated_parallel_symbols` requires at least five
distinct treads, spacing coefficient of variation <= 0.3, at least two
perpendicular rail segments, sufficient common overlap and similar lengths.
These are sensible hatch-rejection constraints but none produces an accepted
stair on this cohort. P09's saved late-stage abstentions contain only groups of
three or four lines, including small central edge traces, not the complete
visible stair flight. Earlier exits (spacing/overlap/length) are not individually
logged, so the exact rejection gate for the complete flight is **not proven**.
The raw Track1 `stair_candidate` in P09 covers the entire 1000 x 1000 page and is
correctly kept unclassified.

**Next experiment:** instrument every stair rejection gate; inspect a fixed crop
of P09 and both CubiCasa stair pages. Group fragmented tread segments across
occlusion/arrow strokes and test rail association on the complete flight. Do not
simply reduce the five-tread threshold: furniture and hatching are confounders.
Keep a separate diagnostic category for stair-like repetition that is not yet a
semantic detection.

## 4. Agreement can reproduce the same mistake

**Observed shared error:** P05 classical room
`p0:classical:5c10a40d668a899f` (overlay C6) and main ML room
`p0:ml-main:df7627a5458eb8f4` (source `p0-room-00025`) agree at polygon IoU
**0.7743**, bbox `[230.5, 444, 358, 582]`. Both are false positives according to
the labeled-room evaluator. Both methods also miss the same three ground-truth
room regions on this drawing. This is a concrete shared error, not a hypothetical
warning. The labels themselves have not been newly manually verified.

Main/classical room agreements on P05 total seven while each method matches only
six labeled rooms. Main/classical agreement counts across the whole pack are
2 wall, 30 room, 1 door and 20 window pairs, with no stair pairs. Those 53 pairs
are not 53 verified objects or a fused accuracy score.

**Display rule:** separate agreement IoU from ground-truth performance and from
per-method support/confidence. Mark green as “methods agree,” not “correct.”
Provide both source outlines and a clean crop. When available, evaluator-only
false-positive markers should remain a separate review layer. A classical miss
cannot veto ML, given the measured recall deficits.

## 5. The CAD specialist has a distinct, limited role

**Observed:** on P09, the CAD specialist matches all four doors with no door FP
and matches the one stair with one extra stair proposal. Main ML matches neither
the four doors nor the stair. On P10, the CAD specialist has 3 door TP / 16 FP /
10 FN: it is not consistently strong even across these two cases. Pooling its
predictions into the main model's counts or pretending it predicts walls/rooms
would obscure this result.

Keep its two-case scores separate, and mark wall/room coverage unavailable. Its
successful P09 stair is a useful localized comparison example, not evidence of
generalized stair reliability.

## Priority order

1. Make class/source toggles, same-cohort scores, error counts and agreement
   semantics explicit in the reviewer. This improves diagnosis without changing
   frozen predictions or flattering the displayed results.
2. Repair wall-band interpretation on outlines/scans and exclusion of annotation
   strokes; measure whether room topology improves without regressing clean plans.
3. Add representation-aware opening evidence and reduce wall-as-window errors.
4. Instrument stair grouping/rejection, inspect visibility of labeled stair
   regions, and validate any revised recognizer against hatch/furniture negatives.
5. Address ML proposal duplication/fragmentation with a separate development
   experiment. P02 has 28 room proposals for 16 matched rooms; P04 has 63 window
   proposals with 23 matches and 40 false positives. A score threshold change
   needs measured precision/recall tradeoffs, not visual preference alone.

OCR is unavailable in this pack; deterministic room categories consequently
remain unknown. Metric scale, physical counts, final Stoa QA and combined-agent
accuracy are unverified. Missing stair annotations on P07/P08 and missing CAD
wall/room annotations must remain N/A rather than zero or perfect scores.
