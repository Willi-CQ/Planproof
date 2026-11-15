# Agent evaluation — 15 November 2026

The selected backend is `9909cb35d4c7b82227a590861b71354e7e3eb60f`
(V6). It improves the fixed regression score and a separately frozen holdout,
with essentially unchanged question latency. It does not solve reliable global
door/window counting or guarantee correct geometric calibration. The frontend
tree remains `77ec18603f3ec800b42a5bcd556a9e412689fd5b`.

## Matched results

| Cohort | Fresh deployed control | Selected candidate | Median question time, control → candidate |
| --- | ---: | ---: | ---: |
| Fixed regression, 50 drawings | 34/50 (68%) | 39/50 (78%) | 47.8 → 48.3 s |
| Frozen holdout, 30 new drawings | 18/30 (60%) | 20/30 (66.7%) | 52.2 → 49.9 s |
| Development, 30 other drawings | 17/30 | 21/30, V5 diagnostic | 52.9 → 50.4 s |

These are single stochastic passes through production upload/run/poll routes on
the existing A100. Question timing starts after upload and frame preparation;
it excludes upload/render latency. Provider conditions and shared load vary.
The V5 development result is not a measured V6 development result.

The largest improvement is area questions: **4/12 → 10/12** on regression and
**4/8 → 7/8** on holdout. On holdout, openings fell from **2/6 to 1/6** and
apartments from **3/3 to 2/3**. All errors and abstentions remain in denominators.
V6 produced six wrong numeric answers plus five abstentions on regression,
and seven wrong numeric answers plus three abstentions on holdout.

The paired discordance checks are p=0.180 and p=0.688, respectively. These small
samples support a measured improvement on these runs, not a statistically
established general accuracy gain. The earlier historical 33/50 run is preserved
but is not the fresh 34/50 control used here.

## What changed

- Bounded native PDF text helps locate labels without treating file text as
  instructions or trusting its semantic interpretation.
- Native straight, filled PDF paths supply candidate boundaries. Calibrated area
  tools require cited scale anchors and inspected region coverage; paths are not
  automatically rooms. Unsupported curved/compound paths are rejected.
- Requested ML and deterministic results are delivered between agent turns when
  ready. Public tool coordinates now consistently use the crop tools' normalized
  page frame, while internal source geometry remains in native pixels.
- Nemotron can inspect a lossless quarter-turned detail view, with returned
  coordinates transformed back into the original page frame.
- Counting instructions distinguish physical door symbols from entrance roles;
  count-unit normalization preserves explicit structural nouns without changing
  the predicted quantity.

These changes were evaluated together; this experiment does not isolate the
causal contribution of each feature. Algorithm-result delivery is now observed
in real traces, but that alone does not prove it improved an answer.

## Regressions and hard drawings

Every matched regression was reviewed against saved answers and traces. Remaining
errors include adjacent partial dwellings entering the count, assuming an
unlabelled second apartment, counting continuous window openings rather than
panes, missed openings, and misclassifying bathroom fixtures. Some failures are
provider interruptions, but several are ordinary semantic errors with successful
tool calls. A well-formed citation is not proof of a correct answer.

Two previously used large PNGs were also uploaded unchanged for an **unscored**
stress comparison. Both agents returned the same stair-core count on the dormitory sheet
(45.7 s control, 44.6 s candidate); there is no independent confirmed count.
On the school hall, V6 returned a materially larger area than the control and a
prior independent manual polygon estimate, which is not official ground truth.
The candidate's scale-bar endpoints were shorter despite a similar room polygon.
Its internally consistent scale checks therefore accepted a likely wrong scale.
This is an unresolved measurement limitation, retained prominently in the private
review. Original bytes and decoded source pixels match both runs.

The next measurement work should verify scale tick endpoints independently and
report uncertainty from their localization. Two nested intervals on one bar can
agree while both are misplaced; arithmetic consistency alone is insufficient.
Any implementation must get a new frozen evaluation rather than reuse these
holdout answers for tuning and continue calling the set untouched.

## Experiment accounting and discarded candidate

The session retained **399 core question attempts across 13 trials**: three
fixed-50 passes, five development-30 passes, three synthetic-13 passes, and two
holdout-30 passes. The synthetic checks are reported separately: control 13/13,
V4 12/13, V5 13/13. V6 count-unit handling also passed saved-response replay;
there was no additional live V6 synthetic-13 pass.

Four native stress attempts and three targeted V7 probes bring new evaluation
attempts to **406**. V7 gave native locator calls a separate bounded allowance.
It scored **0/3** on its predeclared development probe and was rejected; the
conditional additional 90 calls were not run. Three probe responses were retained
without rerunning inference. Two runner records needed recovery from their
already completed server runs after missing cohort metadata caused serialization
errors; their recovered server timing is not a comparable HTTP latency metric.

A further 20 source-disjoint reference questions were prepared but **not run or
scored**. Their answers were not used for tuning. Keep them separate from all
reported denominators. Subsequent release plumbing checks, if performed, are
operational checks and not additions to benchmark accuracy.

After the native scale-bar discrepancy, a further paired check on those 20 files
was proposed. Automatic approval review blocked its 40 API requests; none were
launched. Explicit approval is pending. The candidate passes its original
qualification gates, but the additional confirmation gate is unmeasured. Keep
the existing live demo on its preserved backend until that decision is resolved.

## Independence, safety and cost limits

The new development and holdout files are disjoint from the old 100-reference-file
pool by name and content hash, and from one another. Building-family independence
has not been established. The detector itself was frozen. Publisher source IDs
overlap its external CubiCasa manifest for 24 regression, 10 development and 11
holdout drawings; 17, 10 and 9 respectively belong to its training pool. Exact
decoded raster hashes did not match those training rasters. Shared source geometry
still prevents an unseen-detector-data claim, and training-pool membership does
not prove a sample was selected in a finite training run.

Gains were concentrated in PDF sources: regression Swiss Dwellings improved
15/25 → 21/25 while CubiCasa fell 19/24 → 17/24. On holdout, Swiss improved
8/11 → 9/11 and IFC-derived drawings 4/8 → 5/8; CubiCasa stayed 6/11. These small
subgroups further limit any claim of broadly improved visual recognition.

The agent never receives reference answers. It does receive the publisher's unit
hint identically in both matched conditions; ordinary UI questions may omit that
hint. Sealed final challenge labels were not opened. Integer counts and the
existing 2% area tolerance were not relaxed or relabelled after mistakes.

The selected candidate passed 495 local tests and the source CI checks. Structural
audits of its 80 qualification responses found no source identity, citation-bound,
or checked delegation-contract violations. This does not certify semantic
correctness or exhaustive visual coverage. Traces retained 13 server errors and
one timeout on regression, and three server errors and two timeouts on holdout;
all requests reached a terminal answer or abstention.

Reported Gemini tokens increased on holdout (2.02M → 2.73M); regression was
3.30M → 3.54M. Token totals can include cached input and are not invoice amounts.
Nemotron calls were 338 → 315 on regression and 208 → 204 on holdout. No new GPU
was purchased and model weights were unchanged.

## Reproduction and rollback

See [AGENT_LAB.md](AGENT_LAB.md) for runners, freezing, scoring and private report
generation. The retained offline review contains English/original questions,
reference and model answers, source images, citations, traces, gains and losses.
Raw records, source packs, rejected trials and a checksum-verified backup stay
private; this repository contains only reusable source, tests and documentation.

The selected image is
`sha256:e1826b9c803e1ff116d641108897c7aa1fd06747a522f26a6a98717b7bb098b4`.
The previous deployed image is `planproof-product:fad04ec`; source rollback tag
`agent-before-overnight-20261115` points to `d8ecb51`. Backend promotion preserves
the frontend and saved run files. Restarting the API clears its in-memory page
cache, so the browser may need to re-upload a cached drawing.
