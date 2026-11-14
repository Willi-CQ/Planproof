# Cited plan-area measurements

The question agent can answer room-area questions in square feet or metric units
through the existing `POST /api/v1/documents/{id}/ask` endpoint. The request and
existing answer/citation fields are unchanged. No frontend changes are required.

## Evidence and calculation

The coordinator first identifies the requested space, inspects clean source
crops, reads dimensions/area labels and checks drawing notes. It chooses:

| Source evidence | Tool | Result |
| --- | --- | --- |
| Area printed for the actual target | `calculate_area`, `printed` | Read/conversion of the cited area label |
| Length and width explicitly bounding the target | `calculate_area`, `rectangles` | Sum/subtraction of dimensioned rectangular components |
| Visible boundary and two distinct, consistent dimensions in the same view | `measure_polygon` | Approximate calibrated polygon area, minus holes |
| Missing/unclear target, units, boundary or calibration | Abstention | Explain the evidence needed |

Feet, inches, proper fractions, metric lengths and squared units are supported.
Bare numeric dimensions require an explicit unit context and a separate citation
to the drawing's unit note. Scale-ratio titles alone cannot calibrate raster
pixels. Comma-separated numbers require an explicit `number_format` convention
(`decimal_point` or `decimal_comma`) supported by the drawing context.
Furniture sizes and assumed DPI cannot supply scale. The host never
executes model-supplied expressions or code.

Polygon vertices, holes, dimension endpoints and the view extent use normalized
0..1000 **original-page** coordinates. Text/region citations use the existing
registered **sent-image pixel** coordinates. Geometry is converted to native
pixels before arithmetic; different page width/height scales are respected.
The host rejects self-intersecting rings, holes outside the polygon, overlapping
holes, repeated anchors, short anchors, out-of-view geometry and scales that
disagree by more than 3%. Calibration uses a length-weighted least-squares fit.
This threshold is a consistency check, not a certified accuracy/error bound.

The selected boundary must state whether it uses interior faces, exterior faces,
or another basis. Common-space/wing totals must not be attributed to an individual
gym. Adjacent circulation and open-to-below voids must be excluded when outside
the requested scope. Dimensioned components must not overlap. Those semantic
associations are proposed by the model and remain subject to visual verification.

## API and frontend handoff

Each successful measurement tool returns a host-owned `measurement_id` such as
`M1`, numeric `value`, `unit`, `formula`, `boundary_basis`, `approximate`, inputs,
and both image and canonical citations. `submit_answer` references that ID.
The host checks the final number/unit against the registered calculation and
attaches all input citations. Non-direct numeric area answers with recognized
area units require a successful measurement. The direct comparison condition
remains tool-free and is instructed to answer printed labels only.

The response adds a nullable `measurement` object. For raster estimates it also
contains `canonical_polygon`, `canonical_holes`, native-pixel anchors and
calibration details, suitable for a future boundary overlay. Existing frontends
continue to render the numeric answer, explanation, uncertainties and source
boxes. The host enforces an approximation warning for raster measurements even
if the model omits it. Calculations survive the compact token-recovery path.
Recovery also allows up to two local calculation attempts using the already
inspected images. A provider timeout must not disable arithmetic merely because
the model has not called the measurement tool yet. Missing width labels alone
do not prevent an approximate takeoff using other same-view dimension anchors.
New image inspections stay disabled; invalid/unreadable inputs still require
abstention. Recovery abstentions distinguish interrupted processing from an
actual lack of drawing evidence.

With the observation-enabled backend, both the legacy `/ask` endpoint and the
resumable `/runs` endpoint return measurements. Completed measurement events
include the result, calculation and cited source locations in their summary.
Selected PDF page numbers are propagated to measurement citations as well as
the final answer. Polling/replaying a completed run performs no new calculation
or provider call.

Raster estimates are reported in whole requested area units; dimension/printed
calculations retain up to two decimals. Precision is not a correctness guarantee.
Conceptual/approximate source dimensions must also be reported as approximate.

## Installation and verification

`backend/requirements.txt` now includes Shapely 2.0.7, matching the existing
geometry worker. Install the host requirements and restart the backend after
updating. Existing deployments that only update source must install this added
dependency. The geometry/ML workers and their frozen weights are unchanged.

```powershell
python -m pip install -r backend/requirements.txt -e 'feedback[test]'
python -m pytest backend feedback/tests tools/test_agent_cohort.py tools/test_mvp.py tools/test_extend_case_review.py -q
```

Synthetic tests cover unit arithmetic, conversions, cutouts, irregular polygons,
source transforms, invalid geometry/calibration, tool permissions, final-answer
binding and recovery. They establish host behavior, not model reading accuracy.
No filename-specific coordinates or answers are embedded in production code.
Drawings, provider credentials and development-run artifacts remain local.
