import { useEffect, useState } from "react";
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Layers,
  List,
  X,
} from "lucide-react";
import {
  durationSuffix,
  sameSource,
  stageNames,
  terminal,
  type AgentRun,
  type EvidenceBundle,
  type EvidenceEvent,
  type EvidenceShape,
  type Region,
  type SourceFrame,
} from "../lib/runs";
import type { Drawing } from "../lib/api";

const ruleDescriptions: Record<string, string> = {
  "bounded-pixel-check":
    "Geometry was checked against pixels inside an ML proposal. This refinement depends on the original prediction.",
  "quarter-arc-leaf":
    "A quarter-circle arc with a supporting leaf line suggests a door swing.",
  "parallel-glazing-band-endcaps":
    "A band of parallel strokes bounded by endcaps suggests glazing between wall ends.",
  "filled-core-and-boundary-pairs":
    "Filled cores and paired boundary lines propose wall regions from the drawing pixels.",
  attached_leaf_and_swing_arc:
    "A line attached to the wall end and a supported swing arc suggest a door.",
  bounded_parallel_glazing_at_wall_ends:
    "Parallel glazing strokes bounded by wall ends suggest a window.",
  clear_bounded_passage_no_supported_symbol:
    "A clear gap between wall ends has no supported door or window symbol. A damaged symbol remains possible.",
  conflicting_door_and_glazing_symbols:
    "Both door and glazing cues fired. This opening remains unresolved.",
  symbol_evidence_inconclusive:
    "The visible cues do not support a reliable opening classification.",
  "repeated-treads-rails":
    "Repeated parallel treads with supporting rails suggest a stair symbol.",
  "parallel-lines-wall-context":
    "Parallel strokes near a wall suggest a glazing symbol.",
};
const cueNames: Record<string, string> = {
  arc_support: "Swing arc support",
  leaf_support: "Door leaf support",
  glazing_stroke_groups: "Glazing stroke groups",
  passage_ink_fraction: "Ink fraction in passage",
  boundary_support: "Boundary support",
  parallel_line_count: "Parallel lines",
  spacing_cv: "Spacing variation",
  rail_count: "Supporting rails",
  wall_context: "Wall context",
};

export const layerColors = {
  ml: "#3671ae",
  deterministic: "#ac6a0d",
  hybrid: "#8054a2",
};
export function CropPreview({
  region,
  drawing,
}: {
  region: Region;
  drawing: Drawing;
}) {
  const [x0, y0, x1, y1] = region.canonical_bbox;
  return (
    <svg
      viewBox={`${x0} ${y0} ${x1 - x0} ${y1 - y0}`}
      className="crop-preview"
      aria-label="Source crop"
      role="img"
    >
      <image
        href={drawing.image_url}
        width={drawing.width}
        height={drawing.height}
      />
    </svg>
  );
}
interface Props {
  run?: AgentRun;
  drawing: Drawing;
  frame?: SourceFrame;
  connection: string;
  onBack: () => void;
  onLoadLayers: () => void;
  loading: boolean;
  layerError: string;
  onFocus: (region: Region) => void;
  onShapes: (shapes: EvidenceShape[]) => void;
  onCompare: (split: number | null) => void;
  follow: boolean;
  onFollow: (value: boolean) => void;
}
export function EvidenceInspector({
  run,
  drawing,
  frame,
  connection,
  onBack,
  onLoadLayers,
  loading,
  layerError,
  onFocus,
  onShapes,
  onCompare,
  follow,
  onFollow,
}: Props) {
  const [tab, setTab] = useState<"steps" | "layers">("steps");
  const [selected, setSelected] = useState<number | null>(null);
  const [layers, setLayers] = useState<string[]>([]);
  const [filter, setFilter] = useState("all");
  const [objectId, setObjectId] = useState<string | null>(null);
  const [compare, setCompare] = useState(false);
  const [diagnostics, setDiagnostics] = useState(false);
  const [split, setSplit] = useState(50);
  const [limit, setLimit] = useState(50);
  const bound = sameSource(run?.source, frame);
  const bundle: EvidenceBundle | undefined =
    bound && sameSource(run?.evidence?.source, frame)
      ? run?.evidence
      : undefined;
  const shapes = (bundle?.shapes || []).filter(
    (s) => layers.includes(s.layer) && (filter === "all" || s.class === filter),
  );
  useEffect(() => {
    onCompare(tab === "layers" && diagnostics && bound && compare ? split : null);
  }, [tab, bound, compare, split]);
  const chosen = shapes.find((s) => s.id === objectId);
  const delivered = new Set(
    run?.events.flatMap((e) => e.delivered_ids || []) || [],
  );
  const events = run?.events || [];
  useEffect(() => {
    onShapes(tab === "layers" && diagnostics ? shapes : []);
  }, [tab, bundle, layers, filter, diagnostics]);
  useEffect(() => {
    if (!follow || !bound) return;
    const event = [...events]
      .reverse()
      .find(
        (e) =>
          e.images?.length &&
          ["completed", "cached", "running"].includes(e.state),
      );
    if (event) {
      setSelected(event.sequence);
      onFocus(event.images[0]);
    }
  }, [run?.last_sequence, follow, bound]);
  useEffect(() => {
    if (
      tab === "layers" && diagnostics &&
      (!run?.evidence || run.evidence.status === "not_run") &&
      events.some(
        (e) => e.stage === "get_algorithms" && e.state === "completed",
      )
    )
      onLoadLayers();
  }, [tab, run?.last_sequence]);
  function select(event: EvidenceEvent) {
    setSelected(event.sequence);
    const region = event.images?.[0] || event.citations?.[0];
    if (bound && region) onFocus(region);
  }
  const stepIndex = events.findIndex((e) => e.sequence === selected);
  return (
    <section
      className="assistant-panel inspector"
      aria-label="Evidence inspector"
    >
      <header className="assistant-header">
        <button
          className="icon-button"
          onClick={onBack}
          aria-label="Back to conversation"
        >
          <ArrowLeft size={19} />
        </button>
        <h2>Task details</h2>
      </header>
      <div className="inspector-tabs">
        <button aria-pressed={tab === "steps"} onClick={() => setTab("steps")}>
          <List size={15} />
          Steps
        </button>
        <button
          aria-pressed={tab === "layers"}
          onClick={() => {
            setTab("layers");

          }}
        >
          <Layers size={15} />
          Diagnostics
        </button>
      </div>
      <div className="inspector-body">
        {!run ? (
          <p>
            Execution details were not saved for this answer. Its citations
            remain available in the conversation.
          </p>
        ) : (
          <>
            <p className="inspector-note">
              {terminal(run)
                ? "Recorded activity · replay does not rerun the agent."
                : connection || "Live activity · actual execution events."}
            </p>
            {run.message && <p role="status">{run.message}</p>}
            {run.telemetry_warning && (
              <p role="status">{run.telemetry_warning}</p>
            )}
            {!bound && (
              <p className="inspector-warning">
                Source identity could not be matched. Drawing overlays and crop
                previews are withheld.
              </p>
            )}
            {tab === "steps" ? (
              <>
                <div className="replay-controls">
                  <label>
                    <input
                      type="checkbox"
                      checked={follow}
                      onChange={(e) => onFollow(e.target.checked)}
                    />
                    Follow live regions
                  </label>
                  <div>
                    <button
                      className="icon-button"
                      aria-label="Previous step"
                      disabled={stepIndex <= 0}
                      onClick={() => select(events[stepIndex - 1])}
                    >
                      <ChevronLeft size={16} />
                    </button>
                    <button
                      className="icon-button"
                      aria-label="Next step"
                      disabled={
                        !events.length || stepIndex >= events.length - 1
                      }
                      onClick={() => select(events[stepIndex + 1])}
                    >
                      <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
                {!events.length && (
                  <p>
                    {terminal(run)
                      ? "Execution details were not recorded for this answer."
                      : "Waiting for the first recorded event."}
                  </p>
                )}
                <ol className="event-list">
                  {events.map((event) => (
                    <li
                      key={event.sequence}
                      className={
                        selected === event.sequence ? "event-selected" : ""
                      }
                    >
                      <button
                        className="event-title"
                        onClick={() => select(event)}
                      >
                        <span>
                          {stageNames[event.stage] || "Agent activity"}
                        </span>
                        <small>
                          {event.state}
                          {durationSuffix(event.seconds)}
                        </small>
                      </button>
                      {event.phase && event.phase !== "normal" && (
                        <p>
                          Recovery phase: {event.phase.replaceAll("_", " ")}
                        </p>
                      )}
                      {event.summary && <p>{event.summary}</p>}
                      {event.question && <p>{event.question}</p>}
                      {!!event.classes?.length && (
                        <p>Requested: {event.classes.join(", ")}</p>
                      )}
                      {event.advisory?.findings?.map((f, i) => (
                        <p key={i}>{f}</p>
                      ))}
                      {event.advisory && (
                        <small>
                          Advisory for these crops only; not independent
                          verification.
                        </small>
                      )}
                      {event.checks?.map((c, i) => (
                        <p key={i}>{c}</p>
                      ))}
                      {event.unresolved?.map((c, i) => (
                        <p className="inspector-warning" key={i}>
                          {c}
                        </p>
                      ))}
                      {event.stage === "reflect" && (
                        <small>
                          Agent self-check; not independent validation.
                        </small>
                      )}
                      {bound && (
                        <div className="event-crops">
                          {(event.images?.length
                            ? event.images
                            : event.citations || []
                          ).map((region, i) => (
                            <button
                              key={i}
                              aria-label={`Inspect step ${event.sequence} region ${i + 1}`}
                              onClick={() => {
                                setSelected(event.sequence);
                                onFocus(region);
                              }}
                            >
                              <CropPreview region={region} drawing={drawing} />
                              <span>
                                {region.image_id || `Evidence ${i + 1}`}
                              </span>
                            </button>
                          ))}
                        </div>
                      )}
                    </li>
                  ))}
                </ol>
                {run.result?.answer && (
                  <div className="scope-note">
                    <strong>Remaining scope</strong>
                    <p>{run.result.answer.coverage}</p>
                    {run.result.answer.uncertainties?.map((u, i) => (
                      <p key={i}>{u}</p>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <>
                {loading && <p role="status">Loading recorded evidence…</p>}
                {!diagnostics && <div className="diagnostic-gate">
                  <h3>Raw algorithm proposals</h3>
                  <p>These are exploratory candidates, including mistakes. They are not the final findings or proof behind the answer.</p>
                  <p>The work log and cited source details show how the agent used the available evidence.</p>
                  <button className="subtle-button" onClick={() => { setDiagnostics(true); onLoadLayers(); }}>Show raw algorithm output</button>
                </div>}
                {diagnostics && <>
                {layerError && (
                  <p role="alert">
                    {layerError}{" "}
                    <button className="text-button" onClick={onLoadLayers}>
                      Try loading again
                    </button>
                  </p>
                )}
                {bundle?.status === "not_run" && (
                  <p>
                    {terminal(run)
                      ? events.some((e) => e.stage === "get_algorithms")
                        ? "Algorithm evidence was unavailable for this answer."
                        : "Algorithms not used for this answer."
                      : "No algorithm evidence recorded yet."}
                  </p>
                )}
                {bundle?.status === "available" && (
                  <>
                    <p className="inspector-note">
                      Candidates, not verified physical counts. Hybrid
                      boundaries depend on ML.
                    </p>
                    <div className="method-map" aria-label="Evidence methods">
                      <button
                        onClick={() => {
                          setCompare(false);
                          setLayers(["ml"]);
                        }}
                      >
                        <span className="method-dot ml-dot" />
                        ML predictions<small>Learned shapes and symbols</small>
                      </button>
                      <button
                        onClick={() => {
                          setCompare(false);
                          setLayers(["deterministic"]);
                        }}
                      >
                        <span className="method-dot rule-dot" />
                        Pixel + geometry<small>Recorded geometric rules</small>
                      </button>
                      <button
                        className="hybrid-method"
                        onClick={() => {
                          setCompare(false);
                          setLayers(["hybrid"]);
                        }}
                      >
                        <span className="method-dot hybrid-dot" />
                        ML-assisted boundaries
                        <small>Room and opening hypotheses</small>
                      </button>
                    </div>
                    <button
                      className="compare-toggle subtle-button"
                      aria-pressed={compare}
                      onClick={() => {
                        setCompare((v) => !v);
                        if (!compare) setLayers(["ml", "deterministic"]);
                      }}
                    >
                      Compare ML / rules
                    </button>
                    {compare && (
                      <div className="compare-control">
                        <label htmlFor="compare-split">
                          Move the divider to compare the same drawing
                        </label>
                        <input
                          id="compare-split"
                          aria-label="Comparison split"
                          type="range"
                          min="0"
                          max="100"
                          value={split}
                          onChange={(e) => setSplit(Number(e.target.value))}
                        />
                        <div>
                          <span>ML predictions</span>
                          <span>Pixel + geometry</span>
                        </div>
                      </div>
                    )}
                    <div className="layer-options">
                      {(["ml", "deterministic", "hybrid"] as const).map(
                        (layer) => (
                          <label key={layer}>
                            <input
                              type="checkbox"
                              checked={layers.includes(layer)}
                              onChange={(e) => (
                                setCompare(false),
                                setLayers((v) =>
                                  e.target.checked
                                    ? [...v, layer]
                                    : v.filter((l) => l !== layer),
                                )
                              )}
                            />
                            <span style={{ color: layerColors[layer] }}>
                              {layer === "ml"
                                ? "ML detections"
                                : layer === "hybrid"
                                  ? "ML-assisted boundaries"
                                  : "Deterministic evidence"}
                            </span>
                          </label>
                        ),
                      )}
                    </div>
                    <label className="class-filter">
                      Show{" "}
                      <select
                        aria-label="Evidence class"
                        value={filter}
                        onChange={(e) => {
                          setFilter(e.target.value);
                          setLimit(50);
                        }}
                      >
                        <option value="all">All classes</option>
                        {[
                          "room",
                          "wall",
                          "door",
                          "window",
                          "stairs",
                          "walkthrough",
                        ].map((c) => (
                          <option key={c}>{c}</option>
                        ))}
                      </select>
                    </label>
                    <p className="inspector-note">
                      {shapes.length} candidates in enabled layers. Agreement is
                      spatial/class overlap, not correctness.
                    </p>
                    {chosen && (
                      <div className="object-detail">
                        <button
                          className="icon-button"
                          aria-label="Close object details"
                          onClick={() => setObjectId(null)}
                        >
                          <X size={14} />
                        </button>
                        <strong>
                          {chosen.class} · {chosen.layer}
                        </strong>
                        <p>{chosen.role || chosen.status}</p>
                        <div className="candidate-source">
                          <CropPreview
                            drawing={drawing}
                            region={{
                              page_index: frame!.page_index,
                              canonical_bbox: [
                                Math.max(0, chosen.bbox[0] - 20),
                                Math.max(0, chosen.bbox[1] - 20),
                                Math.min(drawing.width, chosen.bbox[2] + 20),
                                Math.min(drawing.height, chosen.bbox[3] + 20),
                              ],
                            }}
                          />
                          <small>
                            Original source · context around this candidate
                          </small>
                        </div>
                        <h3>Why this candidate?</h3>
                        <p>
                          {ruleDescriptions[
                            chosen.reason || chosen.method || ""
                          ] ||
                            (chosen.layer === "ml"
                              ? "The learned detector proposed this region. It is a candidate, not a confirmed object."
                              : (
                                  chosen.reason ||
                                  chosen.method ||
                                  chosen.status ||
                                  "No additional rule explanation was recorded."
                                ).replaceAll("_", " "))}
                        </p>
                        {!!chosen.cues?.length && (
                          <>
                            <dl className="rule-cues">
                              {chosen.cues.map((c) => (
                                <div key={c.name}>
                                  <dt>
                                    {cueNames[c.name] ||
                                      c.name.replaceAll("_", " ")}
                                  </dt>
                                  <dd>
                                    {typeof c.value === "boolean"
                                      ? c.value
                                        ? "Present"
                                        : "Absent"
                                      : Number.isInteger(c.value)
                                        ? c.value
                                        : c.value.toFixed(3)}
                                  </dd>
                                </div>
                              ))}
                            </dl>
                            <small>
                              Recorded measurements / rule support, not
                              correctness probabilities.
                            </small>
                          </>
                        )}

                        <p>
                          {delivered.has(chosen.id.replace(/^boundary:/, ""))
                            ? "Supplied to the agent"
                            : "Available output; not recorded as supplied to the agent"}
                        </p>
                        {bundle.comparisons
                          .filter((c) => c.source_ids.includes(chosen.id))
                          .map((c, i) => (
                            <p key={i}>
                              {c.relation.replaceAll("_", " ")}
                              {c.iou !== undefined
                                ? ` · IoU ${c.iou.toFixed(2)}`
                                : ""}{" "}
                              ·{" "}
                              {c.semantics ||
                                "Candidate comparison, not verification"}
                            </p>
                          ))}
                      </div>
                    )}
                    <div className="object-list">
                      {shapes.slice(0, limit).map((shape) => (
                        <button
                          key={shape.id}
                          aria-pressed={objectId === shape.id}
                          onClick={() => {
                            setObjectId(shape.id);
                            onFocus({
                              canonical_bbox: shape.bbox,
                              page_index: frame!.page_index,
                            });
                          }}
                        >
                          <span style={{ color: layerColors[shape.layer] }}>
                            ●
                          </span>
                          {shape.class}
                          {shape.role ? ` · ${shape.role}` : ""}
                          <small>{shape.id.slice(-8)}</small>
                        </button>
                      ))}
                    </div>
                    {shapes.length > limit && (
                      <button
                        className="text-button"
                        onClick={() => setLimit((v) => v + 50)}
                      >
                        Show more candidates
                      </button>
                    )}
                    {!!bundle.warnings?.length && (
                      <details className="answer-notes">
                        <summary>Evidence limitations</summary>
                        {bundle.warnings.map((w, i) => (
                          <p key={i}>{w}</p>
                        ))}
                      </details>
                    )}
                  </>
                )}
                </>}
              </>
            )}
          </>
        )}
      </div>
    </section>
  );
}
