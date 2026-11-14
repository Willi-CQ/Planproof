import { sameSource, terminal, type AgentRun, type EvidenceEvent, type Region, type SourceFrame } from "./runs";

export interface WorkStep extends EvidenceEvent {
  startSequence: number;
  startedAt?: number;
  returnedAt?: number;
  detailNumber?: number;
}
const recordedTime = (value: unknown) => typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
export function eventTime(value: unknown): string | null {
  const seconds = recordedTime(value);
  return seconds === undefined ? null : `+${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}
export function workSteps(run: AgentRun): WorkStep[] {
  const steps: WorkStep[] = [];
  let detailNumber = 0;
  for (const event of run.events) {
    if (event.stage === "run" || event.stage === "provider_attempt") continue;
    const pending = steps.reduce((found, e, i) => e.stage === event.stage && e.inspection_id === event.inspection_id && e.state === "running" ? i : found, -1);
    if (pending >= 0 && event.state !== "running") {
      const previous = steps[pending];
      steps[pending] = { ...previous, ...event, images: event.images?.length ? event.images : previous.images,
        returnedAt: recordedTime(event.elapsed) };
    } else steps.push({ ...event, startSequence: event.sequence,
      startedAt: event.state === "running" ? recordedTime(event.elapsed) : undefined,
      returnedAt: event.state !== "running" ? recordedTime(event.elapsed) : undefined,
      detailNumber: event.stage === "inspect_nemotron" ? ++detailNumber : undefined });
  }
  return steps;
}
export function toolFamily(stage: string) {
  if (stage === "inspect_nemotron" || stage === "review_answer") return "nemotron";
  if (stage === "get_algorithms") return "algorithms";
  if (stage === "crop_region" || stage === "crop_room") return "source";
  if (stage === "calculate_area" || stage === "measure_polygon") return "measurement";
  return "coordinator";
}
export const toolLabels = {nemotron: "Nemotron", algorithms: "ML + deterministic", source: "Source crop", measurement: "Measurement", coordinator: "Agent"};
export function needsAttention(step: EvidenceEvent, done: boolean) {
  return ["unavailable", "partial"].includes(step.state) || (done && step.state === "running")
    || !!step.unresolved?.length || ["uncertain", "unsupported", "conflicting"].includes(step.advisory?.verdict || "")
    || ["uncertain", "unavailable", "partial"].includes(step.advisory?.status || "");
}
const titles: Record<string, [string, string]> = {
  gemini: ["Reviewing the drawing and evidence", "Drawing and evidence reviewed"],
  get_algorithms: ["Requesting ML and rule-based evidence", "ML and rule-based evidence returned"],
  crop_region: ["Opening a closer view", "Closer view prepared"],
  crop_room: ["Opening a room detail", "Room detail prepared"],
  inspect_nemotron: ["Checking this detail with Nemotron", "Focused visual check returned"],
  inspect_regions: ["Delegating focused detail checks", "Detail inspections returned"],
  review_answer: ["Cross-checking the proposed answer", "Answer review returned"],
  reflect: ["Checking coverage and uncertainty", "Coverage and uncertainty checked"],
  calculate_area: ["Calculating from cited dimensions", "Source-based calculation returned"],
  measure_polygon: ["Measuring the selected boundary", "Boundary measurement returned"],
  answer: ["Preparing the briefing", "Briefing ready"],
};
export function activityTitle(event?: EvidenceEvent) {
  if (!event) return "Question queued";
  if (event.stage === "get_algorithms" && (event.state === "pending" || event.state === "busy"))
    return event.state === "pending" ? "Structural analysis continues in the background" : "Structural analysis is busy";
  const title = titles[event.stage] || ["Working with the drawing", "Step recorded"];
  if (event.state === "unavailable") return `${title[0]} — unavailable`;
  if (event.state === "partial") return "Some detail checks need another look";
  return title[event.state === "running" ? 0 : 1];
}
export function activityDetail(event?: EvidenceEvent) {
  if (!event) return "Waiting for the first execution event.";
  if (event.state === "pending" || event.state === "busy")
    return "No algorithm evidence was supplied by this request. Source inspection can continue while analysis runs in the background.";
  if (event.question) return event.question;
  if (event.classes?.length) return `Requested clues: ${event.classes.join(", ")}. Candidates will guide the next inspection.`;
  if (event.checks?.length) return event.checks.join(" ");
  if (event.stage === "inspect_regions") return event.state === "running"
    ? "Dispatching focused source views to Nemotron. Each detail has its own check and can return independently."
    : "The batch returned. Check each detail for findings, failed views and remaining uncertainty.";
  if (event.stage === "gemini") return event.state === "running"
    ? "Deciding whether the available evidence answers the question or needs a closer look."
    : "The next action follows the question and the evidence available.";
  if (event.stage === "crop_region" || event.stage === "crop_room") return "A clean source crop keeps the inspection focused on this part of the drawing.";
  return event.summary || "Recorded tool activity for this question.";
}
export function boundRegions(event: EvidenceEvent | undefined, source?: SourceFrame): Region[] {
  if (!event || !source) return [];
  const regions = event.images?.length ? event.images : event.citations || [];
  return regions.filter(r => {
    const b = r?.canonical_bbox;
    return r?.page_index === source.page_index && Array.isArray(b) && b.length === 4 && b.every(Number.isFinite)
      && 0 <= b[0] && b[0] < b[2] && b[2] <= source.width
      && 0 <= b[1] && b[1] < b[3] && b[3] <= source.height;
  });
}
export function inspectionColor(number = 1) {
  return ["#4669c7", "#8957b8", "#157c88", "#b75677"][(Math.max(1, number) - 1) % 4];
}
export function liveActivity(run: AgentRun | undefined, frame?: SourceFrame) {
  if (!run || terminal(run)) return null;
  const steps = workSteps(run);
  const event = steps.at(-1);
  const spatial = [...steps].reverse().find(e => boundRegions(e, frame).length);
  const inspections = steps.filter(e => e.stage === "inspect_nemotron" && e.state === "running");
  const views = inspections.flatMap(e => {
    const regions = boundRegions(e, frame);
    return regions.map((region, i) => ({region,
      label: `Nemotron ${e.detailNumber}${regions.length > 1 ? ` · ${i + 1}` : ""}`, color: inspectionColor(e.detailNumber), active: true}));
  });
  const fallback = boundRegions(spatial, frame).map(region => ({region,
    label: spatial?.state === "running" ? "Inspection view" : "Last inspection", color: "#69758a", active: spatial?.state === "running"}));
  const visible = sameSource(run.source, frame) ? views.length ? views : fallback : [];
  return {
    event,
    views: visible,
    focusKey: visible.map(view => `${view.region.image_id}:${view.region.canonical_bbox.join(",")}`).join("|"),
    regions: visible.map(view => view.region),
    currentRegion: visible.some(view => view.active),
    sequence: spatial?.sequence,
  };
}
