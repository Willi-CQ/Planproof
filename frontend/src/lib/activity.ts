import { sameSource, terminal, type AgentRun, type EvidenceEvent, type Region, type SourceFrame } from "./runs";

export function workSteps(run: AgentRun): EvidenceEvent[] {
  const steps: EvidenceEvent[] = [];
  for (const event of run.events) {
    if (event.stage === "run" || event.stage === "provider_attempt") continue;
    const pending = steps.reduce((found, e, i) => e.stage === event.stage && e.state === "running" ? i : found, -1);
    if (pending >= 0 && event.state !== "running") {
      const previous = steps[pending];
      steps[pending] = { ...previous, ...event, images: event.images?.length ? event.images : previous.images };
    } else steps.push(event);
  }
  return steps;
}
const titles: Record<string, [string, string]> = {
  gemini: ["Reviewing the drawing and evidence", "Drawing and evidence reviewed"],
  get_algorithms: ["Locating rooms and structural clues", "Room and structural clues collected"],
  crop_region: ["Opening a closer view", "Closer view prepared"],
  crop_room: ["Opening a room detail", "Room detail prepared"],
  inspect_nemotron: ["Checking this detail with Nemotron", "Focused visual check returned"],
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
  return title[event.state === "running" ? 0 : 1];
}
export function activityDetail(event?: EvidenceEvent) {
  if (!event) return "Waiting for the first execution event.";
  if (event.state === "pending" || event.state === "busy")
    return "Continuing with the drawing itself. No algorithm evidence has been supplied yet.";
  if (event.question) return event.question;
  if (event.classes?.length) return `Requested clues: ${event.classes.join(", ")}. Candidates will guide the next inspection.`;
  if (event.checks?.length) return event.checks.join(" ");
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
    const b = r.canonical_bbox;
    return r.page_index === source.page_index && b.length === 4 && b.every(Number.isFinite)
      && 0 <= b[0] && b[0] < b[2] && b[2] <= source.width
      && 0 <= b[1] && b[1] < b[3] && b[3] <= source.height;
  });
}
export function liveActivity(run: AgentRun | undefined, frame?: SourceFrame) {
  if (!run || terminal(run)) return null;
  const steps = workSteps(run);
  const event = steps.at(-1);
  const spatial = [...steps].reverse().find(e => boundRegions(e, frame).length);
  return {
    event,
    regions: sameSource(run.source, frame) ? boundRegions(spatial, frame) : [],
    currentRegion: spatial?.sequence === event?.sequence,
    sequence: spatial?.sequence,
  };
}
