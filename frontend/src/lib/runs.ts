import { ApiError, type AgentResult } from "./api";

export interface SourceFrame {
  raster_sha256: string;
  page_index: number;
  width: number;
  height: number;
}
export interface Region {
  image_id?: string;
  canonical_bbox: number[];
  page_index: number;
}
export interface EvidenceEvent {
  sequence: number;
  stage: string;
  state: string;
  elapsed: number;
  // Older running events explicitly stored null before a duration was known.
  seconds?: number | null;
  summary?: string;
  question?: string;
  inspection_id?: string;
  classes?: string[];
  delivered_ids?: string[];
  images: Region[];
  citations: Region[];
  checks?: string[];
  unresolved?: string[];
  next_action?: string;
  phase?: string;
  category?: string;
  cached?: boolean;
  advisory?: {
    status?: string;
    answer?: string | number | null;
    findings?: string[];
    verdict?: string;
  };
}
export interface EvidenceShape {
  cues?: { name: string; value: number | boolean }[];
  id: string;
  class: string;
  layer: "ml" | "deterministic" | "hybrid";
  bbox: number[];
  polygons?: { outer: number[][]; holes?: number[][][] }[];
  role?: string;
  reason?: string;
  method?: string;
  status?: string;
  source_family?: string;
}
export interface EvidenceBundle {
  version: number;
  status: string;
  source: SourceFrame;
  shapes: EvidenceShape[];
  comparisons: {
    relation: string;
    source_ids: string[];
    bbox: number[];
    iou?: number;
    semantics?: string;
  }[];
  warnings?: string[];
  scope?: string;
}
export interface AgentRun {
  created?: number;
  request_id: string;
  status: "queued" | "running" | "completed" | "failed" | "interrupted";
  source: SourceFrame;
  events: EvidenceEvent[];
  last_sequence: number;
  result?: AgentResult | null;
  message?: string;
  telemetry_warning?: string;
  evidence?: EvidenceBundle;
}
export const terminal = (run: AgentRun) =>
  ["completed", "failed", "interrupted"].includes(run.status);
export function durationSuffix(seconds: unknown): string {
  return typeof seconds === "number" && Number.isFinite(seconds) && seconds >= 0
    ? ` · ${seconds.toFixed(1)}s`
    : "";
}
export const sameSource = (a?: SourceFrame, b?: SourceFrame) =>
  !!a &&
  !!b &&
  !!a.raster_sha256 &&
  a.raster_sha256 === b.raster_sha256 &&
  a.page_index === b.page_index &&
  a.width === b.width &&
  a.height === b.height;
async function read<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok)
    throw new ApiError(
      body?.detail || "Activity is unavailable.",
      response.status,
    );
  return body as T;
}
export async function startRun(
  documentId: string,
  requestId: string,
  question: string,
) {
  return read<AgentRun>(
    await fetch(`/api/v1/documents/${encodeURIComponent(documentId)}/runs`, {
      method: "POST",
      signal: AbortSignal.timeout(15000),
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request_id: requestId,
        request: {
          question,
          mode: "auto",
          allow_gemini: true,
          allow_nemotron: true,
        },
      }),
    }),
  );
}
export async function getRun(id: string, after = 0) {
  return read<AgentRun>(
    await fetch(`/api/v1/runs/${encodeURIComponent(id)}?after=${after}`, {
      signal: AbortSignal.timeout(15000),
    }),
  );
}
export async function getEvidence(id: string) {
  return read<EvidenceBundle>(
    await fetch(`/api/v1/runs/${encodeURIComponent(id)}/evidence`),
  );
}
export async function getFrame(documentId: string) {
  return read<SourceFrame>(
    await fetch(`/api/v1/documents/${encodeURIComponent(documentId)}/frame`),
  );
}
export function mergeRun(
  previous: AgentRun | undefined,
  next: AgentRun,
): AgentRun {
  const events = new Map((previous?.events || []).map((e) => [e.sequence, e]));
  next.events.forEach((e) => events.set(e.sequence, e));
  return {
    ...previous,
    ...next,
    events: [...events.values()].sort((a, b) => a.sequence - b.sequence),
  };
}
export const stageNames: Record<string, string> = {
  run: "Drawing inspection",
  gemini: "Gemini · evidence review",
  get_algorithms: "ML + deterministic evidence",
  crop_room: "Room detail",
  crop_region: "Drawing detail",
  inspect_nemotron: "Nemotron · focused inspection",
  inspect_regions: "Nemotron · parallel detail inspection",
  review_answer: "Nemotron · answer review",
  reflect: "Gemini · self-check",
  provider_attempt: "Connection / recovery",
  answer: "Final answer",
  calculate_area: "Source-based area calculation",
  measure_polygon: "Calibrated boundary measurement",
};
