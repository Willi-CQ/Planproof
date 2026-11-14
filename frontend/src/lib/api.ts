// Only the existing Python agent; no credentials, alternate agent or automatic retries.
export interface Drawing {
  document_id: string;
  page: number;
  image_url: string;
  width: number;
  height: number;
}
export interface Citation {
  canonical_bbox: number[];
  page_index: number;
  image_id?: string;
}
export interface AgentResult {
  status: string;
  answer?: {
    status: string;
    answer: number | string | null;
    unit?: string | null;
    explanation: string;
    coverage: string;
    uncertainties: string[];
  } | null;
  source?: { width: number; height: number; page_index: number };
  canonical_citations?: Citation[];
  algorithm_used?: boolean;
  nemotron_calls?: number;
  reflection_calls?: number;
  seconds?: number;
  stop_reason?: string;
  tool_trace?: { tool?: string; name?: string; result?: { status?: string } }[];
  reliability?: {
    recovery_used?: boolean;
    recovery_succeeded?: boolean;
    token_repair_used?: boolean;
    token_repair_succeeded?: boolean;
    retry_after_seconds?: number;
  };
}
export interface Permissions {
  allow_gemini: boolean;
  allow_nemotron: boolean;
}
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
async function readResponse<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof data?.detail === "string" ? data.detail : null;
    const message =
      response.status === 404
        ? "This uploaded page has expired. Re-upload it before asking again."
        : response.status === 413
          ? "This file is too large. The upload limit is 10 MiB."
          : detail ||
            `Request failed (${response.status}). Please try again explicitly.`;
    throw new ApiError(message, response.status);
  }
  if (!data || typeof data !== "object")
    throw new Error("The server returned an invalid response.");
  return data as T;
}
export function validateFile(file: Pick<File, "type" | "size">): string | null {
  if (!["application/pdf", "image/png", "image/jpeg"].includes(file.type))
    return "Choose a PDF, PNG or JPEG file.";
  if (file.size > 10 * 1024 * 1024)
    return "The upload limit is 10 MiB. Your original drawing will not be resized.";
  return null;
}
export async function uploadDrawing(
  file: File,
  page: number,
): Promise<Drawing> {
  const error = validateFile(file);
  if (error) throw new Error(error);
  const form = new FormData();
  form.append("file", file);
  form.append("page", String(page));
  return readResponse<Drawing>(
    await fetch("/api/documents", { method: "POST", body: form }),
  );
}
export async function askDrawing(
  documentId: string,
  question: string,
  permissions: Permissions,
): Promise<AgentResult> {
  return readResponse<AgentResult>(
    await fetch(`/api/v1/documents/${encodeURIComponent(documentId)}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: "auto", ...permissions }),
    }),
  );
}
export async function agentStatus(): Promise<{
  configured: boolean;
  model: string | null;
  connection_verified: boolean;
}> {
  return readResponse(await fetch("/api/v1/agent"));
}
export function isAnswered(result: AgentResult): boolean {
  return (
    result.status === "ok" &&
    result.answer?.status === "answered" &&
    result.answer.answer !== null
  );
}
export function resultText(result: AgentResult): string {
  if (isAnswered(result))
    return `${result.answer!.answer}${result.answer!.unit ? " " + result.answer!.unit : ""}`;
  if (result.status === "not_configured")
    return "Agent configuration is missing on the server.";
  if (result.status === "not_authorized")
    return "Provider permission is required.";
  if (result.answer?.status === "abstain")
    return "Not enough evidence to answer";
  return "The agent could not complete this question";
}
export function visibleCitations(
  result: AgentResult,
  drawing: Drawing,
): Citation[] {
  if (
    !result.source ||
    result.source.width !== drawing.width ||
    result.source.height !== drawing.height
  )
    return [];
  return (result.canonical_citations || []).filter((c) => {
    const b = c.canonical_bbox;
    return (
      c.page_index === drawing.page - 1 &&
      Array.isArray(b) &&
      b.length === 4 &&
      b.every(Number.isFinite) &&
      b[0] >= 0 &&
      b[1] >= 0 &&
      b[2] > b[0] &&
      b[3] > b[1] &&
      b[2] <= drawing.width &&
      b[3] <= drawing.height
    );
  });
}
