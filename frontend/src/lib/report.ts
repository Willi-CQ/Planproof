import { ApiError } from "./api";

/** Export the saved run. This never submits a question or invokes an agent. */
export async function downloadTaskReport(runId: string, documentId: string, question: string, filename: string) {
  const response = await fetch(`/api/v1/runs/${encodeURIComponent(runId)}/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: documentId, question, filename }),
    signal: AbortSignal.timeout(60000),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(typeof body?.detail === "string" ? body.detail : "The report could not be downloaded. Please try again.", response.status);
  }
  if (!response.headers.get("content-type")?.includes("application/pdf"))
    throw new Error("The server did not return a PDF. Please try again.");
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = `planproof-task-${runId.slice(0, 8)}.pdf`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
