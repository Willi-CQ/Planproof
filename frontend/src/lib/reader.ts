import type { AgentResult, Citation, Drawing } from "./api";
import { drawingId } from "./library";
import { rasterSource } from "./rasterSource";
import pdfWorker from "pdfjs-dist/build/pdf.worker.min.mjs?url";

// Leave space for multipart headers below Vercel's function body limit.
const requestBytes = 4_000_000;
const previewEdge = 3600;

export interface ReaderPage {
  drawing: Drawing;
  payload: File;
  payloadPage: number;
  payloadHash: string;
  payloadWidth: number;
  payloadHeight: number;
  originalHash: string;
  rasterPreviewSource?: Blob;
  representation: "original" | "page" | "image";
}

export function validateReaderFile(file: File) {
  if (!["application/pdf", "image/png", "image/jpeg"].includes(file.type))
    return "Choose a PDF, PNG or JPEG drawing.";
  if (!file.size) return "This file is empty.";
  return null;
}
function canvasFor(width: number, height: number, edge: number) {
  const scale = Math.min(1, edge / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  return canvas;
}
function encode(canvas: HTMLCanvasElement, type: string, quality?: number) {
  return new Promise<Blob>((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob)
    : reject(new Error("This browser could not render the selected page.")), type, quality));
}
async function compactImage(canvas: HTMLCanvasElement) {
  let current = canvas;
  while (true) {
    const png = await encode(current, "image/png");
    if (png.size <= requestBytes) return { blob: png, canvas: current };
    for (const quality of [.94, .84, .72]) {
      const jpeg = await encode(current, "image/jpeg", quality);
      if (jpeg.size <= requestBytes) return { blob: jpeg, canvas: current };
    }
    const edge = Math.floor(Math.max(current.width, current.height) * .75);
    if (edge < 512) throw new Error("This page exceeds the upload limit. Export a smaller sheet and open it here.");
    const smaller = canvasFor(current.width, current.height, edge);
    smaller.getContext("2d")!.drawImage(current, 0, 0, smaller.width, smaller.height);
    current = smaller;
  }
}

async function pdfPreview(file: File, pageNumber: number) {
  const { getDocument, GlobalWorkerOptions } = await import("pdfjs-dist");
  GlobalWorkerOptions.workerSrc = pdfWorker;
  const assets = import.meta.env.DEV ? "/node_modules/pdfjs-dist/" : "/pdfjs/";
  const task = getDocument({data: new Uint8Array(await file.arrayBuffer()),
    cMapUrl: assets + "cmaps/", cMapPacked: true, standardFontDataUrl: assets + "standard_fonts/",
    wasmUrl: assets + "wasm/", iccUrl: assets + "iccs/"});
  try {
    const doc = await task.promise;
    if (pageNumber < 1 || pageNumber > doc.numPages) throw new Error("This page is outside the PDF.");
    const page = await doc.getPage(pageNumber);
    const {width, height} = page.getViewport({scale: 1});
    const viewport = page.getViewport({scale: Math.min(3, previewEdge / Math.max(width, height))});
    const canvas = canvasFor(viewport.width, viewport.height, previewEdge);
    await page.render({canvas, viewport, background: "white"}).promise;
    return {width, height, pageCount: doc.numPages, canvas};
  } finally {
    await task.destroy();
  }
}
async function imagePreview(file: File) {
  const {width, height, blob} = await rasterSource(file);
  const canvas = canvasFor(width, height, previewEdge);
  const image = await createImageBitmap(blob, {imageOrientation: "none", resizeWidth: canvas.width,
    resizeHeight: canvas.height, resizeQuality: "high"});
  try {
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    return {width, height, pageCount: 1, canvas, rasterPreviewSource: blob};
  } finally {
    image.close();
  }
}
async function singlePage(file: File, page: number) {
  const {PDFDocument} = await import("pdf-lib");
  const source = await PDFDocument.load(await file.arrayBuffer(), {updateMetadata: false});
  const output = await PDFDocument.create();
  const [sheet] = await output.copyPages(source, [page - 1]);
  output.addPage(sheet);
  return new File([new Uint8Array(await output.save()).buffer], "sheet.pdf", {type: "application/pdf"});
}

export async function prepareReaderPage(file: File, page: number, knownHash?: string): Promise<ReaderPage> {
  const error = validateReaderFile(file);
  if (error) throw new Error(error);
  const pdf = file.type === "application/pdf";
  const preview = pdf ? await pdfPreview(file, page) : await imagePreview(file);
  const originalHash = knownHash || await drawingId(file);
  let payload = file, payloadPage = page, payloadWidth = preview.width, payloadHeight = preview.height;
  let representation: ReaderPage["representation"] = "original";
  if (pdf && file.size > requestBytes) {
    // A PDF which renders but cannot be copied can still be submitted as an opt-in page image.
    const selected = await singlePage(file, page).catch(() => null);
    if (selected && selected.size <= requestBytes) {
      payload = selected;
      payloadPage = 1;
      representation = "page";
    }
  }
  let previewBlob: Blob;
  if (payload.size > requestBytes) {
    const {blob, canvas} = await compactImage(preview.canvas);
    payload = new File([blob], blob.type === "image/png" ? "page.png" : "page.jpg", {type: blob.type});
    payloadPage = 1;
    payloadWidth = canvas.width;
    payloadHeight = canvas.height;
    previewBlob = blob;
    representation = "image";
  } else {
    previewBlob = await encode(preview.canvas, "image/png");
  }
  const payloadHash = payload === file ? originalHash : await drawingId(payload);
  const drawing = {document_id: "reader-" + originalHash + "-p" + page, page,
    page_count: preview.pageCount, image_url: URL.createObjectURL(previewBlob),
    original_url: URL.createObjectURL(file),
    width: preview.width, height: preview.height,
    source: {page_index: page - 1, width: preview.width, height: preview.height, raster_sha256: originalHash}};
  return {drawing, payload, payloadPage, payloadHash, payloadWidth, payloadHeight, originalHash, representation,
    rasterPreviewSource: "rasterPreviewSource" in preview ? preview.rasterPreviewSource as Blob : undefined};
}

interface NativeAnswer {
  status: "answered" | "partial" | "unresolved";
  value: string | number | string[] | null;
  unit: string | null;
  reason: string;
  reason_code?: string;
  program?: {op?: string};
  raster_coverage?: {tiles_total?: number; tiles_read?: number; tiles_blank?: number; tiles_timed_out?: number};
  calculation: string | null;
  width: number;
  height: number;
  page: number;
  coordinates: string;
  source_sha256: string;
  evidence: {id: string; bbox: number[]; polygon?: number[][]; point?: number[]; text: string; kind: string}[];
}
function mappedAnswer(data: NativeAnswer, prepared: ReaderPage): AgentResult {
  const {drawing, payload, payloadPage, payloadWidth, payloadHeight, payloadHash} = prepared;
  if (!data || !["answered", "partial", "unresolved"].includes(data.status) || !Array.isArray(data.evidence)
      || data.source_sha256 !== payloadHash || data.page !== payloadPage
      || data.coordinates !== (payload.type === "application/pdf" ? "displayed_pdf_points" : "image_pixels")
      || !Number.isFinite(data.width) || !Number.isFinite(data.height) || data.width <= 0 || data.height <= 0
      || Math.abs(data.width - payloadWidth) > .02 || Math.abs(data.height - payloadHeight) > .02) {
    throw new Error("The reader response did not match this sheet. Reopen the drawing and try again.");
  }
  const sx = drawing.width / data.width, sy = drawing.height / data.height;
  const point = (p: number[]) => [p[0] * sx, p[1] * sy];
  const validPoint = (p: unknown): p is number[] => Array.isArray(p) && p.length === 2 &&
    p.every(Number.isFinite) && p[0] >= 0 && p[1] >= 0 && p[0] <= data.width && p[1] <= data.height;
  const citations: Citation[] = [];
  const details: NonNullable<NonNullable<AgentResult["answer"]>["citation_details"]> = [];
  for (const item of data.evidence) {
    if (!Array.isArray(item?.bbox) || item.bbox.length !== 4 || !validPoint(item.bbox.slice(0,2))
      || !validPoint(item.bbox.slice(2)) || item.bbox[0] >= item.bbox[2] || item.bbox[1] >= item.bbox[3]) continue;
    const number = citations.length + 1;
    citations.push({source_index: number, page_index: drawing.page - 1,
      canonical_bbox: [...point(item.bbox.slice(0,2)), ...point(item.bbox.slice(2))],
      polygon: Array.isArray(item.polygon) && item.polygon.every(validPoint) ? item.polygon.map(point) : undefined,
      point: validPoint(item.point) ? point(item.point) : undefined});
    details.push({citation: number, kind: "text", label: [item.kind?.replaceAll("_"," "), item.text].filter(Boolean).join(" · ")});
  }
  const value = Array.isArray(data.value) ? data.value.join(", ") : data.value;
  const reason = data.reason_code || (data.program?.op === "unsupported" ? "unsupported_question" : undefined);
  const tiles = data.raster_coverage;
  const coverage = tiles && Number.isInteger(tiles.tiles_total) && (tiles.tiles_total ?? 0) > 0
    ? `Native image scan: ${tiles.tiles_read ?? 0} of ${tiles.tiles_total} tiles read, ${tiles.tiles_blank ?? 0} blank, ${tiles.tiles_timed_out ?? 0} timed out.` : "";
  return {status: "ok", source: drawing.source, canonical_citations: citations,
    reader_reason: reason,
    reader_input: {representation:prepared.representation, sha256:payloadHash, page:data.page,
      width:data.width, height:data.height, coordinates:data.coordinates},
    completion_reason: data.status === "answered" ? "answered" : "source_insufficient",
    answer: {status: data.status === "unresolved" ? "abstain" : data.status,
      answer: value, unit: data.unit === "count" ? null : data.unit, citation_details: details,
      explanation: data.calculation || data.reason, uncertainties: [], coverage,
      briefing: [
        ...(data.calculation ? [{section: "basis" as const, text: data.calculation, citations: citations.map(c => c.source_index!)}] : []),
        ...(data.reason ? [{section: "limitation" as const, text: data.reason, citations: []}] : [])
      ]}};
}
export async function askReader(prepared: ReaderPage, question: string) {
  const body = new FormData();
  body.append("file", prepared.payload);
  body.append("page", String(prepared.payloadPage));
  body.append("question", question);
  const response = await fetch("/api/reader", {method: "POST", body});
  const result = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof result?.detail === "string" ? result.detail : typeof result?.error === "string" ? result.error : null;
    throw new Error(detail || (response.status === 413 ? "This sheet exceeds the upload limit." :
      response.status === 504 ? "The reader timed out on this sheet. Try a smaller page export." :
      "The drawing reader could not finish this question (" + response.status + ")."));
  }
  return mappedAnswer(result, prepared);
}
