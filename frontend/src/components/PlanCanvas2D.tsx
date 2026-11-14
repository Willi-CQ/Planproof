import { useEffect, useRef } from "react";
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Upload,
  X,
  ExternalLink,
} from "lucide-react";
import { visibleCitations, type Drawing, type AgentResult } from "../lib/api";

interface Props {
  drawing: Drawing | null;
  name?: string;
  result?: AgentResult;
  selectedIndex: number | null;
  onSelectEvidence: (index: number) => void;
  zoom: number;
  onZoomChange: (zoom: number) => void;
  onReset: () => void;
  onUpload: () => void;
  disabled: boolean;
  uploading: boolean;
}
export function PlanCanvas2D({
  drawing,
  name,
  result,
  selectedIndex,
  onSelectEvidence,
  zoom,
  onZoomChange,
  onReset,
  onUpload,
  disabled,
  uploading,
}: Props) {
  const viewport = useRef<HTMLDivElement>(null);
  const sheet = useRef<HTMLDivElement>(null);
  const citations = result && drawing ? visibleCitations(result, drawing) : [];
  useEffect(() => {
    if (
      selectedIndex === null ||
      !drawing ||
      !viewport.current ||
      !sheet.current
    )
      return;
    const box = citations[selectedIndex]?.canonical_bbox;
    if (!box) return;
    const container = viewport.current,
      page = sheet.current;
    container.scrollTo({
      left:
        page.offsetLeft +
        ((box[0] + box[2]) / 2 / drawing.width) * page.clientWidth -
        container.clientWidth / 2,
      top:
        page.offsetTop +
        ((box[1] + box[3]) / 2 / drawing.height) * page.clientHeight -
        container.clientHeight / 2,
      behavior: "smooth",
    });
  }, [selectedIndex, result, drawing, zoom]);

  return (
    <section
      aria-label="2D Plan Viewer"
      className="flex min-h-[420px] flex-1 flex-col overflow-hidden rounded-lg border border-slate-200 bg-white md:min-h-0"
    >
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-slate-200 bg-[#F8FAFC] px-4 py-3">
        <div className="min-w-0">
          <h2 className="font-display text-sm font-semibold">
            {drawing ? "Source drawing" : "Your drawing, in focus"}
          </h2>
          <p className="max-w-64 truncate text-xs text-slate-500">
            {drawing ? `${name} · page ${drawing.page}` : "PDF · PNG · JPEG"}
          </p>
        </div>
        <div className="flex items-center gap-1">
          <button
            className="viewer-button"
            aria-label="Zoom out"
            disabled={!drawing || zoom <= 50}
            onClick={() => onZoomChange(Math.max(50, zoom - 25))}
          >
            <ZoomOut size={16} />
          </button>
          <span
            className="min-w-12 text-center font-mono text-xs"
            data-testid="zoom-level"
          >
            {zoom}%
          </span>
          <button
            className="viewer-button"
            aria-label="Zoom in"
            disabled={!drawing || zoom >= 800}
            onClick={() => onZoomChange(Math.min(800, zoom + 25))}
          >
            <ZoomIn size={16} />
          </button>
          <button
            className="viewer-button"
            aria-label="Fit drawing"
            disabled={!drawing}
            onClick={() => onZoomChange(100)}
          >
            <RotateCcw size={15} />
          </button>
          {drawing && (
            <a
              className="viewer-button"
              href={drawing.image_url}
              target="_blank"
              rel="noreferrer"
              aria-label="Open full-resolution source"
            >
              <ExternalLink size={15} />
            </a>
          )}
          <button
            className="viewer-button"
            aria-label="Remove drawing"
            disabled={!drawing || disabled}
            onClick={onReset}
          >
            <X size={15} />
          </button>
        </div>
      </div>
      <div
        ref={viewport}
        className="relative flex-1 overflow-auto bg-[#DDE1E6] p-4 sm:p-6"
        data-testid="plan-viewport"
      >
        {drawing ? (
          <div
            ref={sheet}
            className="relative mx-auto bg-white shadow-lg"
            style={{ width: `${zoom}%` }}
          >
            <img
              src={drawing.image_url}
              alt={`Source drawing: ${name}, page ${drawing.page}`}
              width={drawing.width}
              height={drawing.height}
              className="block h-auto w-full"
              data-testid="source-image"
            />
            <svg
              className="pointer-events-none absolute inset-0 h-full w-full"
              viewBox={`0 0 ${drawing.width} ${drawing.height}`}
              aria-label="Source evidence locations"
              data-testid="evidence-overlay"
            >
              {citations.map((c, index) => {
                const [x0, y0, x1, y1] = c.canonical_bbox,
                  selected = selectedIndex === index;
                return (
                  <g key={index}>
                    <rect
                      data-testid="evidence-box"
                      tabIndex={0}
                      role="button"
                      aria-label={`Highlight evidence ${index + 1}`}
                      className="pointer-events-auto cursor-pointer"
                      x={x0}
                      y={y0}
                      width={x1 - x0}
                      height={y1 - y0}
                      fill={selected ? "#1e40af35" : "#1e40af10"}
                      stroke={selected ? "#f97316" : "#1e40af"}
                      strokeWidth={selected ? 3 : 2}
                      vectorEffect="non-scaling-stroke"
                      onClick={() => onSelectEvidence(index)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          onSelectEvidence(index);
                        }
                      }}
                    />
                    <title>
                      Evidence {index + 1}: source pixels{" "}
                      {c.canonical_bbox.join(", ")}
                    </title>
                  </g>
                );
              })}
            </svg>
          </div>
        ) : (
          <div className="flex min-h-[350px] h-full flex-col items-center justify-center rounded border border-dashed border-slate-400 bg-white/60 px-8 text-center">
            <Upload className="mb-4 text-slate-400" size={32} />
            <h3 className="font-display text-xl font-semibold">
              {uploading
                ? "Opening your drawing…"
                : "Start with a construction drawing"}
            </h3>
            <p className="mt-2 max-w-sm text-sm leading-relaxed text-slate-500">
              Upload a plan, ask a question, and inspect the answer directly on
              its source.
            </p>
            <button
              disabled={disabled}
              onClick={onUpload}
              className="mt-5 rounded-md bg-[#1E40AF] px-4 py-2.5 text-sm font-medium text-white disabled:opacity-40"
            >
              Choose a drawing
            </button>
            <p className="mt-3 text-xs text-slate-500">
              Up to 10 MiB · original image retained
            </p>
          </div>
        )}
      </div>
      <div className="shrink-0 border-t border-slate-200 px-4 py-2 text-xs text-slate-500">
        {drawing
          ? `${drawing.width.toLocaleString()} × ${drawing.height.toLocaleString()} source pixels · ${citations.length} evidence regions`
          : "Evidence highlights appear here after the agent answers."}
      </div>
    </section>
  );
}
