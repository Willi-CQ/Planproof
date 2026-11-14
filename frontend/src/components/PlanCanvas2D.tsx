import { useEffect, useRef, useState } from "react";
import {
  Minus,
  Plus,
  Maximize,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  LoaderCircle,
} from "lucide-react";
import { visibleCitations, type Drawing, type AgentResult } from "../lib/api";
interface Props {
  drawing: Drawing;
  name?: string;
  result?: AgentResult;
  selectedIndex: number | null;
  onSelectEvidence: (index: number) => void;
  zoom: number;
  onZoomChange: (zoom: number) => void;
  isPdf: boolean;
  disabled: boolean;
  loadingPage: boolean;
  onPageChange: (page: number) => void;
}
export function PlanCanvas2D({
  drawing,
  name,
  result,
  selectedIndex,
  onSelectEvidence,
  zoom,
  onZoomChange,
  isPdf,
  disabled,
  loadingPage,
  onPageChange,
}: Props) {
  const viewport = useRef<HTMLDivElement>(null),
    sheet = useRef<HTMLDivElement>(null);
  const pan = useRef<{
    x: number;
    y: number;
    left: number;
    top: number;
  } | null>(null);
  const [panning, setPanning] = useState(false);
  const [pageInput, setPageInput] = useState(String(drawing.page));
  useEffect(
    () => setPageInput(String(drawing.page)),
    [drawing.document_id, drawing.page],
  );
  const citations = result ? visibleCitations(result, drawing) : [];
  useEffect(() => {
    if (selectedIndex === null || !viewport.current) return;
    const box = citations[selectedIndex]?.canonical_bbox;
    const container = viewport.current;
    if (!box || container.clientWidth === 0) return;
    const style = getComputedStyle(container);
    const availableWidth =
      container.clientWidth -
      parseFloat(style.paddingLeft) -
      parseFloat(style.paddingRight);
    const widthAtFit = ((box[2] - box[0]) / drawing.width) * availableWidth;
    const heightAtFit = ((box[3] - box[1]) / drawing.width) * availableWidth;
    const magnification = Math.min(
      Math.min(300, container.clientWidth * 0.6) / widthAtFit,
      (container.clientHeight * 0.55) / heightAtFit,
    );
    onZoomChange(Math.max(100, Math.min(400, Math.round(magnification * 100))));
  }, [selectedIndex, result, drawing, onZoomChange]);
  useEffect(() => {
    if (selectedIndex === null || !viewport.current || !sheet.current) return;
    const box = citations[selectedIndex]?.canonical_bbox;
    if (!box) return;
    const container = viewport.current,
      source = sheet.current;
    container.scrollTo({
      left:
        source.offsetLeft +
        ((box[0] + box[2]) / 2 / drawing.width) * source.clientWidth -
        container.clientWidth / 2,
      top:
        source.offsetTop +
        ((box[1] + box[3]) / 2 / drawing.height) * source.clientHeight -
        container.clientHeight / 2,
      behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "instant"
        : "smooth",
    });
  }, [selectedIndex, result, drawing, zoom]);
  return (
    <section className="drawing-viewer" aria-label="Drawing viewer">
      <div
        ref={viewport}
        className={`plan-viewport ${panning ? "is-panning" : ""}`}
        data-testid="plan-viewport"
        onPointerDown={(e) => {
          if (
            e.pointerType !== "mouse" ||
            e.button !== 0 ||
            (e.target as Element).closest('[role="button"]')
          )
            return;
          const v = viewport.current!;
          pan.current = {
            x: e.clientX,
            y: e.clientY,
            left: v.scrollLeft,
            top: v.scrollTop,
          };
          v.setPointerCapture(e.pointerId);
          setPanning(true);
        }}
        onPointerMove={(e) => {
          if (pan.current && viewport.current) {
            viewport.current.scrollLeft =
              pan.current.left - (e.clientX - pan.current.x);
            viewport.current.scrollTop =
              pan.current.top - (e.clientY - pan.current.y);
          }
        }}
        onPointerUp={() => {
          pan.current = null;
          setPanning(false);
        }}
        onLostPointerCapture={() => {
          pan.current = null;
          setPanning(false);
        }}
      >
        <div
          ref={sheet}
          className="drawing-sheet"
          style={{ width: `${zoom}%` }}
        >
          <img
            src={drawing.image_url}
            alt={`Source drawing: ${name}, page ${drawing.page}`}
            width={drawing.width}
            height={drawing.height}
            data-testid="source-image"
            draggable={false}
          />
          <svg
            className="evidence-overlay"
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
                    className="evidence-box"
                    x={x0}
                    y={y0}
                    width={x1 - x0}
                    height={y1 - y0}
                    fill={selected ? "#001d0020" : "#001d000a"}
                    stroke={selected ? "#001d00" : "#42634a"}
                    strokeWidth={selected ? 3 : 1.5}
                    vectorEffect="non-scaling-stroke"
                    onClick={() => onSelectEvidence(index)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onSelectEvidence(index);
                      }
                    }}
                  />
                  <title>Evidence {index + 1}</title>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
      <div className="viewer-toolbar" aria-label="Drawing controls">
        <div className="page-flipper" aria-label="Page navigation">
          <button
            className="icon-button"
            aria-label="Previous page"
            title="Previous page"
            disabled={disabled || !isPdf || drawing.page <= 1}
            onClick={() => onPageChange(drawing.page - 1)}
          >
            <ChevronLeft size={16} />
          </button>
          {isPdf ? (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                const nextPage = Number(pageInput);
                setPageInput(String(drawing.page));
                if (
                  Number.isInteger(nextPage) &&
                  nextPage > 0 &&
                  nextPage !== drawing.page
                )
                  onPageChange(nextPage);
              }}
            >
              <input
                aria-label="PDF page"
                title="Go to page (Enter)"
                type="number"
                min="1"
                step="1"
                required
                value={pageInput}
                disabled={disabled}
                onChange={(e) => setPageInput(e.target.value)}
                onBlur={() => setPageInput(String(drawing.page))}
              />
            </form>
          ) : (
            <span className="current-page" aria-label="Current page">
              1
            </span>
          )}
          <button
            className="icon-button"
            aria-label="Next page"
            title="Next page"
            disabled={disabled || !isPdf}
            onClick={() => onPageChange(drawing.page + 1)}
          >
            <ChevronRight size={16} />
          </button>
        </div>
        <span className="toolbar-divider" />
        <button
          className="icon-button"
          aria-label="Zoom out"
          title="Zoom out"
          disabled={zoom <= 50}
          onClick={() => onZoomChange(Math.max(50, zoom - 25))}
        >
          <Minus size={16} />
        </button>
        <span data-testid="zoom-level" className="zoom-value">
          {zoom}%
        </span>
        <button
          className="icon-button"
          aria-label="Zoom in"
          title="Zoom in"
          disabled={zoom >= 800}
          onClick={() => onZoomChange(Math.min(800, zoom + 25))}
        >
          <Plus size={16} />
        </button>
        <span className="toolbar-divider" />
        <button
          className="icon-button"
          aria-label="Fit drawing"
          title="Fit drawing"
          onClick={() => onZoomChange(100)}
        >
          <Maximize size={15} />
        </button>
        <a
          className="icon-button"
          href={drawing.image_url}
          target="_blank"
          rel="noreferrer"
          aria-label="Open full-resolution source"
          title="Open original"
        >
          <ExternalLink size={15} />
        </a>
      </div>
      {loadingPage && (
        <div className="page-loading" role="status">
          <LoaderCircle size={15} className="spinning" />
          Opening page…
        </div>
      )}
      {citations.length > 0 && (
        <div className="evidence-count">
          <span />
          {citations.length === 1
            ? "1 source highlight"
            : `${citations.length} source highlights`}
        </div>
      )}
    </section>
  );
}
