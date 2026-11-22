import { useEffect, useRef, useState } from "react";
import {
  Minus,
  Plus,
  Maximize,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  LoaderCircle,
  ListChecks,
  LocateFixed,
  LocateOff,
  DoorOpen,
  Box,
  Scan,
} from "lucide-react";
import { visibleCitations, visibleCountObjects, type Drawing, type AgentResult } from "../lib/api";
import { AlgorithmOverlay } from "./AlgorithmOverlay";
import { SpatialPlan } from "./SpatialPlan";
import type { AgentRun, EvidenceShape, Region } from "../lib/runs";
import { liveActivity } from "../lib/activity";
interface Props {
  spatialSample?: boolean;
  hasTask: boolean;
  taskOpen: boolean;
  onTaskDetails: () => void;
  liveRun?: AgentRun;
  followLive?: boolean;
  onFollowLive?: (value: boolean) => void;
  shapes?: EvidenceShape[];
  comparisonSplit?: number | null;
  focusRegion?: Region | null;
  onManualView?: () => void;
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
  roomPick?: {german:boolean; onPoint:(point:[number,number])=>void; onCancel:()=>void};
}
export function PlanCanvas2D({
  spatialSample = false,
  hasTask, taskOpen, onTaskDetails,
  drawing,
  liveRun, followLive = false, onFollowLive,
  shapes = [],
  comparisonSplit,
  focusRegion,
  onManualView,
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
  roomPick,
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
  const countObjects = result ? visibleCountObjects(result, drawing) : [];
  const [showCandidates, setShowCandidates] = useState(true);
  const [spatialRequested, setSpatialRequested] = useState(spatialSample);
  const [spatialVisited, setSpatialVisited] = useState(spatialSample);
  const spatial = spatialRequested && !roomPick && !focusRegion && comparisonSplit == null;
  const activity = liveActivity(liveRun, drawing.source);
  useEffect(() => {
    if (spatial || (!focusRegion && selectedIndex === null) || !viewport.current) return;
    const box =
      focusRegion?.canonical_bbox ||
      (selectedIndex !== null
        ? citations[selectedIndex]?.canonical_bbox
        : null);
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
  }, [selectedIndex, result, drawing, focusRegion, onZoomChange, spatial]);
  useEffect(() => {
    if (
      spatial ||
      (!focusRegion && selectedIndex === null) ||
      !viewport.current ||
      !sheet.current
    )
      return;
    const box =
      focusRegion?.canonical_bbox ||
      (selectedIndex !== null
        ? citations[selectedIndex]?.canonical_bbox
        : null);
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
  }, [selectedIndex, result, drawing, zoom, focusRegion, spatial]);
  const answer = result?.answer;
  const caption = answer?.answer != null
    ? (result?.measurement?.approximate ? "≈ " : "") + [answer.answer, answer.unit].filter(v => v != null && v !== "").join(" ")
    : undefined;
  function choosePoint(clientX:number, clientY:number, bounds:DOMRect) {
    if (!roomPick || !bounds.width || !bounds.height) return;
    const x=(clientX-bounds.left)/bounds.width, y=(clientY-bounds.top)/bounds.height;
    if (x>0 && x<1 && y>0 && y<1) roomPick.onPoint([x,y]);
  }
  return (
    <section className="drawing-viewer" aria-label="Drawing viewer">
      {roomPick && <div className="room-pick-banner" data-testid="room-picker">
        <span role="status">{roomPick.german ? "Klick in die freie Bodenfläche des Flurs oder Raums." :
          "Click an open spot inside the hallway or room."}</span>
        <button type="button" onClick={roomPick.onCancel}>{roomPick.german ? "Abbrechen" : "Cancel"}</button>
      </div>}
      {!roomPick && <button type="button" className="view-mode-button" aria-label={spatial ? "Return to 2D" : "Explore in 3D"}
        aria-pressed={spatial} disabled={comparisonSplit != null} onClick={() => {
          onManualView?.();
          setSpatialVisited(true);
          setSpatialRequested(!spatial);
        }}>
        {spatial ? <Scan size={15}/> : <Box size={15}/>} {spatial ? "2D plan" : "3D view"}
      </button>}
      {spatialVisited && <SpatialPlan drawing={drawing} citations={citations} selectedIndex={selectedIndex}
        onSelectEvidence={onSelectEvidence} caption={caption} sample={spatialSample} hidden={!spatial} />}
      <div
        ref={viewport}
        hidden={spatial}
        className={`plan-viewport ${panning ? "is-panning" : ""} ${roomPick ? "is-room-picking" : ""}`}
        data-testid="plan-viewport"
        onWheel={() => onManualView?.()}
        onPointerDown={(e) => {
          onManualView?.();
          if (
            roomPick || e.pointerType !== "mouse" ||
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
            tabIndex={roomPick ? 0 : undefined}
            aria-keyshortcuts={roomPick ? "Enter Escape" : undefined}
            onClick={roomPick ? event => choosePoint(event.clientX,event.clientY,event.currentTarget.getBoundingClientRect()) : undefined}
            onKeyDown={roomPick ? event => {
              if (event.key==="Escape") roomPick.onCancel();
              if (event.key!=="Enter") return;
              event.preventDefault();
              const bounds = event.currentTarget.getBoundingClientRect();
              const visible = viewport.current?.getBoundingClientRect() ?? bounds;
              choosePoint((Math.max(bounds.left,visible.left)+Math.min(bounds.right,visible.right))/2,
                          (Math.max(bounds.top,visible.top)+Math.min(bounds.bottom,visible.bottom))/2,bounds);
            } : undefined}
          />
          <AlgorithmOverlay
            shapes={shapes}
            split={comparisonSplit}
            width={drawing.width}
            height={drawing.height}
          />
          <svg
            className="evidence-overlay"
            viewBox={`0 0 ${drawing.width} ${drawing.height}`}
            aria-label="Source evidence locations"
            data-testid="evidence-overlay"
          >
            {activity?.regions.map((region, i) => (
              <rect key={`${activity.sequence}-${i}`} data-testid="live-inspection-region" className={activity.currentRegion ? "live-region current" : "live-region"}
                x={region.canonical_bbox[0]} y={region.canonical_bbox[1]}
                width={region.canonical_bbox[2] - region.canonical_bbox[0]}
                height={region.canonical_bbox[3] - region.canonical_bbox[1]}
                style={{stroke: activity.views[i].color, fill: `${activity.views[i].color}0c`, strokeDasharray: "8 5"}} vectorEffect="non-scaling-stroke"><title>{activity.currentRegion ? "Requested inspection region" : "Last source region"}</title></rect>
            ))}
            {focusRegion && (
              <rect
                data-testid="activity-region"
                x={focusRegion.canonical_bbox[0]}
                y={focusRegion.canonical_bbox[1]}
                width={
                  focusRegion.canonical_bbox[2] - focusRegion.canonical_bbox[0]
                }
                height={
                  focusRegion.canonical_bbox[3] - focusRegion.canonical_bbox[1]
                }
                stroke="#4669c7"
                fill="#4669c70a"
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
                strokeDasharray="7 5"
              />
            )}
            {citations.map((c, index) => {
              const [x0, y0, x1, y1] = c.canonical_bbox,
                selected = selectedIndex === index;
              const detail = result?.answer?.citation_details?.find(d => d.citation === c.source_index);
              const measured = detail?.kind === "measurement" && (c.polygon || c.line);
              if (focusRegion || (!selected && !measured && (countObjects.length > 0 || detail?.kind !== "detection"))) return null;
              return (
                <g key={index}>
                  {(selected || measured) && c.polygon && <polygon points={c.polygon.map(p => p.join(",")).join(" ")}
                    fill="#81b89550" stroke="#35664c" strokeWidth={selected ? 2.5 : 1.5}
                    vectorEffect="non-scaling-stroke" data-testid="room-outline" pointerEvents="none" />}
                  {(selected || measured) && c.line && <polyline points={c.line.map(p => p.join(",")).join(" ")}
                    fill="none" stroke="#a66a26" strokeWidth={selected ? 4 : 2.5}
                    vectorEffect="non-scaling-stroke" data-testid="scale-anchor" pointerEvents="none" />}
                  {selected && c.point && <circle cx={c.point[0]} cy={c.point[1]} r={Math.max(1, drawing.width / 500)}
                    fill="#42634a" pointerEvents="none" />}
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
                    fill={measured ? "transparent" : selected ? "#001d0020" : "#001d000a"}
                    stroke={measured ? "none" : selected ? "#001d00" : "#42634a"}
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
            {showCandidates && countObjects.map(o => {
              const [x0, y0, x1, y1] = o.canonical_bbox;
              const uncertain = o.status === "uncertain";
              return <rect key={o.id} data-testid="count-candidate" data-status={o.status}
                x={x0} y={y0} width={x1 - x0} height={y1 - y0}
                fill={uncertain ? "#b7791014" : "#42634a0c"}
                stroke={uncertain ? "#a66a00" : "#42634a"}
                strokeWidth={2} strokeDasharray={uncertain ? "5 3" : undefined}
                vectorEffect="non-scaling-stroke" role="img" pointerEvents="all" tabIndex={0}
                aria-label={`${o.id}: ${o.status}. ${o.reason}`}>
                <title>{o.id}: {o.status} — {o.reason}</title>
              </rect>;
            })}
          </svg>
          {activity?.views.map(({region, label, active, color}, i) => <span key={`${activity.sequence}-label-${i}`} className="live-region-label" data-active={active}
            style={{backgroundColor: color, color: "#fff", left: `${region.canonical_bbox[0] / drawing.width * 100}%`, top: `${region.canonical_bbox[1] / drawing.height * 100}%`}}>{label}</span>)}
        </div>
      </div>
      {comparisonSplit !== null && comparisonSplit !== undefined && (
        <div className="comparison-legend">
          <span>ML predictions</span>
          <span>Pixel + geometry rules</span>
        </div>
      )}
      <div className={`viewer-toolbar ${spatial ? "spatial-sheet-toolbar" : ""}`} aria-label="Drawing controls">
        {!spatial && countObjects.length > 0 && <button className="icon-button" aria-pressed={showCandidates}
          aria-label="Show door candidates" onClick={() => setShowCandidates(v => !v)}
          title={`${countObjects.length} door candidates. Green: supported; dashed amber: uncertain. Not human-verified.`}>
          <DoorOpen size={16} />
        </button>}
        {(!spatial || (isPdf && (drawing.page_count ?? 1) > 1)) && <div className="page-flipper" aria-label="Page navigation">
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
                  (!drawing.page_count || nextPage <= drawing.page_count) &&
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
                max={drawing.page_count}
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
          {isPdf && drawing.page_count && <span className="page-total">/ {drawing.page_count}</span>}
          <button
            className="icon-button"
            aria-label="Next page"
            title="Next page"
            disabled={disabled || !isPdf || (drawing.page_count !== undefined && drawing.page >= drawing.page_count)}
            onClick={() => onPageChange(drawing.page + 1)}
          >
            <ChevronRight size={16} />
          </button>
        </div>}
        {!spatial && <><span className="toolbar-divider" />
        <button
          className="icon-button"
          aria-label="Zoom out"
          title="Zoom out"
          disabled={zoom <= 50}
          onClick={() => {
            onManualView?.();
            onZoomChange(Math.max(50, zoom - 25));
          }}
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
          onClick={() => {
            onManualView?.();
            onZoomChange(Math.min(800, zoom + 25));
          }}
        >
          <Plus size={16} />
        </button>
        <span className="toolbar-divider" />
        <button
          className="icon-button"
          aria-label="Fit drawing"
          title="Fit drawing"
          onClick={() => {
            onManualView?.();
            onZoomChange(100);
          }}
        >
          <Maximize size={15} />
        </button></>}
        <a
          className="icon-button"
          href={drawing.original_url ? drawing.original_url + (isPdf ? "#page=" + drawing.page : "") : drawing.image_url}
          target="_blank"
          rel="noreferrer"
          aria-label="Open full-resolution source"
          title="Open original"
        >
          <ExternalLink size={15} />
        </a>
        {(!spatial || hasTask) && <span className="toolbar-divider" />}
        {!spatial && onFollowLive && <button
          className="icon-button follow-view-button"
          aria-label="Follow view"
          title={followLive ? "Following agent view — click to pause" : "Follow agent view"}
          aria-pressed={followLive}
          onClick={() => onFollowLive(!followLive)}
        >
          {followLive ? <LocateFixed size={17} /> : <LocateOff size={17} />}
        </button>}
        {(!spatial || hasTask) && <button className="icon-button task-details-button" aria-label="Task details"
          title="Latest task details" aria-pressed={taskOpen} disabled={!hasTask} onClick={onTaskDetails}>
          <ListChecks size={17} />
        </button>}
      </div>
      {loadingPage && (
        <div className="page-loading" role="status">
          <LoaderCircle size={15} className="spinning" />
          Opening page…
        </div>
      )}
      {!roomPick && !spatial && !activity && citations.length > 0 && (
        <div className="evidence-count">
          <span />
          {result?.measurement?.approximate ? caption : citations.length === 1
            ? "1 linked source"
            : `${citations.length} linked sources`}
        </div>
      )}
    </section>
  );
}
