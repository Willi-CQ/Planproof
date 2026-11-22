import { useEffect, useRef, useState } from "react";
import {
  Box, ChevronLeft, ChevronRight, Focus, Layers3, MapPin, Minus, PencilRuler,
  Plus, RotateCcw, Scan, Undo2, X,
} from "lucide-react";
import type { Citation, Drawing } from "../lib/api";
import { overviewCamera, projectPlan, simpleFootprint, type PlanPoint } from "../lib/planProjection";
import "./spatial-plan.css";

type Guide = { name: string; points: PlanPoint[] };
const outline = (points: PlanPoint[]) => points.map(p => p.join(",")).join(" ");
const cornerPoint = (p: number[]): PlanPoint => [p[0], p[1]];
const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));
const anchor = (c: Citation): PlanPoint => {
  const [left, top, right, bottom] = c.canonical_bbox;
  return c.point ? cornerPoint(c.point) : c.polygon?.length ? cornerPoint(c.polygon[0])
    : [(left + right) / 2, (top + bottom) / 2];
};

export const sampleRooms: Guide[] = [
  {name:"Room A", points:[[160,180],[552,180],[552,404],[160,404]]},
  {name:"Room B", points:[[568,180],[960,180],[960,404],[568,404]]},
  {name:"Hallway", points:[[160,420],[960,420],[960,580],[160,580]]},
];

interface Props {
  drawing: Drawing;
  citations: Citation[];
  selectedIndex: number | null;
  onSelectEvidence: (index: number) => void;
  caption?: string;
  sample?: boolean;
  hidden: boolean;
}

export function SpatialPlan({drawing, citations, selectedIndex, onSelectEvidence, caption, sample, hidden}: Props) {
  const svg = useRef<SVGSVGElement>(null), model = useRef<SVGGElement>(null);
  const gesture = useRef<{x:number; y:number; yaw:number; tilt:number} | null>(null);
  const [frame, setFrame] = useState<PlanPoint>([720, 520]);
  const [camera, setCamera] = useState(overviewCamera);
  const [focal, setFocal] = useState<PlanPoint>();
  const [showEvidence, setShowEvidence] = useState(true);
  const [showVolumes, setShowVolumes] = useState(true);
  const [depth, setDepth] = useState(45);
  const [rooms, setRooms] = useState<Guide[]>([]);
  const [draft, setDraft] = useState<PlanPoint[] | null>(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const {width, height} = drawing;
  const view = projectPlan(width, height, frame, camera, focal);
  const thickness = Math.min(width, height) * .032;
  const relief = Math.min(width, height) * (.035 + depth * .0014);
  const corners: PlanPoint[] = [[0,0],[width,0],[width,height],[0,height]];
  const zones = [...(sample ? sampleRooms : []), ...rooms];
  const active = selectedIndex === null ? undefined : citations[selectedIndex];
  const pinIndexes = citations.slice(0, 48).map((_, i) => i);
  if (active && selectedIndex !== null && !pinIndexes.includes(selectedIndex)) pinIndexes.push(selectedIndex);

  useEffect(() => {
    const element = svg.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      const {width, height} = entry.contentRect;
      if (width > 0 && height > 0) setFrame([width, height]);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  function overview() {
    setFocal(undefined);
    setCamera(overviewCamera);
  }

  function changeZoom(amount: number) {
    setCamera(value => ({...value, zoom:clamp(value.zoom + amount, .65, 4)}));
  }

  function focusEvidence() {
    if (!active) return;
    const [left, top, right, bottom] = active.canonical_bbox;
    setFocal([(left + right) / 2, (top + bottom) / 2]);
    setCamera(value => ({...value, zoom:clamp(Math.min(width / (right-left), height / (bottom-top)) / 2.5, 1.2, 3)}));
    setShowEvidence(true);
  }

  function finishZone() {
    if (!draft || !simpleFootprint(draft)) {
      setError("Use at least three distinct corners and keep the edges from crossing.");
      return;
    }
    setRooms(previous => [...previous, {name:`Sketch ${previous.length + 1}`, points:draft}]);
    setDraft(null);
    setShowVolumes(true);
    setError("");
    overview();
  }

  function drawSides(points: PlanPoint[], bottom: number, top: number, kind: string) {
    return points.map((p, i) => {
      const next = points[(i + 1) % points.length];
      return {
        index:i, distance:(view.depth(p)+view.depth(next))/2,
        face:outline([view.point(p,bottom),view.point(next,bottom),view.point(next,top),view.point(p,top)]),
      };
    }).sort((a,b) => a.distance-b.distance).map(({index,face}, i) => (
      <polygon key={index} points={face} className={kind} data-light={i % 2 === 0} />
    ));
  }

  return <div className="spatial-view" hidden={hidden}>
    <div className="spatial-top">
      <div className="spatial-controls" aria-label="3D camera presets">
        <button type="button" onClick={overview} aria-label="Reset 3D view" title="Reset 3D view"><RotateCcw size={16} /></button>
        <button type="button" onClick={() => {setFocal(undefined);setCamera({yaw:0,tilt:0,zoom:1});}}
          aria-pressed={camera.tilt === 0} title="Top view"><Scan size={16} /><span>Top view</span></button>
        <button type="button" onClick={() => setShowEvidence(value => !value)} disabled={!citations.length}
          aria-pressed={showEvidence && citations.length > 0} title="Evidence pins"><MapPin size={16} /><span>Evidence</span></button>
      </div>
    </div>
    <svg ref={svg} className={`spatial-stage ${draft ? "is-sketching" : dragging ? "is-rotating" : ""}`}
      viewBox={`0 0 ${frame[0]} ${frame[1]}`} aria-label="Interactive 3D drawing" role="group" tabIndex={0}
      data-testid="spatial-stage" data-yaw={Math.round(camera.yaw)} data-tilt={Math.round(camera.tilt)}
      onKeyDown={event => {
        if (event.target !== event.currentTarget || draft) return;
        const turns: Record<string, PlanPoint> = {ArrowLeft:[-12,0],ArrowRight:[12,0],ArrowUp:[0,8],ArrowDown:[0,-8]};
        const turn = turns[event.key];
        if (!turn) return;
        event.preventDefault();
        setCamera(value => ({...value, yaw:value.yaw+turn[0], tilt:clamp(value.tilt+turn[1],0,68)}));
      }}
      onPointerDown={event => {
        if (draft || event.button !== 0 || (event.target as Element).closest("[data-source-pin]")) return;
        gesture.current = {x:event.clientX,y:event.clientY,yaw:camera.yaw,tilt:camera.tilt};
        event.currentTarget.setPointerCapture(event.pointerId);
        setDragging(true);
      }}
      onPointerMove={event => {
        const start = gesture.current;
        if (!start) return;
        setCamera(value => ({...value, yaw:(start.yaw + (event.clientX-start.x)*.52) % 360,
          tilt:clamp(start.tilt - (event.clientY-start.y)*.32,0,68)}));
      }}
      onLostPointerCapture={() => {gesture.current = null;setDragging(false);}}
      onClick={event => {
        if (!draft || draft.length >= 50) return;
        const transform = model.current?.getScreenCTM();
        if (!transform) return;
        const {x,y} = new DOMPoint(event.clientX,event.clientY).matrixTransform(transform.inverse());
        if (x < 0 || y < 0 || x > width || y > height) return;
        const point: PlanPoint = [Math.round(x), Math.round(y)];
        if (draft.some(p => p[0] === point[0] && p[1] === point[1])) return;
        setDraft([...draft,point]);
        setError("");
      }}>
      <polygon points={outline(corners.map(p => view.point(p,-thickness*2)))} className="spatial-shadow" />
      {drawSides(corners,-thickness,0,"spatial-slab")}
      <g ref={model} transform={view.matrix} data-testid="spatial-projection">
        <rect width={width} height={height} fill="#fff" />
        <image width={width} height={height} href={drawing.image_url} preserveAspectRatio="none" />
        {showEvidence && active && (active.polygon && active.polygon.length >= 3
          ? <polygon className="spatial-highlight" points={outline(active.polygon.map(cornerPoint))} />
          : <rect className="spatial-highlight" x={active.canonical_bbox[0]} y={active.canonical_bbox[1]}
            width={active.canonical_bbox[2]-active.canonical_bbox[0]} height={active.canonical_bbox[3]-active.canonical_bbox[1]} />)}
        {draft && <>
          <polyline points={outline(draft)} className="spatial-draft" />
          {draft.map(([x,y]) => <circle key={`${x},${y}`} cx={x} cy={y} r={3.2 / view.scale} className="spatial-corner" />)}
        </>}
      </g>
      {showVolumes && !draft && zones.map((zone, i) => {
        const points = zone.points;
        const center: PlanPoint = [points.reduce((sum,p) => sum+p[0],0)/points.length,
          points.reduce((sum,p) => sum+p[1],0)/points.length];
        return {zone, center, index:i};
      }).sort((a,b) => view.depth(a.center)-view.depth(b.center)).map(({zone,center,index}) => {
        const [x,y] = view.point(center,relief);
        return <g key={zone.name} className="spatial-room" data-testid="sketch-volume" data-hallway={index === 2 && sample}>
          {drawSides(zone.points,0,relief,"spatial-wall")}
          <polygon points={outline(zone.points.map(point => view.point(point,relief)))} className="spatial-roof" />
          {sample && index < 3 && frame[0] > 490 && <text x={x} y={y} className="spatial-room-label">{zone.name}</text>}
        </g>;
      })}
      {showEvidence && !draft && pinIndexes.map(i => {
        const c = citations[i], active = selectedIndex === i;
        const [x,y] = view.point(anchor(c));
        const rise = active ? 30 : 19;
        const select = () => onSelectEvidence(i);
        return <g key={i} data-source-pin data-active={active} className="spatial-pin" data-testid="spatial-pin"
          tabIndex={0} role="button" aria-label={`Show 3D evidence ${c.source_index ?? i+1}`}
          aria-pressed={active} onClick={select} onKeyDown={event => {
            if (event.key === "Enter" || event.key === " ") {event.preventDefault();select();}
          }}>
          <line x1={x} y1={y} x2={x} y2={y-rise} />
          <circle cx={x} cy={y} r={2.5} className="spatial-pin-base" />
          <circle cx={x} cy={y-rise} r={active ? 12 : 10} className="spatial-pin-head" />
          <text x={x} y={y-rise} dy=".34em">{c.source_index ?? i+1}</text>
        </g>;
      })}
    </svg>
    {active && showEvidence && !draft && <div className="spatial-caption">
      <MapPin size={15} /><span>{caption || `Source ${active.source_index ?? (selectedIndex ?? 0) + 1}`}</span>
      <button type="button" onClick={focusEvidence} aria-label="Focus selected evidence" title="Focus selected evidence"><Focus size={15}/></button>
    </div>}
    <div className="spatial-actions">
      <div className="spatial-controls" aria-label="3D camera controls">
        <button type="button" title="Rotate left" aria-label="Rotate left" onClick={() => setCamera(v => ({...v,yaw:v.yaw-30}))}><ChevronLeft size={17}/></button>
        <button type="button" title="Rotate right" aria-label="Rotate right" onClick={() => setCamera(v => ({...v,yaw:v.yaw+30}))}><ChevronRight size={17}/></button>
        <span className="spatial-divider" />
        <button type="button" title="3D zoom out" aria-label="3D zoom out" disabled={camera.zoom <= .65} onClick={() => changeZoom(-.25)}><Minus size={16}/></button>
        <button type="button" title="3D zoom in" aria-label="3D zoom in" disabled={camera.zoom >= 4} onClick={() => changeZoom(.25)}><Plus size={16}/></button>
        <span className="spatial-divider" />
        <button type="button" onClick={() => {
          setDraft(draft ? null : []);setError("");setFocal(undefined);
          setCamera(draft ? overviewCamera : {yaw:0,tilt:0,zoom:1});
        }} aria-pressed={draft !== null} title="Sketch a footprint"><PencilRuler size={16}/><span>Sketch</span></button>
        {zones.length > 0 && !draft && <button type="button" onClick={() => setShowVolumes(value => !value)}
          aria-pressed={showVolumes} aria-label="Show sketch volumes" title="Show sketch volumes"><Layers3 size={16}/></button>}
      </div>
      {draft ? <div className="spatial-subpanel">
        <span role="status">{error || (draft.length ? `${draft.length} corners · click the next corner` : "Click the corners of a footprint")}</span>
        <button type="button" onClick={() => setDraft(draft.slice(0,-1))} disabled={!draft.length} title="Undo corner" aria-label="Undo corner"><Undo2 size={15}/></button>
        <button type="button" disabled={draft.length < 3} onClick={finishZone}>Raise sketch</button>
        <button type="button" onClick={() => {setDraft(null);setError("");overview();}} title="Cancel sketch" aria-label="Cancel sketch"><X size={15}/></button>
      </div> : zones.length > 0 && showVolumes && <div className="spatial-subpanel">
        <Box size={15}/><label title="Illustrative depth for all sketched volumes">Illustrative depth
          <input type="range" min="0" max="100" value={depth} onChange={e => setDepth(Number(e.target.value))} />
        </label>
        {rooms.length > 0 && <button type="button" aria-label="Remove last sketch" title="Remove last sketch" onClick={() => setRooms(current => current.slice(0,-1))}><Undo2 size={15}/></button>}
      </div>}
    </div>
  </div>;
}
