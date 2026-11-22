import { simpleFootprint, type PlanPoint } from "./planProjection";
import { visibleCitations, type AgentResult, type Drawing } from "./api";

export type TakeoffKind = "length" | "area" | "count";
export type TakeoffTool = "scale" | TakeoffKind;
export type TakeoffPoint = PlanPoint;
export type TakeoffDraft = {kind:TakeoffTool; points:TakeoffPoint[]};
export type Calibration = {points:[TakeoffPoint, TakeoffPoint]; metres:number; source:"manual" | "answer"};
export type TakeoffItem = {
  id:string; name:string; kind:TakeoffKind; points:TakeoffPoint[]; rate:number | null; source:"manual" | "answer";
};
export type SheetTakeoff = {id:string; scale:Calibration | null; items:TakeoffItem[]};
type Frame = Pick<Drawing,"width"|"height">;
export const takeoffUnit = {length:"m",area:"m²",count:"pcs"};
export const takeoffLabel = {length:"Length",area:"Area",count:"Count",scale:"Scale"};

export const emptyTakeoff = (id: string): SheetTakeoff => ({id, scale:null, items:[]});
const sourcePoint = ([x,y]: TakeoffPoint, frame:Frame): PlanPoint => [x*frame.width,y*frame.height];
const arePoints = (p:TakeoffPoint[]) => p.every(([x,y]) => Number.isFinite(x) && Number.isFinite(y) && x>=0 && x<=1 && y>=0 && y<=1);

function pathLength(points: PlanPoint[]) {
  return points.slice(1).reduce((sum,p,i) => sum+Math.hypot(p[0]-points[i][0],p[1]-points[i][1]),0);
}

export function calibrationFactor(scale:Calibration|null, frame:Frame) {
  if (!scale || !Number.isFinite(scale.metres) || scale.metres <= 0 || !arePoints(scale.points)) return null;
  const length=pathLength(scale.points.map(p=>sourcePoint(p,frame)));
  return length >= 3 ? scale.metres / length : null;
}

function ring(points:TakeoffPoint[]) {
  const first=points[0], last=points.at(-1);
  return points.length>2 && last && Math.hypot(first[0]-last[0],first[1]-last[1])<1e-9 ? points.slice(0,-1) : points;
}

export function geometryError(draft:TakeoffDraft, frame:Frame) {
  if (!arePoints(draft.points)) return "Choose points inside this sheet.";
  if (draft.kind==="count") return draft.points.length ? "" : "Place at least one marker.";
  const p=(draft.kind==="area" ? ring(draft.points) : draft.points).map(point=>sourcePoint(point,frame));
  if (draft.kind==="area") return simpleFootprint(p) ? "" : "Draw at least three corners with edges that do not cross.";
  return p.length>=2 && pathLength(p)>=3 ? "" : "Choose two points at least three drawing units apart.";
}

export function quantity(item:TakeoffItem, scale:Calibration|null, frame:Frame):number|null {
  if (item.kind==="count") return item.points.length;
  const k=calibrationFactor(scale,frame);
  if (k===null) return null;
  const points=(item.kind==="area" ? ring(item.points) : item.points).map(p=>sourcePoint(p,frame));
  if (item.kind==="length") return pathLength(points)*k;
  const doubleArea=points.reduce((sum,[x,y],i) => {
    const [u,v]=points[(i+1)%points.length];
    return sum+x*v-u*y;
  },0);
  return Math.abs(doubleArea)*.5*k*k;
}

export function takeoffBounds(points:TakeoffPoint[], frame:Frame) {
  const src=points.map(p=>sourcePoint(p,frame));
  const xs=src.map(p=>p[0]), ys=src.map(p=>p[1]);
  return [Math.max(0,Math.min(...xs)-4), Math.max(0,Math.min(...ys)-4),
          Math.min(frame.width,Math.max(...xs)+4), Math.min(frame.height,Math.max(...ys)+4)];
}

export function decimalInput(value:string) {
  const trimmed=value.trim();
  return /^\d+(?:[.,]\d+)?$/.test(trimmed) ? Number(trimmed.replace(",",".")) : NaN;
}

export function makeItem(kind:TakeoffKind, points:TakeoffPoint[], name:string, source:TakeoffItem["source"]="manual"):TakeoffItem {
  return {id:crypto.randomUUID(),kind,points:kind==="area" ? ring(points) : points,name,rate:null,source};
}

export function readerTakeoff(result:AgentResult|undefined,drawing:Drawing) {
  const found:{scale?:Calibration; outline?:{name:string;points:TakeoffPoint[]}}={};
  if (!result) return found;
  const normalize = (p:number[]):TakeoffPoint => [p[0]/drawing.width,p[1]/drawing.height];
  for (const c of visibleCitations(result,drawing)) {
    const detail=result.answer?.citation_details?.find(d=>d.citation===c.source_index);
    if (!detail || detail.kind!=="measurement") continue;
    const scale=/^graphic scale · (\d+(?:[.,]\d+)?) (m|ft)$/i.exec(detail.label);
    if (scale && c.line?.length===2) {
      const points = c.line.map(normalize) as Calibration["points"];
      const proposed:Calibration={points,metres:decimalInput(scale[1])*(scale[2].toLowerCase()==="ft"?.3048:1),source:"answer"};
      if (calibrationFactor(proposed,drawing)!==null) found.scale=proposed;
    }
    if (detail.label.startsWith("room polygon · ") && c.polygon?.length) {
      const points=ring(c.polygon.map(normalize));
      if (!geometryError({kind:"area",points},drawing))
        found.outline={name:detail.label.slice("room polygon · ".length),points};
    }
  }
  return found;
}

export function samePoints(a:TakeoffPoint[],b:TakeoffPoint[]) {
  return a.length===b.length && a.every(([x,y],i)=>x===b[i][0] && y===b[i][1]);
}

export function totalFor(item:TakeoffItem,scale:Calibration|null,frame:Frame) {
  const amount=quantity(item,scale,frame);
  return item.rate===null || amount===null ? null : Math.round(amount*item.rate*100)/100;
}

export function takeoffCsv(sheet:SheetTakeoff,frame:Frame,page:number) {
  const csvCell=(v:string|number|null) => {
    let text=String(v??"");
    if (typeof v==="string" && /^[\s\uFEFF]*[=+\-@]/.test(text)) text="'"+text;
    return '"'+text.replaceAll('"','""')+'"';
  };
  const rows=[
    ["Page","Item","Type","Quantity","Unit","Unit price (EUR)","Total (EUR)","Geometry source",
      "Calibration span (m)","Calibration source"],
    ...sheet.items.map(i=>{
      const amount=quantity(i,sheet.scale,frame);
      return [page,i.name,takeoffLabel[i.kind],amount===null ? null : Number(amount.toFixed(4)),
        takeoffUnit[i.kind],i.rate,totalFor(i,sheet.scale,frame),i.source,
        i.kind==="count" ? null : sheet.scale?.metres??null,i.kind==="count" ? null : sheet.scale?.source??null];
    }),
  ];
  return "\uFEFF"+rows.map(row=>row.map(csvCell).join(",")).join("\r\n")+"\r\n";
}
