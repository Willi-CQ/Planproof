import { useState } from "react";
import { ArrowDownToLine, Check, ChevronLeft, Layers3, ListOrdered, Pencil, Ruler, Shapes, Trash2, Undo2 } from "lucide-react";
import type { Drawing } from "../lib/api";
import {
  calibrationFactor, decimalInput, quantity, samePoints, takeoffCsv, takeoffLabel, takeoffUnit, totalFor,
  type Calibration, type TakeoffItem, type TakeoffPoint, type TakeoffTool,
} from "../lib/takeoff";
import type { TakeoffControls } from "../lib/useTakeoff";
import "./takeoff.css";

const number = new Intl.NumberFormat(undefined,{maximumFractionDigits:2});
const euro = new Intl.NumberFormat(undefined,{style:"currency",currency:"EUR"});
const icons = {scale:Ruler,length:Ruler,area:Shapes,count:ListOrdered};

interface PanelProps {
  drawing:Drawing;
  filename:string;
  controls:TakeoffControls;
  suggested:{scale?:Calibration;outline?:{name:string;points:TakeoffPoint[]}};
  onShowPlan:()=>void;
  onClose:()=>void;
}

export function TakeoffPanel({drawing,filename,controls,suggested,onShowPlan,onClose}:PanelProps) {
  const {sheet,saving,error}=controls;
  const items=sheet?.items ?? [];
  const scale=sheet?.scale ?? null;
  const calibrated=calibrationFactor(scale,drawing)!==null;
  const costs=items.map(item=>totalFor(item,scale,drawing));
  const totals = (["length","area","count"] as const).map(kind => ({
    kind, values:items.filter(item=>item.kind===kind).map(item=>quantity(item,scale,drawing)),
  })).filter(row=>row.values.length);
  const priced=costs.filter(v=>v!==null);
  const [editing,setEditing]=useState<string|null>(null);
  function download() {
    if (!sheet?.items.length) return;
    const href=URL.createObjectURL(new Blob([takeoffCsv(sheet,drawing,drawing.page)],{type:"text/csv;charset=utf-8"}));
    const anchor=document.createElement("a");
    anchor.href=href;
    anchor.download=filename.replace(/\.[^/.]+$/,"")+"-page-"+drawing.page+"-takeoff.csv";
    anchor.click();
    setTimeout(()=>URL.revokeObjectURL(href),3000);
  }

  return <aside className="assistant-panel takeoff-panel" aria-label="Sheet takeoff">
    <header className="takeoff-header">
      <button type="button" className="takeoff-back" onClick={onClose}><ChevronLeft size={16}/> Back to chat</button>
      <div><h2>Takeoff</h2><span>Page {drawing.page}</span></div>
    </header>
    <div className="takeoff-content">
      <section className="takeoff-section" aria-label="Scale">
        <div className="takeoff-section-label"><Ruler size={16}/><h3>Scale</h3>
          <span className={calibrated ? "has-scale" : "missing-scale"}>{calibrated ? "Calibrated" : "Set a scale"}</span>
        </div>
        <p>{calibrated && scale
          ? `${number.format(scale.metres)} m reference · ${scale.source==="answer" ? "From a plan answer" : "Set on this sheet"}`
          : "Set two marks on a known distance to measure in m or m²."}</p>
        {suggested.scale && !scale && <button type="button" className="takeoff-button" disabled={saving||!sheet}
          onClick={()=>void controls.useScale(suggested.scale!)}><Check size={15}/> Use scale from answer</button>}
      </section>
      {!!totals.length && <div className="takeoff-summary" aria-label="Sheet totals">
        {totals.map(({kind,values}) => <div key={kind}>
          <span>{takeoffLabel[kind]}</span>
          <strong>{calibrated || kind==="count"
            ? number.format(values.reduce<number>((sum,n)=>sum+(n??0),0))+" "+takeoffUnit[kind]
            : "Needs scale"}</strong>
        </div>)}
      </div>}
      <section className="takeoff-section takeoff-list" aria-label="Quantities">
        <div className="takeoff-section-label"><Layers3 size={16}/><h3>Quantities</h3><span>{items.length}</span></div>
        {!items.length && <p>Choose Length, Area or Count above the drawing. Saved items appear here.</p>}
        {suggested.outline && !items.some(item=>item.source==="answer"&&samePoints(item.points,suggested.outline!.points)) &&
          <button type="button" className="takeoff-button" disabled={saving||!sheet}
            onClick={()=>void controls.addOutline(suggested.outline!,suggested.scale)}>Add outline from answer</button>}
        {items.map(item=><ItemRow key={item.id} item={item} amount={quantity(item,scale,drawing)} cost={totalFor(item,scale,drawing)}
          selected={controls.selected===item.id} editing={editing===item.id} disabled={saving}
          onSelect={()=>{controls.select(item.id);onShowPlan();}} onEdit={()=>setEditing(item.id)}
          onCancel={()=>setEditing(null)}
          onSave={async(name,rate)=>{if(await controls.edit(item,name,rate)) setEditing(null);}}
          onRemove={()=>void controls.remove(item)}/>)}
      </section>
      {!!items.length && <section className="takeoff-cost" aria-label="Estimate">
        <div><span>Estimate · priced items</span><strong>{priced.length ? euro.format(priced.reduce((sum,n)=>sum+n,0)) : "—"}</strong></div>
        <p>{priced.length} of {items.length} items have a quantity and unit price.</p>
      </section>}
      {error && !controls.draft && <p className="takeoff-error" role="alert">{error}</p>}
    </div>
    <footer className="takeoff-footer">
      <button type="button" className="takeoff-button" disabled={!items.length} onClick={download}>
        <ArrowDownToLine size={16}/> Export CSV
      </button>
      <span>Saved in this browser</span>
    </footer>
  </aside>;
}

interface ItemProps {
  item:TakeoffItem; amount:number|null; cost:number|null; selected:boolean; editing:boolean; disabled:boolean;
  onSelect:()=>void; onEdit:()=>void; onCancel:()=>void; onRemove:()=>void;
  onSave:(name:string,rate:number|null)=>Promise<void>;
}
function ItemRow(props:ItemProps) {
  const {item,amount,cost,selected,editing,disabled,onSelect,onEdit,onCancel,onRemove,onSave}=props;
  const Icon=icons[item.kind];
  return <div className="takeoff-row" data-selected={selected}>
    <div className="takeoff-row-summary">
      <button type="button" onClick={onSelect} className="takeoff-focus" aria-label={`Show ${item.name} on plan`}>
        <Icon size={17}/><span><strong>{item.name}</strong><small>{amount===null ? "Needs scale" : number.format(amount)+" "+takeoffUnit[item.kind]}
          {item.source==="answer" ? " · From answer" : ""}</small></span>
      </button>
      {cost!==null && <span className="takeoff-row-price">{euro.format(cost)}</span>}
      {!editing && <button type="button" className="takeoff-icon" onClick={onEdit} title="Edit item" aria-label={`Edit ${item.name}`}><Pencil size={15}/></button>}
    </div>
    {editing && <ItemEditor item={item} disabled={disabled} onSave={onSave} onCancel={onCancel} onRemove={onRemove}/>}
  </div>;
}
function ItemEditor({item,disabled,onSave,onCancel,onRemove}:Pick<ItemProps,"item"|"disabled"|"onSave"|"onCancel"|"onRemove">) {
  const [error,setError]=useState("");
  return <form className="takeoff-item-editor" onSubmit={event=>{
    event.preventDefault();
    const data=new FormData(event.currentTarget);
    const name=String(data.get("name")??"").trim();
    const raw=String(data.get("rate")??"").trim();
    const rate=raw ? decimalInput(raw) : null;
    if (!name || (rate!==null && (!Number.isFinite(rate)||rate<0||rate>1e9))) {
      setError("Enter a name and a valid positive or zero unit price.");return;
    }
    setError("");
    void onSave(name,rate);
  }}>
    <label>Item name<input name="name" defaultValue={item.name} maxLength={80} required/></label>
    <label>Price per {takeoffUnit[item.kind]} (€)<input name="rate" defaultValue={item.rate??""} inputMode="decimal" placeholder="Optional"/></label>
    <div className="takeoff-editor-actions">
      <button type="button" className="takeoff-icon" disabled={disabled} aria-label="Remove item" title="Remove item" onClick={onRemove}><Trash2 size={16}/></button>
      <button type="button" className="takeoff-button" onClick={onCancel}>Cancel</button>
      <button type="submit" className="takeoff-button primary" disabled={disabled}>Save item</button>
    </div>
    {error && <p className="takeoff-error" role="alert">{error}</p>}
  </form>;
}

export function TakeoffTools({controls,onClose}:{controls:TakeoffControls;onClose:()=>void}) {
  const {draft,sheet,saving,error}=controls;
  const [distance,setDistance]=useState("");
  const [unit,setUnit]=useState("m");
  const count=draft?.points.length ?? 0;
  const sufficient=draft && count >= (draft.kind==="count" ? 1 : draft.kind==="area" ? 3 : 2);
  return <div className="takeoff-tools">
    <div className="takeoff-tools-row" aria-label="Measurement tools">
      {(["scale","length","area","count"] as const).map(kind=>{
        const Icon=icons[kind];
        return <button key={kind} type="button" disabled={!sheet||saving} aria-pressed={draft?.kind===kind}
          onClick={()=>{setDistance("");controls.start(kind);}}><Icon size={15}/><span>{takeoffLabel[kind]}</span></button>;
      })}
      <button type="button" onClick={onClose} className="takeoff-tools-done" aria-label="Close takeoff"><Check size={17}/><span>Done</span></button>
    </div>
    {draft && <div className="takeoff-draft-tools">
      <p role="status">{error || (draft.kind==="scale" ? count<2 ? "Mark both ends of a known distance." : "Enter the real distance between the marks."
        : draft.kind==="area" ? "Trace the boundary in order." : draft.kind==="length" ? "Mark the ends and any bends." : "Place one marker per item.")}</p>
      <div className="takeoff-draft-actions">
        {draft.kind==="scale" && count===2 && <>
          <input aria-label="Reference distance" inputMode="decimal" placeholder="Distance" value={distance}
            onChange={event=>setDistance(event.target.value)} onKeyDown={event=>{
              if(event.key==="Enter"){event.preventDefault();void controls.finish(distance,unit);}
            }}/>
          <select aria-label="Distance unit" value={unit} onChange={event=>setUnit(event.target.value)}>
            {["m","cm","mm","ft","in"].map(u=><option key={u}>{u}</option>)}
          </select>
        </>}
        <button type="button" onClick={controls.undo} disabled={!count||saving} aria-label="Undo last measurement point" title="Undo point"><Undo2 size={15}/></button>
        <button type="button" onClick={controls.cancel} disabled={saving}>Cancel</button>
        <button type="button" className="primary" disabled={!sufficient||saving} onClick={()=>void controls.finish(distance,unit)}>
          {draft.kind==="scale" ? "Save scale" : "Save measurement"}
        </button>
      </div>
    </div>}
  </div>;
}

export function TakeoffMarks({drawing,controls}:{drawing:Drawing;controls:TakeoffControls}) {
  const {width,height}=drawing;
  const dots = width*.0034;
  const toSvg = (points:TakeoffPoint[])=>points.map(([x,y])=>[x*width,y*height]);
  function mark(kind:TakeoffTool,points:TakeoffPoint[],active=false) {
    const px=toSvg(points);
    const path=px.map(p=>p.join(",")).join(" ");
    return <g className="takeoff-shape" data-kind={kind} data-selected={active}>
      {kind!=="count" && (kind==="area" && px.length>=3
        ? <polygon points={path}/> : <polyline points={path}/>)}
      {px.map(([x,y],i)=><circle key={i} cx={x} cy={y} r={dots*(active ? 1.2 : .88)}/>)}
    </g>;
  }
  return <svg className="evidence-overlay takeoff-marks" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Measurement marks">
    {controls.sheet?.scale && mark("scale",controls.sheet.scale.points)}
    {controls.sheet?.items.map(item=><g key={item.id} data-takeoff-item={item.id}><title>{item.name}</title>
      {mark(item.kind,item.points,item.id===controls.selected)}</g>)}
    {controls.draft && mark(controls.draft.kind,controls.draft.points,true)}
  </svg>;
}
