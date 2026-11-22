import { useEffect, useRef, useState } from "react";
import type { Drawing } from "./api";
import { readTakeoff, updateTakeoff } from "./library";
import {
  calibrationFactor, decimalInput, geometryError, makeItem, samePoints, takeoffLabel,
  type Calibration, type SheetTakeoff, type TakeoffDraft, type TakeoffItem, type TakeoffPoint, type TakeoffTool,
} from "./takeoff";

export function useTakeoff(id:string|null, drawing:Drawing|null) {
  const [record,setRecord]=useState<SheetTakeoff|null>(null);
  const [draft,setDraft]=useState<TakeoffDraft|null>(null);
  const [selected,setSelected]=useState<string|null>(null);
  const [error,setError]=useState("");
  const [saving,setSaving]=useState(false);
  const currentId=useRef(id);
  currentId.current=id;
  const inFlight=useRef(false);
  const sheet=record?.id===id ? record : null;

  useEffect(()=>{
    let active=true;
    setRecord(null);
    setDraft(null);
    setSelected(null);
    setError("");
    if (id) void readTakeoff(id).then(value=>{
      if (active) setRecord(value);
    }).catch(()=>{
      if (active) setError("The takeoff could not be opened in this browser.");
    });
    return ()=>{active=false;};
  },[id]);

  async function change(edit:(value:SheetTakeoff)=>SheetTakeoff) {
    if (!id || !sheet || inFlight.current) return false;
    inFlight.current=true;
    setSaving(true);
    setError("");
    try {
      const written=await updateTakeoff(id,edit);
      if (currentId.current===id) setRecord(written);
      return currentId.current===id;
    } catch {
      if (currentId.current===id) setError("The changes could not be saved. Try again before you close this drawing.");
      return false;
    } finally {
      inFlight.current=false;
      setSaving(false);
    }
  }

  function start(kind:TakeoffTool) {
    if (!sheet || saving) return;
    setError("");
    setSelected(null);
    setDraft({kind,points:[]});
  }

  function addPoint(point:TakeoffPoint) {
    if (!draft || !drawing || !sheet || saving) return;
    const near=draft.points.some(p=>Math.hypot((p[0]-point[0])*drawing.width,(p[1]-point[1])*drawing.height)<2);
    if (near) {
      setError("Choose a different point.");
      return;
    }
    const limit=draft.kind==="scale" ? 2 : 128;
    if (draft.points.length>=limit) return;
    setError("");
    setDraft({...draft,points:[...draft.points,point]});
  }

  async function finish(amount="",unit="m") {
    if (!draft || !drawing || !sheet) return;
    const problem=geometryError(draft,drawing);
    if (problem) {setError(problem);return;}
    if (draft.kind==="scale") {
      const factors:Record<string,number>={m:1,cm:.01,mm:.001,ft:.3048,in:.0254};
      const metres=decimalInput(amount)*factors[unit];
      const scale:Calibration={points:[draft.points[0],draft.points[1]],metres,source:"manual"};
      if (draft.points.length!==2 || metres>100000 || !calibrationFactor(scale,drawing)) {
        setError("Enter the positive length between the two marks.");
        return;
      }
      if (await change(current=>({...current,scale}))) setDraft(null);
      return;
    }
    const item=makeItem(draft.kind,draft.points,takeoffLabel[draft.kind]);
    if (await change(current=>{
      const used=new Set(current.items.map(i=>i.name));
      let next=1;
      while (used.has(item.name+" "+next)) next++;
      return {...current,items:[...current.items,{...item,name:item.name+" "+next}]};
    })) {
      setDraft(null);
      setSelected(item.id);
    }
  }

  function cancel() {setDraft(null);setError("");}
  function undo() {
    if (!draft?.points.length || saving) return;
    setError("");
    setDraft({...draft,points:draft.points.slice(0,-1)});
  }
  function select(itemId:string|null) {
    cancel();
    setSelected(itemId===selected ? null : itemId);
  }
  async function useScale(scale:Calibration) {
    if (drawing && calibrationFactor(scale,drawing) && await change(c=>({...c,scale}))) cancel();
  }
  async function addOutline(outline:{name:string;points:TakeoffPoint[]},scale?:Calibration) {
    if (!drawing || geometryError({kind:"area",points:outline.points},drawing)) return;
    const matched=sheet?.items.find(i=>i.source==="answer" && samePoints(i.points,outline.points));
    if (matched) {setSelected(matched.id);cancel();return;}
    const item=makeItem("area",outline.points,outline.name.slice(0,80),"answer");
    let savedId=item.id;
    if (await change(current=>{
      const existing=current.items.find(i=>i.source==="answer" && samePoints(i.points,outline.points));
      if (existing) {savedId=existing.id;return current;}
      return {...current,scale:current.scale ?? scale ?? null,items:[...current.items,item]};
    })) {
      setSelected(savedId);
      cancel();
    }
  }
  function edit(item:TakeoffItem, name:string, rate:number|null) {
    return change(c=>({...c,items:c.items.map(i=>i.id===item.id ? {...i,name,rate} : i)}));
  }
  async function remove(item:TakeoffItem) {
    if (await change(c=>({...c,items:c.items.filter(i=>i.id!==item.id)})) && selected===item.id) setSelected(null);
  }

  return {sheet,draft,selected,error,saving,start,addPoint,finish,cancel,undo,select,useScale,addOutline,edit,remove};
}

export type TakeoffControls = ReturnType<typeof useTakeoff>;
