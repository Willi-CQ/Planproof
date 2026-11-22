import { useEffect, useRef, useState } from "react";
import { ArrowUpRight, Box, Layers3 } from "lucide-react";
import { projectPlan, type PlanPoint } from "../lib/planProjection";

// Architectural study of the bundled 1120 × 740 sheet, in its source coordinates.
// Elevations, material and cutaway heights are illustrative.
const rectangle = (x: number, y: number, width: number, height: number): PlanPoint[] =>
  [[x,y],[x+width,y],[x+width,y+height],[x,y+height]];
const floor = rectangle(144,164,832,432);
const hall = rectangle(160,420,800,160);
const walls = [
  [144,164,832,16,116], [144,180,16,400,116], [960,180,16,400,48],
  [552,180,16,224,105], [160,404,118,16,66], [352,404,416,16,66],
  [842,404,118,16,66], [144,580,832,16,25],
];

function drawStudy(ctx: CanvasRenderingContext2D, separation: number) {
  const view = projectPlan(1120,740,[680,395],{yaw:26,tilt:56,zoom:1.25},[560,330]);
  function polygon(points: PlanPoint[], fill: string, stroke?: string) {
    ctx.beginPath();
    points.forEach(([x,y], i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
    ctx.closePath(); ctx.fillStyle=fill; ctx.fill();
    if (stroke) { ctx.strokeStyle=stroke; ctx.lineWidth=.85; ctx.stroke(); }
  }
  function line(a: PlanPoint,b: PlanPoint,color: string,width=1) {
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b);
    ctx.strokeStyle=color; ctx.lineWidth=width; ctx.stroke();
  }
  const surface = (points: PlanPoint[], z: number, fill: string, stroke?: string) =>
    polygon(points.map(p=>view.point(p,z)),fill,stroke);
  function block(points: PlanPoint[], bottom: number, top: number, colors: string[]) {
    // Positive yaw exposes the east and south faces.
    for (const side of [1,2]) {
      const a=points[side], b=points[(side+1)%4];
      polygon([view.point(a,bottom),view.point(b,bottom),view.point(b,top),view.point(a,top)],colors[side]);
    }
    surface(points,top,colors[0],"#334c454f");
  }
  const board=rectangle(74,105,972,554), gap=separation*105;
  ctx.save();
  ctx.shadowColor="#051610aa"; ctx.shadowBlur=16; ctx.shadowOffsetY=13;
  surface(board,-14,"#13362b"); ctx.restore();
  block(board,-14,0,["#cfdbcc","#8a9b87","#a4b49f"]);
  surface(rectangle(91,120,938,524),0,"#dce4d6","#9eae99");
  surface(floor,0,"#d6e2ce");
  surface(hall,0,"#bcd08d");
  for(const [x,y,w,h] of walls) surface(rectangle(x,y,w,h),0,"#49644f");
  function dimension(start: PlanPoint,end: PlanPoint,label:string) {
    const a=view.point(start), b=view.point(end);
    line(a,b,"#536f59",.8);
    for(const [x,y] of [a,b]) line([x-2,y+3],[x+2,y-3],"#536f59",1);
    ctx.save(); ctx.translate((a[0]+b[0])/2,(a[1]+b[1])/2);
    ctx.rotate(Math.atan2(b[1]-a[1],b[0]-a[0]));
    ctx.font="9px monospace"; ctx.textAlign="center";
    ctx.fillStyle="#dce4d6"; ctx.fillRect(-22,-7,44,14);
    ctx.fillStyle="#274832"; ctx.fillText(label,0,3); ctx.restore();
  }
  dimension([160,620],[960,620],"8.00 m");
  for(const p of floor) {
    ctx.setLineDash([3,4]); line(view.point(p,1),view.point(p,gap),"#cfe3b6"); ctx.setLineDash([]);
  }
  block(floor,gap,gap+8,["#e5eadc","#809079","#b1bfac"]);
  const z=gap+8;
  surface(hall,z,"#c5df92");
  // Contact shadows taper away from each rear wall into the room.
  for (const x of [160,568]) {
    for(let inset=0;inset<20;inset+=2) {
      const opacity=.13*(1-inset/20);
      surface(rectangle(x,180+inset,392,2),z,`rgba(44,64,48,${opacity})`);
      surface(rectangle(x+inset,180,2,224),z,`rgba(44,64,48,${opacity})`);
    }
  }
  for(let x=160;x<=960;x+=40) line(view.point([x,420],z),view.point([x,580],z),"#91ad6b",.55);
  for(let y=460;y<580;y+=40) line(view.point([160,y],z),view.point([960,y],z),"#91ad6b",.55);
  // Door leaves and swing arcs follow the two openings on the sample.
  for(const x of [278,768]) {
    line(view.point([x,404],z),view.point([x,331],z),"#738974",1);
    let last:PlanPoint=[x,331];
    for(let i=1;i<=16;i++){
      const t=i*Math.PI/32, next:PlanPoint=[x+73*Math.sin(t),404-73*Math.cos(t)];
      line(view.point(last,z),view.point(next,z),"#91a18b",.8); last=next;
    }
  }
  for(const [x,y,w,h,rise] of walls) {
    block(rectangle(x,y,w,h),z,z+rise,["#f5f5de","#81947e","#b4c2a8"]);
  }
  const anchor=view.point([835,508],z+3);
  const elbow:PlanPoint=[anchor[0]+22,anchor[1]-32];
  line(anchor,elbow,"#deefb8",1.3); line(elbow,[elbow[0]+12,elbow[1]],"#deefb8",1.3);
  ctx.fillStyle="#173c2a"; ctx.beginPath();ctx.arc(...anchor,3.5,0,Math.PI*2);ctx.fill();
  ctx.fillStyle="#d8eca5"; ctx.fillRect(elbow[0]+12,elbow[1]-12,78,24);
  ctx.fillStyle="#203c28"; ctx.font="bold 12px monospace"; ctx.textAlign="center";
  ctx.fillText("12.8 m²",elbow[0]+51,elbow[1]+4);
}

export function HallwayStudy({onOpen,disabled}: {onOpen?:()=>void;disabled:boolean}) {
  const canvas=useRef<HTMLCanvasElement>(null);
  const progress=useRef(1);
  const [exploded,setExploded]=useState(true);
  useEffect(()=>{
    const element=canvas.current;
    const ctx=element?.getContext("2d");
    if(!element || !ctx) return;
    const target=exploded ? 1 : 0, start=progress.current;
    const reduced=matchMedia("(prefers-reduced-motion: reduce)");
    let animation=0, started:number | undefined;
    function paint(){
      if(!element || !ctx) return;
      const width=element.clientWidth;
      if(!width) return;
      const pixels=Math.min(window.devicePixelRatio || 1,2);
      element.width=Math.round(width*pixels); element.height=Math.round(element.clientHeight*pixels);
      ctx.setTransform(element.width/680,0,0,element.height/395,0,0);
      drawStudy(ctx,progress.current);
    }
    function animate(time:number) {
      started??=time;
      const t=reduced.matches || start===target ? 1 : Math.min(1,(time-started)/420);
      progress.current=start+(target-start)*(1-(1-t)**3);
      paint();
      if(t<1) animation=requestAnimationFrame(animate);
    }
    const observer=new ResizeObserver(paint); observer.observe(element);
    animation=requestAnimationFrame(animate);
    return ()=>{observer.disconnect();cancelAnimationFrame(animation);};
  },[exploded]);
  return <figure className="hall-study">
    <header><figcaption>Hallway study</figcaption>
      <button type="button" className="study-toggle" aria-pressed={exploded} onClick={()=>setExploded(!exploded)}>
        <Layers3 size={15}/> Exploded view
      </button></header>
    <canvas ref={canvas} width="680" height="395" role="img"
      aria-label="Cutaway of the sample floor plan. Two rooms open onto an 8 by 1.6 metre hallway."/>
    <footer><span className="study-key"><i/>Hallway<span className="key-desktop"> footprint</span><span className="key-mobile"> · 12.8 m²</span></span>
      {onOpen && <button type="button" disabled={disabled} onClick={onOpen}><Box size={15}/> Explore the 3D sample <ArrowUpRight size={14}/></button>}
    </footer>
  </figure>;
}
