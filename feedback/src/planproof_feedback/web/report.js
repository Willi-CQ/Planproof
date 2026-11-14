/* Offline only: no network calls, external libraries or browser-only storage dependency. */
"use strict";
const DATA = JSON.parse(document.getElementById("report-data").textContent);
const $ = id => document.getElementById(id);
const svgNS = "http://www.w3.org/2000/svg";
let selected = DATA.cases[0]?.id, boxes = true;
const byId = new Map(DATA.cases.map((c, i) => [c.id, {...c, number:i+1}]));
const number = n => Number.isFinite(n) ? n.toLocaleString(undefined,{maximumFractionDigits:1}) : "—";
const pct = n => Number.isFinite(n) ? (100*n).toFixed(1)+"%" : "—";
const unit = u => ({m2:"m²",m3:"m³",ft2:"ft²",count:""}[u] ?? u ?? "");
const answerText = a => !a || a.status === "abstain" ? "Abstained" : a.answer == null ? "No answer" : `${a.answer} ${unit(a.unit)}`.trim();
function node(tag, text, className) { const n=document.createElement(tag); if(text!=null)n.textContent=text; if(className)n.className=className; return n; }
function text(id,value){$(id).textContent=value;}
function message(value,error=false){text("message",value);$("message").classList.toggle("error-message",error);}
function current(which){return DATA.runs.find(r=>r.id===$("run-"+which).value)||null;}
function records(run){return new Map((run?.records||[]).map(r=>[r.id,r]));}
function correct(r){return !!r && r.score?.correct===true;}
function outcome(a,b){if(!a||!b){const recorded=a||b;if(!recorded)return "pending";if(recorded.status!=="ok")return "error";return correct(recorded)?"solo-correct":"solo-wrong";} if(a.status!=="ok"||b.status!=="ok")return "error";return correct(a)?(correct(b)?"both":"a-only"):(correct(b)?"b-only":"neither");}
function stats(rows){const scorable=rows.filter(r=>r.score?.scorable!==false);return {n:rows.length, correct:scorable.filter(correct).length, denominator:scorable.length, accuracy:scorable.length?scorable.filter(correct).length/scorable.length:null};}
function fillSelect(id,values,label){const s=$(id);values.forEach(value=>{const option=node("option",label?label(value):value);option.value=value;s.append(option);});}
function populateRuns(preserve=false){
  const oldA=$("run-a").value,oldB=$("run-b").value;
  for(const key of ["a","b"]){const s=$("run-"+key);s.replaceChildren();const blank=node("option",key==="a"?"Nemotron · not run yet":"Gemini Carbon · not run yet");blank.value="";s.append(blank);
    for(const run of DATA.runs){const option=node("option",`${run.label} · ${run.mode} · ${run.status} · ${run.id.split(":")[0]}`);option.value=run.id;s.append(option);}}
  $("run-a").value=preserve&&DATA.runs.some(r=>r.id===oldA)?oldA:(DATA.runs.find(r=>/nemotron/i.test(r.label)&&r.mode==="direct")?.id||DATA.runs[0]?.id||"");
  $("run-b").value=preserve&&DATA.runs.some(r=>r.id===oldB)?oldB:(DATA.runs.find(r=>/gemini|carbon/i.test(r.label))?.id||"");
}
function summary(){
  const a=current("a"),b=current("b"),am=records(a),bm=records(b);
  const paired=DATA.cases.filter(c=>am.has(c.id)&&bm.has(c.id));
  const as=stats(a?.records||[]),bs=stats(b?.records||[]);
  text("state",DATA.runs.length?`${DATA.runs.length} saved run${DATA.runs.length===1?"":"s"}`:"Inputs ready · not run");
  text("paired-n",a&&b?paired.length:"—");text("paired-note",a&&b?`${paired.length} / ${DATA.cases.length} matched source hashes`:"Waiting for two runs");
  for(const [key,run,s] of [["a",a,as],["b",b,bs]]){text(key+"-title",`${run?.label||(key==="a"?"Nemotron":"Gemini Carbon")} accuracy`);text(key+"-accuracy",pct(s.accuracy));text(key+"-sub",run?`${s.correct} correct / ${s.denominator} completed · ${DATA.cases.length-s.n} pending`:"Not run · no score yet");}
  const pa=stats(paired.map(c=>am.get(c.id))),pb=stats(paired.map(c=>bm.get(c.id)));
  const delta=paired.length?(pb.accuracy-pa.accuracy)*100:null;
  text("delta",delta==null?"—":`${delta>0?"+":""}${delta.toFixed(1)} pp`);
  text("delta-sub",paired.length?`On the same ${paired.length} completed questions`:"Only matching question IDs count");
  if(a&&b&&a.id===b.id)message("The same run is selected twice. Choose a second run for a model comparison.");
  else if(a&&b&&(a.image_policy_matches===false||b.image_policy_matches===false))message("Same source drawings and questions; image policies differ. Interpret this as a workflow comparison.");
  else message(DATA.runs.length?"Scores come from the frozen reference evaluator. Source evidence and all run data stay local.":"No accuracy results yet. Paste the Verda key and run the benchmark; this view will populate automatically.");
  typeChart(a,b);outcomes(paired,am,bm);efficiency(a,b);grid();detail();
}
function typeChart(a,b){
  const root=$("accuracy-chart");root.replaceChildren();
  for(const type of [...new Set(DATA.cases.map(c=>c.type))]){const row=node("div",null,"type-row"),label=node("div",type,"type-title"),bars=node("div");
    for(const [key,run] of [["a",a],["b",b]]){const s=stats((run?.records||[]).filter(r=>r.type===type));const line=node("div",null,"bar-row");line.append(node("span",key.toUpperCase()));const track=node("div",null,"track"),fill=node("div",null,"fill "+key);fill.style.width=(s.accuracy==null?0:100*s.accuracy)+"%";track.append(fill);line.append(track,node("span",s.n?`${pct(s.accuracy)} · n=${s.denominator}`:"Not run","bar-value"));bars.append(line);}row.append(label,bars);root.append(row);}
  const axis=node("div",null,"bar-axis");axis.append(node("span","0%"),node("span","50%"),node("span","100%"));root.append(axis);
  text("type-note",Object.entries(DATA.dataset.population.by_type).map(([k,v])=>`${v} ${k}`).join(" · ")+" in this random draw. Unrepresented types have no score.");
}
function outcomes(paired,am,bm){const root=$("outcomes");root.replaceChildren();const counts={both:0,"a-only":0,"b-only":0,neither:0};
  for(const c of paired){const a=correct(am.get(c.id)),b=correct(bm.get(c.id));counts[a?(b?"both":"a-only"):(b?"b-only":"neither")]++;}
  const stack=node("div",null,"outcome-stack"+(paired.length?"":" empty"));if(!paired.length)stack.textContent="Paired outcomes appear after both runs";
  for(const [kind,value]of Object.entries(counts)){if(!value)continue;const mark=node("div",String(value),"outcome-segment "+kind);mark.style.width=(100*value/paired.length)+"%";mark.title=`${kind}: ${value} questions`;stack.append(mark);}root.append(stack);
  const totals=node("div",null,"outcome-counts");for(const [kind,label]of [["both","Both correct"],["a-only","Only A correct"],["b-only","Only B correct"],["neither","Neither correct"]]){const line=node("div");line.append(node("span",label),node("strong",paired.length?String(counts[kind]):"—"));totals.append(line);}root.append(totals);
}
function efficiency(a,b){const root=$("efficiency-rows");root.replaceChildren();for(const [name,run]of [["A · Nemotron",a],["B · Gemini Carbon",b]]){const rows=run?.records||[],times=rows.map(r=>r.latency_seconds).filter(Number.isFinite).sort((x,y)=>x-y),median=times.length?(times[Math.floor((times.length-1)/2)]+times[Math.floor(times.length/2)])/2:null,missing=rows.filter(r=>!Number.isFinite(r.usage?.total_tokens)).length,total=rows.reduce((n,r)=>n+(r.usage?.total_tokens||0),0);const values=[run?run.label+" · "+run.mode:name,run?`${rows.length}/${DATA.cases.length}`:"Not run",median==null?"—":number(median)+" s",times.length?number(times.reduce((x,y)=>x+y,0))+" s":"—",run&&rows.length&&missing<rows.length?number(total)+(missing?" (partial)":""):"—",run?`${missing}/${rows.length}`:"—"];const tr=node("tr");values.forEach(v=>tr.append(node("td",v)));root.append(tr);}}
function filteredCases(){const a=records(current("a")),b=records(current("b")),query=$("search").value.toLocaleLowerCase().trim();return DATA.cases.filter(c=>($("type-filter").value==="all"||c.type===$("type-filter").value)&&($("source-filter").value==="all"||c.source===$("source-filter").value)&&($("outcome-filter").value==="all"||($("outcome-filter").value==="pending"?(!a.has(c.id)||!b.has(c.id)):outcome(a.get(c.id),b.get(c.id))===$("outcome-filter").value))&&(!query||c.question.toLocaleLowerCase().includes(query)||c.id.includes(query)));}
function grid(){const root=$("question-grid"),a=records(current("a")),b=records(current("b")),visible=filteredCases();root.replaceChildren();for(const c of visible){const info=byId.get(c.id),state=outcome(a.get(c.id),b.get(c.id));const cell=node("button",String(info.number).padStart(2,"0"),"question-cell "+state+(selected===c.id?" selected":""));cell.type="button";cell.title=`Q${info.number} · ${c.type} · ${c.source}\n${c.question}`;cell.setAttribute("aria-label",`Question ${info.number}, ${state}: ${c.question}`);cell.setAttribute("aria-pressed",String(selected===c.id));cell.addEventListener("click",()=>{selected=c.id;$("page-select").value="0";grid();detail();});root.append(cell);}text("visible-count",`${visible.length} / ${DATA.cases.length} questions`);if(!visible.length)root.append(node("p","No questions match these filters.","chart-note"));}
function answerPanel(key,run,record){const root=$("answer-"+key);root.replaceChildren();const heading=node("div",null,"answer-heading"),label=node("span",`${key.toUpperCase()} · ${run?.label||(key==="a"?"Nemotron":"Gemini Carbon")}`,"answer-name"),status=node("span",!record?"Awaiting result":record.status!=="ok"?record.status:record.answer?.status==="abstain"?"Abstained":correct(record)?"Correct":"Incorrect","answer-status");heading.append(label,status);root.append(heading,node("div",record?answerText(record.answer):"—","answer-value"));if(!record){root.append(node("p","No answer has been recorded for this question.","chart-note"));return;}
  root.append(node("p",record.answer?.explanation||record.error||record.score?.reason||"No explanation supplied.","answer-explanation"));const meta=node("div",null,"answer-meta");meta.append(node("span",Number.isFinite(record.latency_seconds)?number(record.latency_seconds)+" s":"Timing unavailable"),node("span",Number.isFinite(record.usage?.total_tokens)?number(record.usage.total_tokens)+" tokens":"Token usage unavailable"),node("span",`${record.answer?.citations?.length||0} proposed evidence boxes`));root.append(meta);
  const trace=record.trace?.calls||[];if(trace.length){const stages=node("details"),sum=node("summary",`${trace.length} preserved stages · ${record.trace.stop_reason||"complete"}`);stages.append(sum);trace.forEach((call,index)=>{stages.append(node("p",`${index+1}. ${call.role} · ${call.status}`),node("pre",JSON.stringify(call.result||call.validation_error||call.completion?.content,null,2)));});root.append(stages);}
  const raw=node("details");raw.append(node("summary","Inspect the complete saved record"),node("pre",JSON.stringify(record,null,2)));root.append(raw);
}
function detail(){const c=byId.get(selected);if(!c)return;const a=current("a"),b=current("b"),ar=records(a).get(c.id),br=records(b).get(c.id);text("case-label",`QUESTION ${String(c.number).padStart(2,"0")} · ${c.type.toUpperCase()} · ${c.id.slice(0,12)}`);text("question-text",c.question);text("reference-answer",`${c.reference} ${unit(c.unit)}`.trim());text("case-source",`${c.source} · source SHA256 ${c.input_sha256.slice(0,16)}…`);answerPanel("a",a,ar);answerPanel("b",b,br);
  const page=$("page-select"),previous=page.value;page.replaceChildren();c.images.forEach((image,index)=>{const option=node("option",String(image.page_index+1));option.value=String(index);page.append(option);});page.value=String(Math.min(Number(previous)||0,c.images.length-1));const image=c.images[Number(page.value)||0];$("drawing-image").src=image.asset_path;$("drawing-image").alt=`Source drawing for question ${c.number}: ${c.question}`;$("original-link").href=c.original_path;const overlay=$("drawing-boxes");overlay.replaceChildren();overlay.setAttribute("viewBox",`0 0 ${image.sent_size[0]} ${image.sent_size[1]}`);overlay.style.display=boxes?"block":"none";let omitted=0;
  for(const [key,record]of [["a",ar],["b",br]]){for(const citation of record?.answer?.citations||[]){const supplied=(record?.trace?.final?.images||record?.images||[]).find(i=>i.image_id===citation.image_id);if(!supplied||supplied.sent_sha256!==image.sent_sha256){omitted++;continue;}const box=citation.bbox;if(!Array.isArray(box)||box.length!==4||!box.every(Number.isFinite)||box[0]<0||box[1]<0||box[2]>image.sent_size[0]||box[3]>image.sent_size[1]||box[2]<=box[0]||box[3]<=box[1])continue;const rect=document.createElementNS(svgNS,"rect");rect.setAttribute("class",key);for(const [attr,value]of Object.entries({x:box[0],y:box[1],width:box[2]-box[0],height:box[3]-box[1]}))rect.setAttribute(attr,String(value));overlay.append(rect);}}
  text("image-note",`${image.sent_size.join(" × ")} px · exact model-input preview. Solid green = A; dashed amber = B. Boxes indicate proposed locations, not verified correctness.${omitted?" Some boxes use a different or unknown image frame and are not overlaid.":""}`);text("source-credit",c.credit||"Source attribution is preserved in the linked license notice.");$("license-link").href=c.license_notice||c.original_path;
}
function move(direction){const visible=filteredCases();if(!visible.length)return;const index=visible.findIndex(c=>c.id===selected);selected=visible[(Math.max(index,0)+direction+visible.length)%visible.length].id;$("page-select").value="0";grid();detail();}
function save(){const blob=new Blob([JSON.stringify(DATA,null,2)],{type:"application/json"}),url=URL.createObjectURL(blob),a=node("a");a.href=url;a.download="planproof-comparison-"+new Date().toISOString().replaceAll(":","-")+".json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message("Comparison snapshot exported. Original run files and verified ZIP backups remain on disk.");}
function addReport(value,name){let runs;if(value.schema==="planproof-comparison-v1"){if(value.dataset?.sha256!==DATA.dataset.sha256)throw Error("Different question/input set; comparison refused.");runs=value.runs;}else if(value.binding?.manifest_sha256===DATA.dataset.manifest_sha256&&Array.isArray(value.cases)){const groups=new Map();for(const row of value.cases){if(!groups.has(row.mode))groups.set(row.mode,[]);groups.get(row.mode).push(row);}runs=[...groups.entries()].map(([mode,rows])=>({id:name+":"+mode,label:value.label||Object.values(value.binding.providers||{})[0]?.model||name,model:Object.values(value.binding.providers||{})[0]?.model||"unknown",mode,status:rows.length===DATA.cases.length?"complete":"partial",records:rows,binding:value.binding,image_policy_matches:JSON.stringify(value.binding.images)===JSON.stringify(DATA.dataset.image_policy)}));}else throw Error("Load a saved report.json or comparison snapshot. For Gemini predictions use benchmark-import first.");
  for(const run of runs){const seen=new Set();for(const row of run.records||[]){const c=byId.get(row.id);if(!c||seen.has(row.id)||row.input_sha256&&row.input_sha256!==c.input_sha256||typeof row.score?.correct!=="boolean"&&row.score?.correct!==null)throw Error("Unknown, duplicate, unscored or mismatched case; comparison refused.");seen.add(row.id);}if(!Array.isArray(run.records)||!run.records.length)throw Error("Saved run has no records.");if(DATA.runs.some(r=>r.id===run.id))throw Error("This run is already loaded; existing records were not replaced.");}
  DATA.runs.push(...runs);populateRuns(true);if(runs[0])$("run-b").value=runs[0].id;summary();message(`Loaded ${runs.length} saved run(s) locally. Save a comparison snapshot to retain this view.`);
}
fillSelect("type-filter",[...new Set(DATA.cases.map(c=>c.type))]);fillSelect("source-filter",[...new Set(DATA.cases.map(c=>c.source))]);
text("cohort",`${DATA.cases.length} questions · seed ${DATA.dataset.population.seed}`);text("dataset-hash","DATASET "+DATA.dataset.sha256.slice(0,16));text("created","Generated "+new Date(DATA.created_utc).toLocaleString());
$("method").append(node("p",DATA.dataset.limitations.join(" ")));
for(const id of ["run-a","run-b"])$(id).addEventListener("change",summary);
for(const id of ["type-filter","source-filter","outcome-filter"])$(id).addEventListener("change",grid);$("search").addEventListener("input",grid);
$("page-select").addEventListener("change",detail);$("previous").addEventListener("click",()=>move(-1));$("next").addEventListener("click",()=>move(1));
$("toggle-boxes").addEventListener("click",()=>{boxes=!boxes;$("toggle-boxes").setAttribute("aria-pressed",String(boxes));text("toggle-boxes",boxes?"Evidence boxes on":"Evidence boxes off");detail();});
$("import-button").addEventListener("click",()=>$("import-file").click());$("export").addEventListener("click",save);$("import-file").addEventListener("change",async event=>{try{const file=event.target.files[0];if(file)addReport(JSON.parse(await file.text()),file.name);}catch(error){message(error.message,true);}finally{event.target.value="";}});
populateRuns();summary();
