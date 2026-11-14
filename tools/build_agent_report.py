"""Build a local comparison viewer with source-image sidecars and actual traces."""

import argparse
import json
from pathlib import Path


def build(directory):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    for case in report["cases"]:
        trace = directory / case["trace"]
        if not trace.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Trace outside report directory")
        case["details"] = json.loads(trace.read_text(encoding="utf-8"))
    payload = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    page = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Planproof — question routing comparison</title>
<style>body{font:16px system-ui;margin:0;background:#f4f6f5;color:#18362c}main{max-width:1400px;margin:32px auto;padding:16px}h1{margin:0}select,button{font:inherit;padding:8px}header,p{line-height:1.5}.layout{display:grid;grid-template-columns:1fr 1fr;gap:22px}.drawing{position:sticky;top:12px;align-self:start;background:white;position:relative}.drawing img{width:100%;display:block}.drawing svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.card{background:white;padding:18px;margin-bottom:14px;border:1px solid #ccd9d2;border-radius:10px}.card p{white-space:pre-line}.metrics{white-space:pre-wrap}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto}.good{color:#176535}.bad{color:#a13922}button{margin:4px}@media(max-width:800px){.layout{grid-template-columns:1fr}.drawing{position:relative}}</style>
<style>select{max-width:100%}table{border-collapse:collapse;margin:12px 0;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #ccd9d2}.filters{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}.drawing{position:sticky}.layout>*{min-width:0}@media(max-width:800px){.drawing{position:relative}}</style>
<main><header><h1>Question-aware agent comparison</h1><p>Same source and question; different tools. This is a development cohort, not a held-out accuracy estimate. Counts require exact matches; areas use the frozen 2% tolerance and compatible units. Citation bounds do not establish semantic correctness.</p><p id="summary" class="metrics"></p><p id="cohort"></p><details id="breakdown"><summary>Performance by category and source</summary></details><div class="filters"><label>Category <select id="group"><option value="all">All categories</option></select></label><label>Result <select id="outcome"><option value="all">All results</option><option value="auto_wrong">Agent incorrect or failed</option><option value="auto_only">Agent correct, direct incorrect</option><option value="direct_only">Direct correct, agent incorrect</option><option value="both_wrong">Both incorrect</option></select></label></div><label>Question <select id="case"></select></label><p id="caseMeta"></p><h2 id="question"></h2></header>
<div class="layout"><div class="drawing"><img id="drawing" alt="Original source drawing"><svg id="overlay"></svg></div><div id="cards"></div></div></main>
<script id="data" type="application/json">PAYLOAD</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const modes={direct:'Direct Gemini',visual_only:'Visual tools only',auto:'Agent chooses tools'};
const ids=[...new Set(data.cases.map(c=>c.id))];
const add=(parent,tag,text,cls)=>{const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;parent.append(e);return e;};
$('summary').textContent=Object.entries(data.summary).map(([m,s])=>`${modes[m]}: ${s.correct}/${s.attempted} correct; ${s.failures} execution failures; ${Math.round(s.seconds)}s summed request time${s.median_seconds!==undefined?`; median ${s.median_seconds.toFixed(1)}s`:''}`).join('\n');
$('cohort').textContent=data.binding.selection?`${ids.length} unique drawings. ${Object.entries(data.binding.selection.distribution.source).map(([s,n])=>`${n} ${s}`).join(', ')}. No drawings repeated from the eight-question Gemini pilot. ${data.binding.limitation}`:data.binding.limitation||'';
for(const field of ['category','source'])if(data.breakdown?.[field]){add($('breakdown'),'h3',field);const table=add($('breakdown'),'table',''),head=add(table,'tr','');add(head,'th','Group');for(const m of Object.keys(data.summary))add(head,'th',modes[m]);for(const [group,values] of Object.entries(data.breakdown[field])){const tr=add(table,'tr','');add(tr,'td',group);for(const m of Object.keys(data.summary)){const s=values[m];add(tr,'td',`${s.correct}/${s.attempted} correct; ${s.failures} failures`);}}}
for(const group of [...new Set(data.cases.map(c=>c.category).filter(Boolean))].sort()){const o=add($('group'),'option',group);o.value=group;}
function filter(){const previous=$('case').value;$('case').replaceChildren();for(const [i,id] of ids.entries()){const rows=data.cases.filter(c=>c.id===id),a=rows.find(c=>c.mode==='auto'),d=rows.find(c=>c.mode==='direct'),outcome=$('outcome').value;if($('group').value!=='all'&&rows[0].category!==$('group').value)continue;if(outcome==='auto_wrong'&&a?.score.correct)continue;if(outcome==='auto_only'&&!(a?.score.correct&&!d?.score.correct))continue;if(outcome==='direct_only'&&!(d?.score.correct&&!a?.score.correct))continue;if(outcome==='both_wrong'&&(a?.score.correct||d?.score.correct))continue;const o=add($('case'),'option',`${i+1}. ${rows[0].question}`);o.value=id;}if([...$('case').options].some(o=>o.value===previous))$('case').value=previous;show();}
function show(){const rows=data.cases.filter(c=>c.id===$('case').value);$('cards').replaceChildren();$('overlay').replaceChildren();$('drawing').hidden=!rows.length;if(!rows.length){$('question').textContent='No cases match these filters.';$('caseMeta').textContent='';return;}$('question').textContent=rows[0].question;$('caseMeta').textContent=[rows[0].category,rows[0].source,rows[0].language,rows[0].type].filter(Boolean).join(' / ');$('drawing').src=`./${rows[0].id}/page.png`;
 for(const c of rows){const card=add($('cards'),'section','', 'card');add(card,'h3',modes[c.mode]);add(card,'p',`${c.score.correct?'Correct under frozen evaluator':'Not correct under frozen evaluator'} · ${c.seconds}s · ${c.gemini_calls} Gemini calls · ${c.nemotron_calls} Nemotron calls`,c.score.correct?'good':'bad');
 add(card,'h2',c.answer.status==='answered'?`${c.answer.answer} ${c.answer.unit||''}`:'Abstained');add(card,'p',c.answer.explanation);add(card,'p','Coverage: '+c.answer.coverage);if(c.answer.uncertainties?.length)add(card,'p','Uncertainty: '+c.answer.uncertainties.join('; '));
 add(card,'p',c.tool_names.length?'Tools: '+c.tool_names.join(' → '):'No observation tools called.');
 add(card,'p',`Algorithms: ${c.algorithm_used?'used':'not requested/available'} · Recorded self-checks: ${c.reflection_calls||0} · Tool failures: ${c.tool_failures||0}`);
 if(c.gemini_reported_tokens!==undefined)add(card,'p',`Reported tokens: Gemini ${c.gemini_reported_tokens}, Nemotron ${c.nemotron_reported_tokens}. Failed requests may have unreported usage.`);
 for(const t of c.details.tool_trace.filter(t=>t.name==='reflect'&&t.result.status==='recorded')){const reflection=add(card,'details','');add(reflection,'summary','Recorded self-check (not independent verification)');add(reflection,'p','Draft: '+t.args.draft);add(reflection,'p','Checks: '+t.args.checks.join('; '));add(reflection,'p','Unresolved: '+(t.args.unresolved.join('; ')||'None reported'));add(reflection,'p','Next action: '+t.args.next_action);}
 const button=add(card,'button','Show visual citations');button.onclick=()=>{const [w,h]=c.details.images[0].source_render_size;$('overlay').setAttribute('viewBox',`0 0 ${w} ${h}`);$('overlay').replaceChildren();for(const citation of c.details.canonical_citations){const [x,y,r,b]=citation.canonical_bbox;const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');for(const [k,v] of Object.entries({x,y,width:r-x,height:b-y,fill:'#20a27e22',stroke:'#db4824','stroke-width':Math.max(w,h)/450}))rect.setAttribute(k,v);$('overlay').append(rect);}};
 const inspections=c.details.tool_trace.filter(t=>['inspect_nemotron','review_answer'].includes(t.name)&&t.result.status!=='unavailable');
 if(inspections.length){const regions=add(card,'details','');add(regions,'summary','Crops actually supplied to Nemotron');const container=add(regions,'div','');let loaded=false;regions.ontoggle=()=>{if(!regions.open||loaded)return;loaded=true;const source=new Image();source.onload=()=>{for(const tool of inspections){add(container,'p',tool.args.question||('Draft review: '+tool.args.candidate));for(const id of tool.args.image_ids){const meta=c.details.images.find(i=>i.image_id===id);if(!meta)continue;const [x,y,r,b]=meta.canonical_bbox,[w,h]=c.details.images[0].source_render_size;add(container,'p',`${id} · ${((r-x)*(b-y)/(w*h)*100).toFixed(1)}% of the original page`);const canvas=document.createElement('canvas');const scale=Math.min(1,360/(r-x));canvas.width=Math.max(1,Math.round((r-x)*scale));canvas.height=Math.max(1,Math.round((b-y)*scale));canvas.style.maxWidth='100%';canvas.getContext('2d').drawImage(source,x,y,r-x,b-y,0,0,canvas.width,canvas.height);container.append(canvas);}add(container,'pre',JSON.stringify(tool.result.advisory,null,2));}};source.src=`./${c.id}/page.png`;};}
 const details=add(card,'details','');add(details,'summary','Actual tool trace and result');add(details,'pre',JSON.stringify(c.details,null,2));
 }}
$('case').onchange=show;$('group').onchange=filter;$('outcome').onchange=filter;filter();
</script>""".replace("PAYLOAD", payload)
    (directory / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build(parser.parse_args().directory)
