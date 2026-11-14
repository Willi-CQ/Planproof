"""Build a local, source-first review of the completed matched pilot."""

import argparse
import json
from pathlib import Path


def build(root):
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    cases = []
    for row in report["cases"]:
        directory = root / row["id"]
        entry = {**row, "image": row["id"] + "/canonical.png"}
        control = directory / "direct-control.json"
        if control.exists():
            entry["control"] = json.loads(control.read_text(encoding="utf-8"))
        trace = directory / "guided-trace.json"
        if trace.exists():
            trace = json.loads(trace.read_text(encoding="utf-8"))
            entry["guided"] = trace["final"]
            entry["catalog"] = trace["algorithm_evidence"]
        cases.append(entry)
    data = json.dumps(
        {"cases": cases, "summary": report["summary"], "binding": report["binding"]},
        ensure_ascii=True,
    ).replace("<", "\\u003c")
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Planproof · Algorithm-guided Nemotron pilot</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f3;color:#17332f;font:14px/1.5 system-ui}header{background:#153a33;color:white;padding:22px 30px}h1{font-size:23px;margin:0}p{margin:7px 0}.muted{color:#657971}.wrap{padding:20px 30px;max-width:1800px;margin:auto}.metrics{display:flex;gap:12px;flex-wrap:wrap}.metric{background:white;border:1px solid #d6e1db;border-radius:8px;padding:15px;min-width:210px}.metric b{display:block;font-size:26px}.layout{display:grid;grid-template-columns:minmax(300px,2fr) minmax(280px,1fr);gap:18px;margin-top:20px}.panel{background:white;border:1px solid #d6e1db;border-radius:9px;padding:16px}select{padding:9px;max-width:100%;font:inherit}svg{width:100%;height:640px}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}label{font-size:12px}h2{font-size:17px}h3{font-size:13px;margin:15px 0 5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.5 monospace;max-height:350px;overflow:auto}.answer{padding:10px;background:#f3f7f4;border-radius:5px;margin:9px 0}.tag{font-size:11px;font-weight:bold}button{padding:6px;cursor:pointer}@media(max-width:850px){.layout{grid-template-columns:1fr}.wrap{padding:12px}svg{height:440px}}
</style><header><h1>Planproof · Algorithm-guided Nemotron</h1><p>Matched pilot: same questions and source pixels · original and controlled direct baselines · explicit failures</p></header>
<main class="wrap"><div id="metrics" class="metrics"></div><p class="muted">Small development pilot. Fresh direct and guided calls use identical generation settings; the old benchmark uses its original settings. These results do not establish general Stoa QA accuracy.</p>
<div class="layout"><section class="panel"><select id="case" aria-label="Choose pilot question"></select><h2 id="question"></h2><div class="controls"><label><input id="rooms" type="checkbox"> ML room proposals</label><label><input id="citations" type="checkbox" checked> Nemotron source citations</label></div><svg id="drawing" xmlns="http://www.w3.org/2000/svg" aria-label="Original drawing and evidence"></svg></section>
<aside class="panel"><p id="meta" class="muted"></p><div id="answers"></div><h3>Nemotron routing decision</h3><pre id="decision"></pre><h3>Evidence and validation</h3><pre id="evidence"></pre></aside></div></main><script>
const DATA=__DATA__,$=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
function add(name,attrs,parent=$('drawing')){const e=document.createElementNS(NS,name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);parent.append(e);return e;}
function box(b,color){add('rect',{x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1],fill:'none',stroke:color,'stroke-width':2,'vector-effect':'non-scaling-stroke'});}
for(const[label,key]of[['Old direct baseline','direct_correct'],['Fresh direct control','control_correct'],['Algorithm-guided','guided_correct']]){const el=document.createElement('div');el.className='metric';const b=document.createElement('b');b.textContent=(DATA.summary[key]??'N/A')+' / '+DATA.summary.attempted;el.append(b,document.createTextNode(label));$('metrics').append(el);}
for(const[c,i]of DATA.cases.map((c,i)=>[c,i])){const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' · '+c.source+' · '+c.type;$('case').append(o);}
function load(){const c=DATA.cases[Number($('case').value)];$('question').textContent=c.question;$('meta').textContent=c.source+' · '+c.type+' · '+c.seconds.toFixed(1)+' seconds · guided status: '+c.status;const img=new Image();img.onload=()=>{$('drawing').replaceChildren();$('drawing').setAttribute('viewBox','0 0 '+img.width+' '+img.height);add('image',{href:c.image,x:0,y:0,width:img.width,height:img.height});if($('rooms').checked)for(const r of c.catalog?.rooms||[])box(r.bbox,'#397cdb');if($('citations').checked)for(const r of c.guided?.canonical_citations||[])box(r.canonical_bbox,'#de6d2a');};img.src=c.image;$('answers').replaceChildren();for(const[label,answer,score]of[['Old direct',c.direct_answer,c.direct_score],['Fresh control',c.control?.result,c.control_score],['Guided answer',c.answer,c.guided_score]]){const e=document.createElement('article');e.className='answer';const h=document.createElement('h3');h.textContent=label+' · '+(score?.correct?'correct':'incorrect / unavailable');const p=document.createElement('p');p.textContent=answer?.status==='answered'?String(answer.answer)+' '+(answer.unit||''):'No validated answer';const note=document.createElement('p');note.className='muted';note.textContent=answer?.explanation||'';e.append(h,p,note);$('answers').append(e);}$('decision').textContent=JSON.stringify(c.decision||{error:c.error||c.error_type},null,2);$('evidence').textContent=JSON.stringify({rooms:c.catalog?.rooms?.length,omitted_observations:c.catalog?.omitted_observations,usage:c.usage,control_status:c.control_status,matched_overview_sha256:c.matched_overview_sha256,strict_score:c.guided_score,limits:DATA.binding.limits},null,2);}
$('case').onchange=load;$('rooms').onchange=load;$('citations').onchange=load;load();window.__guidedReview={cases:DATA.cases.length,summary:DATA.summary};
</script></html>""".replace("__DATA__", data)
    (root / "index.html").write_text(html, encoding="utf-8")
    print(root / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build(parser.parse_args().root)
