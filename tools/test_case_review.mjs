// Offline review verification against its embedded, frozen evaluation records.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {pathToFileURL} from 'node:url';

const out=path.resolve('.artifacts/case-review-check');
await fs.mkdir(out,{recursive:true});
const profile=path.join(out,'browser-'+Date.now());
const child=spawn(process.env.PLANPROOF_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',[
  '--headless','--disable-gpu','--disable-background-networking','--no-first-run',
  '--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{windowsHide:true,stdio:'ignore'});
let ws,seq=0;const pending=new Map(),errors=[],requests=[],checks=[];
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn){for(let i=0;i<160;i++){try{if(await fn())return;}catch{}await delay(150);}throw new Error('Browser condition timeout');}
function rpc(method,params={}){return new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(new Error(method+' timeout')),20000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;}
try{
  let port;await until(async()=>{port=Number((await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0]);return port;});
  const tabs=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
  ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.id){const p=pending.get(d.id);if(!p)return;clearTimeout(p.timer);pending.delete(d.id);d.error?p.reject(new Error(JSON.stringify(d.error))):p.resolve(d.result);}else if(d.method==='Runtime.exceptionThrown')errors.push(d.params);else if(d.method==='Network.requestWillBeSent')requests.push(d.params.request.url);};
  await rpc('Runtime.enable');await rpc('Page.enable');await rpc('Network.enable');
  await rpc('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await rpc('Page.navigate',{url:pathToFileURL(path.resolve(process.argv[2])).href});
  await until(()=>evaluate("document.querySelectorAll('#rows tr').length===50 && document.getElementById('drawing').naturalWidth>0"));
  assert.equal(await evaluate('data.cases.length'),50);
  assert.deepEqual(await evaluate("Object.fromEntries(['correct','wrong','failed'].map(s=>[s,data.cases.filter(c=>c.attempts.auto.outcome===s).length]))"),{correct:33,wrong:2,failed:15});
  checks.push('All 50 cases; 33 correct, 2 wrong, 15 execution failures');
  const mismatches=await evaluate(`(()=>{const bad=[];for(const c of data.cases){choose(c.id);if($('question').textContent!==c.english||$('reference').textContent!==value(c.reference))bad.push(c.number);for(const [i,m] of ['auto','direct'].entries())if(document.querySelectorAll('#attempts .answer')[i].textContent!==value(c.attempts[m].answer))bad.push(c.number);}return bad;})()`);
  assert.deepEqual(mismatches,[]);checks.push('Every English question, reference and both answers render faithfully');
  assert.equal(await evaluate("data.cases.filter(c=>c.language!=='en'&&c.english!==c.question).length"),24);
  checks.push('All 24 non-English questions have display translations with originals retained');
  for(const mode of ['auto','direct'])for(const state of ['correct','wrong','failed','abstained','not_correct','agent_only']){
    const result=await evaluate(`$('mode').value='${mode}';$('outcome').value='${state}';filter();[visible.length,data.cases.filter(c=>'${state}'==='not_correct'?c.attempts['${mode}'].outcome!=='correct':'${state}'==='agent_only'?c.attempts.auto.outcome==='correct'&&c.attempts.direct.outcome!=='correct':c.attempts['${mode}'].outcome==='${state}').length,document.querySelectorAll('#rows tr').length]`);
    assert.equal(result[0],result[1]);assert.equal(result[0],result[2]);
  }checks.push('All outcome filters match recorded results for both modes, including empty state');
  await evaluate("$('mode').value='auto';$('outcome').value='all';$('category').value='area';filter()");
  assert.equal(await evaluate('visible.length'),12);
  await evaluate("$('category').value='all';$('search').value='Stanza';filter()");assert.equal(await evaluate('visible.length'),1);
  checks.push('Category and English question search work');
  await evaluate("$('search').value='';filter();choose(data.cases[27].id);$('detail').scrollIntoView()");
  await until(()=>evaluate("$('drawing').complete&&$('drawing').naturalWidth>0"));
  assert.match(await evaluate("document.querySelector('#attempts .answer').textContent"),/^2$/);
  assert.match(await evaluate("$('reference').textContent"),/^1$/);
  await evaluate("document.querySelector('#attempts button').click()");
  assert.ok(await evaluate("document.querySelectorAll('#overlay rect').length>0"));
  let shot=await rpc('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(out,'wrong-answer.png'),Buffer.from(shot.data,'base64'));
  checks.push('Wrong count versus reference and canonical citation overlays shown');
  await evaluate("choose(data.cases.find(c=>c.attempts.auto.outcome==='failed').id)");
  assert.match(await evaluate("document.querySelector('#attempts .answer').textContent"),/No final answer/);
  assert.match(await evaluate("document.querySelector('#attempts').textContent"),/Recorded Gemini errors/);
  checks.push('Execution failures show no invented answer and actual exception classes');
  await evaluate("choose(data.cases.find(c=>c.attempts.auto.details.tool_trace.some(t=>t.name==='inspect_nemotron')).id);for(const d of document.querySelectorAll('#attempts details'))if(d.querySelector('summary').textContent.startsWith('Nemotron'))d.open=true");
  await until(()=>evaluate("document.querySelectorAll('#attempts canvas').length>0"));
  assert.ok(await evaluate("[...document.querySelectorAll('#attempts canvas')].every(c=>c.width>0&&c.height>0&&c.getContext('2d').getImageData(0,0,c.width,c.height).data.some((v,i)=>i%4!==3&&v<240))"));
  checks.push('Embedded source crops render nonblank pixels offline');
  await evaluate("$('zoom').value='200';$('zoom').dispatchEvent(new Event('change'))");
  assert.equal(await evaluate("$('canvas-wrap').style.width"),'200%');
  await evaluate("choose(data.cases[0].id);window.scrollTo(0,0)");
  shot=await rpc('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(out,'overview.png'),Buffer.from(shot.data,'base64'));
  await rpc('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert.equal(await evaluate('document.documentElement.scrollWidth>window.innerWidth+2'),false);
  await evaluate("$('detail').scrollIntoView()");
  shot=await rpc('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(out,'mobile.png'),Buffer.from(shot.data,'base64'));
  checks.push('Zoom works and mobile viewport has no page overflow');
  assert.equal(errors.length,0);assert.equal(requests.filter(u=>/^https?:/.test(u)).length,0);
  checks.push('No JavaScript errors or external network requests');
  await fs.writeFile(path.join(out,'checks.json'),JSON.stringify({checks,errors,externalRequests:0},null,2));
  console.log(JSON.stringify({passed:checks.length,checks},null,2));
}finally{if(ws)ws.close();child.kill();for(const p of pending.values())clearTimeout(p.timer);}
