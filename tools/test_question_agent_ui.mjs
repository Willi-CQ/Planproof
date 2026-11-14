// Real browser/UI check. The answer response is explicitly simulated; upload is real.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {pathToFileURL} from 'node:url';

const out=path.resolve('.artifacts/agent-ui-check');
await fs.mkdir(out,{recursive:true});
const profile=path.join(out,'browser-'+Date.now());
const child=spawn(process.env.PLANPROOF_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',[
  '--headless','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-address=127.0.0.1',
  '--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{windowsHide:true,stdio:'ignore'});
let ws,seq=0; const pending=new Map(),errors=[],checks=[];
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn){for(let i=0;i<160;i++){try{if(await fn())return;}catch{}await delay(150);}throw new Error('Browser condition timeout');}
function rpc(method,params={}){return new Promise((resolve,reject)=>{const id=++seq;const timer=setTimeout(()=>reject(new Error(method+' timeout')),15000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;}
try{
  let port;await until(async()=>{port=Number((await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0]);return port;});
  const tabs=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
  ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.id){const p=pending.get(d.id);if(!p)return;clearTimeout(p.timer);pending.delete(d.id);d.error?p.reject(new Error(JSON.stringify(d.error))):p.resolve(d.result);}else if(d.method==='Runtime.exceptionThrown')errors.push(d.params);};
  await rpc('Runtime.enable');await rpc('Page.enable');
  await rpc('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await rpc('Page.navigate',{url:(process.env.PLANPROOF_TEST_URL||'http://127.0.0.1:8002')+'/planwise/'});
  await until(()=>evaluate("Boolean(document.getElementById('reasonButton'))"));
  assert.deepEqual(await evaluate("[...document.getElementById('reasonMode').options].map(x=>x.value)"),['auto','visual_only','direct']);checks.push('Three distinct question modes present');
  const source=path.resolve(process.argv[2]||'../_planproof_evidence/.artifacts/reference10-integrated-v1/cases/P05-resplan/plan.png');
  const {root}=await rpc('DOM.getDocument'); const {nodeId}=await rpc('DOM.querySelector',{nodeId:root.nodeId,selector:'#fileInput'});
  await rpc('DOM.setFileInputFiles',{nodeId,files:[source]});
  await evaluate("document.getElementById('reasonQuestion').value='How many rooms are shown?';document.getElementById('reasonButton').click()");
  assert.match(await evaluate("document.getElementById('reasonStatus').textContent"),/Enable drawing inspection/);checks.push('Provider permission checked before request');
  await evaluate(`window.__realFetch=window.fetch;window.__askBodies=[];window.fetch=async(url,options)=>{
    if(String(url).endsWith('/ask')){window.__askBodies.push(JSON.parse(options.body));return new Response(JSON.stringify({status:'ok',answer:{status:'answered',answer:2,unit:'count',explanation:'SIMULATED UI TEST ONLY',coverage:'Generated UI response, not an accuracy result'},algorithm_used:false,nemotron_calls:0,seconds:1,tool_trace:[],source:{width:window.__upload.width,height:window.__upload.height,page_index:0},image_url:window.__upload.image_url,canonical_citations:[{canonical_bbox:[10,10,100,100]}]}),{headers:{'Content-Type':'application/json'}});}
    const result=await window.__realFetch(url,options);if(String(url)==='/api/documents')window.__upload=await result.clone().json();return result;};
    document.getElementById('reasonPermission').checked=true;document.getElementById('reasonMode').value='direct';document.getElementById('reasonButton').click();`);
  await until(()=>evaluate("document.getElementById('reasonStatus').textContent.startsWith('Answer ready')"));
  assert.equal(await evaluate('window.__askBodies[0].mode'),'direct');
  assert.equal(await evaluate('window.__askBodies[0].allow_nemotron'),false);
  assert.match(await evaluate("document.getElementById('reasonAnswer').textContent"),/SIMULATED/);
  assert.equal(await evaluate("document.querySelectorAll('.evidence-marker').length"),1);checks.push('Real upload, selected mode, simulated final answer and source citation rendered');
  await evaluate(`window.__uiFetch=window.fetch;window.fetch=async(url,options)=>{
    if(String(url).endsWith('/ask'))return new Response(JSON.stringify({status:'unavailable',source:{width:window.__upload.width,height:window.__upload.height,page_index:0},image_url:window.__upload.image_url,answer:{status:'abstain',answer:null,explanation:'SIMULATED outage: no answer is asserted.'},canonical_citations:[],tool_trace:[],reliability:{retryable:true,retry_after_seconds:20,last_error:{category:'cooldown'},provider_attempts:[{attempt:1,phase:'normal',category:'server_error',http_status:503,seconds:2}]}}),{headers:{'Content-Type':'application/json'}});
    return window.__uiFetch(url,options);};document.getElementById('reasonButton').click();`);
  await until(()=>evaluate("document.getElementById('reasonStatus').textContent.includes('Wait at least 20 seconds')"));
  assert.match(await evaluate("document.getElementById('reasonAnswer').textContent"),/no answer is asserted/);
  assert.match(await evaluate("document.getElementById('reasonTrace').textContent"),/HTTP 503/);
  assert.equal(await evaluate("document.querySelectorAll('.evidence-marker').length"),0);
  checks.push('Outage rendering clears stale answers/citations and displays cooldown with HTTP status');
  await evaluate(`window.fetch=async(url,options)=>{const r=await window.__uiFetch(url,options);if(!String(url).endsWith('/ask'))return r;const result=await r.json();result.reliability={recovery_succeeded:true,provider_attempts:[]};return new Response(JSON.stringify(result),{headers:{'Content-Type':'application/json'}});};document.getElementById('reasonButton').click();`);
  await until(()=>evaluate("document.getElementById('reasonStatus').textContent.includes('Recovered after an interrupted')"));
  checks.push('Successful recovery is visibly distinguished from an uninterrupted answer');
  await evaluate("document.querySelector('.reasoning-panel').scrollIntoView()");
  const screenshot=await rpc('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(out,'desktop.png'),Buffer.from(screenshot.data,'base64'));
  await rpc('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert.equal(await evaluate('document.documentElement.scrollWidth > window.innerWidth + 2'),false);checks.push('Mobile page does not overflow horizontally');
  await evaluate("document.getElementById('pageNumber').dispatchEvent(new Event('change'))");
  assert.equal(await evaluate("document.getElementById('reasonAnswer').textContent"),'');
  assert.equal(await evaluate("document.querySelectorAll('.evidence-marker').length"),0);checks.push('Changing the selected page clears stale answers and citations');
  assert.equal(errors.length,0);checks.push('No browser runtime errors');
  if(process.argv[3]){
    await rpc('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
    await rpc('Page.navigate',{url:pathToFileURL(path.resolve(process.argv[3])).href});
    await until(()=>evaluate("document.querySelectorAll('#cards .card').length===Object.keys(JSON.parse(document.getElementById('data').textContent).summary).length && document.getElementById('drawing').naturalWidth>0"));
    await evaluate("window.__comparison=JSON.parse(document.getElementById('data').textContent);window.__cropCase=window.__comparison.cases.find(c=>c.mode==='auto'&&c.details.canonical_citations.length&&c.details.tool_trace.some(t=>['inspect_nemotron','review_answer'].includes(t.name)&&t.result.status==='ok'));document.getElementById('case').value=window.__cropCase.id;document.getElementById('case').dispatchEvent(new Event('change'));document.querySelector('#cards button').click()");
    assert.ok(await evaluate("document.querySelectorAll('#overlay rect').length>0"));
    const cropPanels = await evaluate("[...document.querySelectorAll('#cards details')].filter(d=>d.querySelector('summary').textContent==='Crops actually supplied to Nemotron').map(d=>{d.open=true;return true}).length");
    assert.ok(cropPanels>0,'Fixture question must show the crops actually sent to Nemotron');
    await until(()=>evaluate("document.querySelectorAll('#cards canvas').length>0"));
    // file:// image canvases are intentionally origin-tainted in Chromium.
    // Check loaded image rendering dimensions; inspect the saved screenshot visually.
    assert.ok(await evaluate("[...document.querySelectorAll('#cards canvas')].every(c=>c.width>0&&c.height>0&&c.getBoundingClientRect().width>0)"));
    assert.equal(errors.length,0);
    const shot=await rpc('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(out,'live-comparison.png'),Buffer.from(shot.data,'base64'));
    await evaluate("document.querySelector('#cards canvas').scrollIntoView({block:'center'})");
    const cropShot=await rpc('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(out,'live-crops.png'),Buffer.from(cropShot.data,'base64'));
    checks.push('Real comparison report renders questions, all available modes and source citations');
    checks.push('Actual Nemotron detail crops load and render at valid dimensions');
    if(await evaluate("Boolean(window.__comparison.breakdown)")){
      assert.equal(await evaluate("document.querySelectorAll('#case option').length"),await evaluate("window.__comparison.binding.case_ids.length"));
      await evaluate("document.getElementById('outcome').value='auto_wrong';document.getElementById('outcome').dispatchEvent(new Event('change'))");
      assert.equal(await evaluate("document.querySelectorAll('#case option').length"),await evaluate("window.__comparison.cases.filter(c=>c.mode==='auto'&&!c.score.correct).length"));
      await evaluate("document.getElementById('outcome').value='all';document.getElementById('outcome').dispatchEvent(new Event('change'));document.getElementById('group').value='area';document.getElementById('group').dispatchEvent(new Event('change'))");
      assert.equal(await evaluate("document.querySelectorAll('#case option').length"),await evaluate("window.__comparison.cases.filter(c=>c.mode==='auto'&&c.category==='area').length"));
      checks.push('Cohort category and failure filters preserve exact case counts');
    }
    await rpc('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    assert.equal(await evaluate('document.documentElement.scrollWidth > window.innerWidth + 2'),false);
    assert.equal(errors.length,0);checks.push('Comparison report fits mobile width without runtime errors');
  }
  await fs.writeFile(path.join(out,'report.json'),JSON.stringify({simulated_answer:true,checks},null,2));
  console.log(JSON.stringify({checks},null,2));
}finally{if(ws?.readyState===WebSocket.OPEN){try{await rpc('Browser.close');}catch{}ws.close();}child.kill();}
