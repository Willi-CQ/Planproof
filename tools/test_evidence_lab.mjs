// Real offline browser checks through a fresh, isolated Edge DevTools session.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {spawn} from 'node:child_process';

const pack=path.resolve(process.argv[2]||'.artifacts/reference10-integrated-v1');
const browser=process.env.PLANPROOF_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const profile=path.join(pack,'browser-test-'+Date.now());
await fs.mkdir(profile,{recursive:true});
const child=spawn(browser,['--headless','--disable-gpu','--disable-background-networking','--no-first-run',
  '--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],
  {windowsHide:true,stdio:'ignore'});
let ws,sequence=0;
const pending=new Map(),runtimeErrors=[];
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(fn,description,timeout=30000){const end=Date.now()+timeout;while(Date.now()<end){try{const value=await fn();if(value)return value;}catch{}await delay(150);}throw new Error('Timeout: '+description);}
function rpc(method,params={}){return new Promise((resolve,reject)=>{const id=++sequence;const timer=setTimeout(()=>{pending.delete(id);reject(new Error('CDP timeout '+method))},30000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const data=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(data.exceptionDetails)throw new Error(data.exceptionDetails.text+': '+data.exceptionDetails.exception?.description);return data.result.value;}
async function screenshot(name){const data=await rpc('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(path.join(pack,name),Buffer.from(data.data,'base64'));}
const checks=[];
try{
  const active=await until(async()=>await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8'),'isolated browser startup');
  const port=Number(active.split('\n')[0]);
  const targets=await (await fetch('http://127.0.0.1:'+port+'/json/list')).json();
  ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
  ws.onmessage=event=>{const data=JSON.parse(event.data);if(data.id){const task=pending.get(data.id);if(!task)return;clearTimeout(task.timer);pending.delete(data.id);if(data.error)task.reject(new Error(JSON.stringify(data.error)));else task.resolve(data.result);}else if(data.method==='Runtime.exceptionThrown')runtimeErrors.push(data.params.exceptionDetails);};
  await rpc('Runtime.enable');await rpc('Page.enable');
  await rpc('Emulation.setDeviceMetricsOverride',{width:1680,height:1200,deviceScaleFactor:1,mobile:false});
  await rpc('Page.navigate',{url:pathToFileURL(path.join(pack,'evidence-lab.html')).href});
  await until(()=>evaluate('Boolean(window.__evidenceLab)'),'dashboard initialization');
  const ids=await evaluate("[...document.getElementById('case').options].map(o=>o.value)");
  assert.equal(ids.length,10);
  for(const id of ids){const state=await evaluate(`window.__evidenceLab.selectCase(${JSON.stringify(id)});window.__evidenceLab.counts()`);assert.ok(state.observations>0);assert.ok(state.comparisons>0);}
  checks.push('All ten real cases render with observations and comparisons');
  await evaluate("window.__evidenceLab.selectCase('P05-resplan')");
  const agreement=await evaluate('window.__evidenceLab.inspectAgreement()');
  assert.equal(agreement.relation,'agreement');assert.equal(agreement.source_ids.length,2);
  await until(()=>evaluate("document.getElementById('cropEmpty').hidden"),'clean source crop');
  assert.equal((await evaluate('window.__evidenceLab.counts()')).selected.length,2);
  checks.push('Agreement selection highlights both methods and shows the clean crop');
  await screenshot('evidence-lab-agreement.png');
  await evaluate("document.getElementById('layer-ml').click()");
  assert.equal(await evaluate("[...document.querySelectorAll('[data-obs]')].filter(e=>e.dataset.obs.includes(':ml-')).length"),0);
  await evaluate("document.getElementById('layer-ml').click();document.getElementById('layer-pixels').click()");
  assert.ok(await evaluate("[...document.querySelectorAll('#drawing image')].some(e=>e.getAttribute('href').endsWith('pixel-layer.png'))"));
  checks.push('ML and pixel layers toggle independently');
  await evaluate("document.querySelector('[data-stage=geometry]').click()");
  assert.ok(await evaluate("document.querySelectorAll('#drawing polyline').length>0"));
  await evaluate("document.getElementById('layer-truth').click();document.querySelector('[data-stage=source]').click()");
  assert.equal(await evaluate("document.getElementById('layer-truth').checked"),false);
  assert.equal(await evaluate("document.querySelectorAll('#drawing [data-obs]').length"),0);
  checks.push('Pipeline navigation shows raw vectors and a clean source-only stage');
  await evaluate("window.__evidenceLab.selectCase('P09-floorplancad');document.getElementById('model').value='cad';document.getElementById('model').dispatchEvent(new Event('change'))");
  assert.equal(await evaluate("document.querySelector('#counts tr:first-child td:nth-child(2)').textContent"),'N/A');
  checks.push('Object-only CAD specialist keeps wall coverage N/A');
  await evaluate("document.getElementById('model').value='main';window.__evidenceLab.selectCase('P05-resplan');document.querySelector('[data-stage=ml]').click();document.getElementById('fit').click()");
  await screenshot('evidence-lab-overview.png');
  await rpc('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert.ok(await evaluate('document.documentElement.scrollWidth<=390'));
  checks.push('Narrow screen has no document-level horizontal overflow');
  await screenshot('evidence-lab-mobile.png');
  assert.equal(runtimeErrors.length,0);
  const receipt={checks,passed:checks.length,runtime_errors:runtimeErrors,browser:'Isolated Edge / CDP',
                 tested_at:new Date().toISOString(),html:'evidence-lab.html'};
  await fs.writeFile(path.join(pack,'EVIDENCE_LAB_BROWSER_TEST.json'),JSON.stringify(receipt,null,2)+'\n');
  process.stdout.write(JSON.stringify(receipt,null,2)+'\n');
} finally {
  if(ws?.readyState===1){try{await rpc('Browser.close')}catch{}ws.close();}
  child.kill();
}
