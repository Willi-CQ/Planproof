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
  assert.equal(await evaluate("document.querySelector('#scores tr[data-score-class=wall] .score-value').textContent"),'N/A');
  assert.equal(await evaluate("document.querySelector('#scores tr[data-score-class=door] td:last-child').textContent"),'2 plans');
  checks.push('CAD score comparison uses two matched plans and no wall score');
  await evaluate("document.getElementById('model').value='main';window.__evidenceLab.selectCase('P05-resplan');document.querySelector('[data-view=both]').click()");
  const mlWalls=await evaluate("document.querySelectorAll('#drawing [data-method=ml][data-class=wall]').length");
  assert.ok(mlWalls>0);
  await evaluate("document.getElementById('class-det-wall').click()");
  assert.equal(await evaluate("document.querySelectorAll('#drawing [data-method=det][data-class=wall]').length"),0);
  assert.equal(await evaluate("document.querySelectorAll('#drawing [data-method=ml][data-class=wall]').length"),mlWalls);
  await evaluate("document.getElementById('class-det-wall').click();document.querySelector('[data-view=det]').click()");
  assert.equal(await evaluate("document.querySelectorAll('#drawing [data-method=ml]').length"),0);
  assert.ok(await evaluate("document.querySelectorAll('#drawing [data-method=det][data-class=room]').length>0"));
  assert.equal(await evaluate("document.querySelector('#drawing [data-method=det][data-class=wall]').getAttribute('stroke')"),'#755334');
  checks.push('Per-method class toggles, deterministic-only preset and class colors work');
  const beforeScores=await evaluate("document.getElementById('scores').textContent");
  await evaluate("document.querySelector('[data-classes=none]').click()");
  assert.equal(await evaluate("document.querySelectorAll('#drawing [data-obs]').length"),0);
  assert.equal(await evaluate("document.getElementById('scores').textContent"),beforeScores);
  await evaluate("document.querySelector('[data-classes=all]').click()");
  const report=JSON.parse(await fs.readFile(path.join(pack,'failure-report.json'),'utf8'));
  const expectedWall=report.results.filter(r=>r.method==='main'&&r.metrics.wall?.status==='scored').map(r=>r.metrics.wall.mask_iou);
  const wallMean=expectedWall.reduce((a,b)=>a+b,0)/expectedWall.length;
  assert.equal(await evaluate("document.querySelector('#scores tr[data-score-class=wall] td:nth-child(2) .score-value').textContent"),(wallMean*100).toFixed(1)+'%');
  await evaluate("document.getElementById('scoreScope').value='page';document.getElementById('scoreScope').dispatchEvent(new Event('change'))");
  const pageWall=report.results.find(r=>r.method==='deterministic'&&r.case_id==='P05-resplan').metrics.wall.mask_iou;
  assert.equal(await evaluate("document.querySelector('#scores tr[data-score-class=wall] td:nth-child(3) .score-value').textContent"),(pageWall*100).toFixed(1)+'%');
  await evaluate("window.__evidenceLab.selectCase('P07-resplan')");
  assert.equal(await evaluate("document.querySelector('#scores tr[data-score-class=stairs] .score-value').textContent"),'N/A');
  checks.push('Displayed cohort/page scores match frozen report; filters cannot alter scores; missing labels stay N/A');
  await evaluate("window.__evidenceLab.selectCase('P05-resplan');document.getElementById('scoreScope').value='cohort';document.getElementById('scoreScope').dispatchEvent(new Event('change'));document.querySelector('[data-agreement-class=room]').click()");
  assert.equal(await evaluate("[...document.querySelectorAll('#drawing [data-obs]')].every(x=>x.dataset.class==='room')"),true);
  assert.ok(await evaluate("document.querySelectorAll('#drawing [data-relation=agreement]').length>0"));
  assert.equal(await evaluate("document.querySelector('[data-agreement-class=room]').closest('tr').children[1].textContent"),'7');
  checks.push('Class-specific agreement view shows seven room pairs with both source outlines');
  await evaluate("document.querySelectorAll('#feedbackItems button')[3].click()");
  assert.equal((await evaluate('window.__evidenceLab.counts()')).selected.length,2);
  assert.ok(await evaluate("document.getElementById('inspectStatus').textContent.includes('both are false positives')"));
  await until(()=>evaluate("document.getElementById('cropEmpty').hidden"),'review finding clean crop');
  checks.push('Agent review opens the real shared false-positive agreement and clean crop');
  await evaluate("window.scrollTo(0,0)");
  await screenshot('evidence-lab-shared-error.png');
  await evaluate("document.querySelector('.score-panel').scrollIntoView({block:'start'})");
  await screenshot('evidence-lab-scores.png');
  await evaluate("document.querySelector('.agreement-panel').scrollIntoView({block:'start'})");
  await screenshot('evidence-lab-review.png');
  await evaluate("document.getElementById('model').value='main';window.__evidenceLab.selectCase('P05-resplan');document.querySelector('[data-stage=ml]').click();document.getElementById('fit').click()");
  await evaluate("document.querySelector('[data-classes=all]').click();document.querySelector('[data-view=both]').click();window.scrollTo(0,0)");
  await screenshot('evidence-lab-overview.png');
  if(await evaluate('Boolean(DATA.room_report)')){
    for(const id of ['P05-resplan','P01-cubicasa5k']){
      await evaluate(`window.__evidenceLab.selectCase('${id}');document.querySelector('[data-classes=all]').click();document.querySelector('[data-view=ml]').click();window.scrollTo(0,0)`);
      await screenshot('room-baseline-'+id+'.png');
      await evaluate(`window.__evidenceLab.selectCase('${id}');document.querySelector('[data-view=rooms]').click();window.scrollTo(0,0)`);
      assert.ok(await evaluate("document.querySelectorAll('#drawing [data-room]').length>0"));
      assert.equal(await evaluate("document.querySelectorAll('#drawing [data-obs]').length"),0);
      await screenshot('room-boundaries-'+id+'.png');
      await evaluate("document.getElementById('roomSelect').selectedIndex=1;document.getElementById('roomSelect').dispatchEvent(new Event('change'))");
      await until(()=>evaluate("document.getElementById('cropEmpty').hidden"),'room clean crop');
      assert.ok(await evaluate("document.getElementById('details').textContent.includes('ML parents')"));
      await screenshot('room-crop-'+id+'.png');
      await evaluate("document.getElementById('room-layer-regions').click()");
      await evaluate("document.getElementById('room-layer-alternatives').click()");
      assert.ok(await evaluate("document.querySelectorAll('#drawing [data-boundary-alternative]').length>0"));
      await evaluate("document.getElementById('room-layer-alternatives').click()");
      assert.equal(await evaluate("document.querySelectorAll('#drawing [data-room]').length"),0);
      await evaluate("document.getElementById('room-layer-regions').click()");
    }
    checks.push('Simple and complex room views render, select clean crops and toggle independently');
    await evaluate("document.querySelector('[data-opening]')?.dispatchEvent(new MouseEvent('click',{bubbles:true}))");
    assert.ok(await evaluate("document.getElementById('inspectTitle').textContent.includes('hypothesis')"));
    checks.push('Opening hypotheses expose rule evidence and ML alternatives');
  }
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
