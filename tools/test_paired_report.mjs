// Task-owned headless Chrome and Node's built-in CDP client; no user profile.
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const page=path.resolve(process.argv[2]),profile=path.resolve(process.argv[3]);
fs.mkdirSync(profile,{recursive:true});
fs.rmSync(path.join(profile,'DevToolsActivePort'),{force:true});
const browser=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',[
 '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
 '--remote-debugging-port=0','--user-data-dir='+profile,pathToFileURL(page).href,
],{windowsHide:true,stdio:'ignore'});
const pause=ms=>new Promise(r=>setTimeout(r,ms));
let socket;
try{
 const active=path.join(profile,'DevToolsActivePort');
 for(let i=0;i<100&&!fs.existsSync(active);i++)await pause(100);
 const port=fs.readFileSync(active,'utf8').split('\n')[0];
 let target;
 for(let i=0;i<50&&!target;i++){
  const targets=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  target=targets.find(t=>t.type==='page'&&t.url===pathToFileURL(page).href);
  if(!target)await pause(100);
 }
 assert(target,'Owned page exists');
 socket=new WebSocket(target.webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
 const pending=new Map(),errors=[];let serial=0;
 socket.onmessage=event=>{
  const data=JSON.parse(event.data);
  if(data.method==='Runtime.exceptionThrown')errors.push(data.params.exceptionDetails.text);
  if(data.id&&pending.has(data.id)){
   const p=pending.get(data.id);clearTimeout(p.timer);pending.delete(data.id);
   data.error?p.reject(Error(data.error.message)):p.resolve(data.result);
  }
 };
 function send(method,params={}){
  return new Promise((resolve,reject)=>{
   const id=++serial,timer=setTimeout(()=>reject(Error('CDP timeout '+method)),10000);
   pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}));
  });
 }
 async function evaluate(expression){
  const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  assert(!r.exceptionDetails,JSON.stringify(r.exceptionDetails));return r.result.value;
 }
 await send('Runtime.enable');
 await send('Page.enable');
 for(let i=0;i<50&&!await evaluate('Boolean(window.__pairedReview)');i++)await pause(100);
 assert.equal(await evaluate('document.querySelectorAll(".case").length'),8);
 assert.equal(await evaluate('document.querySelectorAll(".answer").length'),2);
 assert.equal(await evaluate('document.querySelectorAll("#timeline li").length'),5);
 const summary=await evaluate('window.__pairedReview.summary');
 if(summary)for(const side of ['direct','guided']){
  assert.equal(await evaluate(`document.getElementById('${side}-score').textContent`),`${summary[side].correct} / ${summary[side].total} correct`);
 }
 for(let i=0;i<8;i++){
  await evaluate(`document.querySelectorAll('.case')[${i}].click()`);
  let loaded=false;
  for(let j=0;j<50;j++){
   loaded=await evaluate('document.getElementById("image").complete&&document.getElementById("image").naturalWidth>0');
   if(loaded)break;await pause(100);
  }
  assert(loaded,`Drawing ${i+1} loads`);
  assert(await evaluate('document.getElementById("original").href.endsWith(".png")'));
 }
 await evaluate("document.querySelector('.case').click();document.getElementById('show-rooms').click()");
 assert(await evaluate('document.querySelectorAll("#overlay .room").length>0'),'Real room proposals render');
 await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1400,deviceScaleFactor:1,mobile:false});
 assert(await evaluate('document.documentElement.scrollWidth<=innerWidth'));
 fs.writeFileSync(path.join(profile,'desktop.png'),Buffer.from((await send('Page.captureScreenshot',{format:'png'})).data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:390,height:1000,deviceScaleFactor:1,mobile:true});
 assert(await evaluate('document.documentElement.scrollWidth<=innerWidth'),'Mobile does not overflow');
 fs.writeFileSync(path.join(profile,'mobile.png'),Buffer.from((await send('Page.captureScreenshot',{format:'png'})).data,'base64'));
 assert.deepEqual(errors,[]);
 const receipt={checks:8,drawings_loaded:8,console_errors:errors,review:await evaluate('window.__pairedReview')};
 fs.writeFileSync(path.join(profile,'receipt.json'),JSON.stringify(receipt,null,2));
 console.log(JSON.stringify(receipt,null,2));
 await send('Browser.close');
}finally{
 socket?.close();browser.kill();
}
