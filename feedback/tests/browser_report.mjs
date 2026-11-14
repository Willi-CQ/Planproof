// Optional browser QA. Uses only Node's built-in CDP client and a task-owned Chrome profile.
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";

const root = process.cwd();
const profile = path.join(root, ".artifacts/report-browser-liveqa");
const port = fs.readFileSync(path.join(profile, "DevToolsActivePort"), "utf8").split("\n")[0];
const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const target = targets.find(t => t.type === "page" && t.url.includes("/comparison/index.html"));
assert(target, "Owned comparison tab must exist");
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
let serial = 0;
const pending = new Map(), errors = [];
socket.onmessage = event => {
  const data = JSON.parse(event.data);
  if (data.method === "Runtime.exceptionThrown") errors.push(data.params.exceptionDetails.text);
  if (data.id && pending.has(data.id)) {
    const {resolve, reject, timer} = pending.get(data.id);
    clearTimeout(timer); pending.delete(data.id);
    data.error ? reject(Error(data.error.message)) : resolve(data.result);
  }
};
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++serial;
    const timer = setTimeout(() => { pending.delete(id); reject(Error("CDP timeout: " + method)); }, 10000);
    pending.set(id, {resolve, reject, timer}); socket.send(JSON.stringify({id, method, params}));
  });
}
async function evaluate(expression) {
  const value = await send("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
  assert(!value.exceptionDetails, JSON.stringify(value.exceptionDetails));
  return value.result.value;
}
const receipt = {};
try {
  await send("Runtime.enable");
  assert.equal(await evaluate('document.querySelectorAll(".question-cell").length'), 100);
  assert.equal(await evaluate('document.getElementById("a-accuracy").textContent'), "—");
  await evaluate('document.getElementById("type-filter").value="area";document.getElementById("type-filter").dispatchEvent(new Event("change"));');
  assert.equal(await evaluate('document.querySelectorAll(".question-cell").length'), 15);
  await evaluate('document.querySelector(".question-cell").click()');
  assert.match(await evaluate('document.getElementById("case-label").textContent'), /AREA/);
  await evaluate('document.getElementById("type-filter").value="all";document.getElementById("type-filter").dispatchEvent(new Event("change"));document.querySelector(".question-cell").click()');
  assert.equal(await evaluate('document.getElementById("drawing-image").naturalWidth > 0'), true);
  await send("Emulation.setDeviceMetricsOverride", {width: 390, height: 1000, deviceScaleFactor: 1, mobile: true});
  assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true, "Mobile must not overflow");
  const mobile = await send("Page.captureScreenshot", {format: "png"});
  fs.writeFileSync(path.join(root, ".artifacts/comparison-mobile.png"), Buffer.from(mobile.data, "base64"));
  await send("Emulation.setDeviceMetricsOverride", {width: 1440, height: 2000, deviceScaleFactor: 1, mobile: false});
  assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true);
  const desktop = await send("Page.captureScreenshot", {format: "png"});
  fs.writeFileSync(path.join(root, ".artifacts/comparison-desktop.png"), Buffer.from(desktop.data, "base64"));
  // Generated in-memory fixtures only. Never save fabricated model results into the real run registry.
  await evaluate(`{
    const c=DATA.cases[0];
    const row=(ok)=>({id:c.id,input_sha256:c.input_sha256,type:c.type,source:c.source,mode:"direct",status:"ok",answer:{status:"answered",answer:ok?c.reference:999999,unit:c.unit,explanation:"SIMULATED BROWSER TEST ONLY",citations:[{image_id:c.images[0].image_id,bbox:[10,10,100,100]}]},score:{scorable:true,correct:ok},images:c.images,usage:{total_tokens:100},latency_seconds:1});
    addReport({schema:"planproof-comparison-v1",dataset:{sha256:DATA.dataset.sha256},runs:[
      {id:"qa-nemotron",label:"SIMULATED Nemotron QA",mode:"direct",records:[row(true)]},
      {id:"qa-gemini",label:"SIMULATED Gemini QA",mode:"direct",records:[row(false)]}
    ]},"generated browser fixture");
    document.getElementById("run-a").value="qa-nemotron";
    document.getElementById("run-b").value="qa-gemini";
    document.getElementById("run-b").dispatchEvent(new Event("change"));
  }`);
  assert.equal(await evaluate('document.getElementById("paired-n").textContent'), "1");
  assert.equal(await evaluate('document.getElementById("delta").textContent'), "-100.0 pp");
  assert.equal(await evaluate('document.querySelectorAll("#drawing-boxes rect").length'), 2);
  await evaluate('document.getElementById("toggle-boxes").click()');
  assert.equal(await evaluate('document.getElementById("drawing-boxes").style.display'), "none");
  assert.equal(await evaluate('(()=>{try{addReport({schema:"planproof-comparison-v1",dataset:{sha256:"wrong"},runs:[]},"wrong");return false}catch{return true}})()'), true);
  assert.deepEqual(errors, []);
  receipt.status = "passed";
  receipt.checks = ["pending scores", "100 case map", "type filter", "selection", "source image",
    "390px layout", "1440px layout", "paired scores with simulated data", "citation overlay",
    "overlay toggle", "dataset mismatch rejection", "no JavaScript exceptions"];
  receipt.fixture_policy = "Simulated run records existed only in browser memory; real report remains not-run";
  fs.writeFileSync(path.join(root, ".artifacts/browser-qa-receipt.json"), JSON.stringify(receipt, null, 2));
  console.log(JSON.stringify(receipt));
} finally {
  await send("Browser.close").catch(() => {});
  socket.close();
}
