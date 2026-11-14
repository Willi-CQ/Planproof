// Build an offline HTML using the actual UI components and allowlisted saved data.
// Private manifests, original drawings and the output stay under .artifacts/.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {build} from '../frontend/node_modules/vite/dist/node/index.js';
import react from '../frontend/node_modules/@vitejs/plugin-react/dist/index.js';
import tailwindcss from '../frontend/node_modules/@tailwindcss/vite/dist/index.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args=process.argv.slice(2);
if(args.length!==2) throw new Error('Usage: node tools/build_visual_replay.mjs PRIVATE_MANIFEST OUTPUT_HTML');
const manifest=JSON.parse(await fs.readFile(path.resolve(args[0]), 'utf8'));
const output=path.resolve(args[1]);
if(!output.startsWith(path.join(root,'.artifacts')+path.sep)) throw new Error('Replay output must stay in the private .artifacts directory.');
const original=await fs.readFile(path.resolve(manifest.image));
if(original.subarray(0,8).toString('hex')!=='89504e470d0a1a0a') throw new Error('Expected an unchanged PNG source.');
const sha256=createHash('sha256').update(original).digest('hex');
const width=original.readUInt32BE(16), height=original.readUInt32BE(20);
const pick=(value,keys)=>Object.fromEntries(keys.filter(k=>value?.[k]!==undefined).map(k=>[k,value[k]]));
const frame=value=>pick(value,['raster_sha256','page_index','width','height']);
const regions=values=>(values||[]).map(value=>pick(value,['image_id','canonical_bbox','page_index','sent_size']));
const examples=[];
for(const item of manifest.examples) {
  const record=JSON.parse(await fs.readFile(path.resolve(item.record),'utf8'));
  if(record.source.source_sha256!==sha256 || record.source.width!==width || record.source.height!==height)
    throw new Error('Saved record does not match the unchanged source image.');
  const events=record.events.map(e=>({...pick(e,['sequence','stage','state','elapsed','seconds','summary','question','inspection_id','classes','delivered_ids','checks','unresolved','next_action','phase','category','cached']),
    images:regions(e.images),citations:regions(e.citations),...(e.advisory?{advisory:pick(e.advisory,['status','answer','findings','verdict'])}:{})}));
  const result=record.result ? {...pick(record.result,['status','completion_reason']),
    answer:pick(record.result.answer,['status','answer','unit','explanation','coverage','uncertainties','briefing','citation_details']),
    source:frame(record.result.source),canonical_citations:regions(record.result.canonical_citations),
    ...(record.result.measurement?{measurement:pick(record.result.measurement,['approximate','boundary_basis','formula'])}:{}),
    ...(record.result.reliability?{reliability:pick(record.result.reliability,['retry_after_seconds','recovery_used'])}:{}),
    ...(record.result.count_evidence?{count_evidence:{...pick(record.result.count_evidence,['unit','supported','uncertain','complete']),
      objects:(record.result.count_evidence.objects||[]).map(o=>pick(o,['id','status','reason','image_id','canonical_bbox','page_index']))}}:{})} : null;
  const run={...pick(record,['request_id','status','created','last_sequence']),source:frame(record.source),events,result};
  const evidence={...pick(record.evidence,['version','status','scope','warnings']),source:frame(record.evidence?.source),
    shapes:(record.evidence?.shapes||[]).map(s=>pick(s,['id','class','layer','bbox','polygons','role','reason','method','status','source_family','cues'])),
    comparisons:(record.evidence?.comparisons||[]).map(c=>pick(c,['relation','source_ids','bbox','iou','semantics']))};
  examples.push({title:item.title,note:item.note,question:record.question,run,evidence});
}
const payload={image:{name:path.basename(manifest.image),base64:original.toString('base64'),sha256},examples,sourceRevision:manifest.sourceRevision};
const outDir=path.join(path.dirname(output),'bundle');
await build({configFile:false,root:path.join(root,'frontend'),plugins:[react(),tailwindcss()],define:{'process.env.NODE_ENV':JSON.stringify('production')},
  build:{outDir,emptyOutDir:false,lib:{entry:path.join(root,'frontend/src/replay/main.tsx'),name:'PlanproofReplay',formats:['iife'],fileName:()=> 'replay.js'},cssCodeSplit:false,minify:true}});
const js=await fs.readFile(path.join(outDir,'replay.js'),'utf8');
const cssFiles=(await fs.readdir(outDir)).filter(name=>name.endsWith('.css'));
if(cssFiles.length!==1) throw new Error('Expected exactly one self-contained stylesheet.');
const css=await fs.readFile(path.join(outDir,cssFiles[0]),'utf8');
const html=`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'"><title>Planproof — Recorded tool replay</title><style>${css.replaceAll('</style','<\\/style')}</style></head><body><div id="root"></div><script id="replay-data" type="application/json">${JSON.stringify(payload).replaceAll('<','\\u003c')}</script><script>${js.replaceAll('</script','<\\/script')}</script></body></html>`;
await fs.writeFile(output,html,'utf8');
await fs.writeFile(path.join(path.dirname(output),'build-receipt.json'),JSON.stringify({file:path.basename(output),bytes:Buffer.byteLength(html),imageSha256:sha256,width,height,examples:examples.map(e=>({title:e.title,run:e.run.request_id,events:e.run.events.length})),network:'blocked by CSP',providerCalls:0},null,2));
console.log(JSON.stringify({output,bytes:Buffer.byteLength(html),examples:examples.length,providerCalls:0}));
