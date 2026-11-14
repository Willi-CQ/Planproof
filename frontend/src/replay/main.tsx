import {createRoot} from "react-dom/client";
import {useEffect, useMemo, useRef, useState} from "react";
import {Play, Pause, SkipForward, SkipBack, RotateCcw, Scan, MessageCircle} from "lucide-react";
import {BrandMark} from "../components/BrandMark";
import {PlanCanvas2D} from "../components/PlanCanvas2D";
import {RunActivity} from "../components/RunActivity";
import {AnswerBriefing} from "../components/AnswerBriefing";
import {EvidenceInspector} from "../components/EvidenceInspector";
import {liveActivity} from "../lib/activity";
import {type AgentRun, type EvidenceBundle, type EvidenceShape, type Region} from "../lib/runs";
import type {Drawing} from "../lib/api";
import "../index.css";
import "./replay.css";

interface Example {title: string; note: string; question?: string; run: AgentRun; evidence: EvidenceBundle}
interface Payload {image: {name: string; base64: string; sha256: string}; examples: Example[]; sourceRevision: string}
const payload: Payload = JSON.parse(document.getElementById("replay-data")!.textContent!);
const image = new Blob([Uint8Array.from(atob(payload.image.base64), c => c.charCodeAt(0))], {type: "image/png"});
const imageUrl = URL.createObjectURL(image);
const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

function ToolReplay() {
  const [exampleIndex, setExampleIndex] = useState(0);
  const example = payload.examples[exampleIndex];
  const events = example.run.events;
  const [cursor, setCursor] = useState(() => Math.max(0, payload.examples[0].run.events.findIndex(e => e.stage === "get_algorithms")) + 1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState("guided");
  const [zoom, setZoom] = useState(100);
  const [follow, setFollow] = useState(false);
  const [focus, setFocus] = useState<Region | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [inspector, setInspector] = useState<"steps" | "candidates" | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [shapes, setShapes] = useState<EvidenceShape[]>([]);
  const [split, setSplit] = useState<number | null>(null);
  const [mobileTab, setMobileTab] = useState("ask");
  const conversation = useRef<HTMLDivElement>(null);
  const followScroll = useRef(true);
  const finished = cursor === events.length;
  const revealed = events.slice(0, cursor);
  const algorithmsReady = revealed.some(e => e.stage === "get_algorithms" && ["completed", "cached"].includes(e.state));
  const run: AgentRun = {...example.run, events: revealed, last_sequence: revealed.at(-1)?.sequence || 0,
    status: finished ? example.run.status : cursor ? "running" : "queued",
    result: finished ? example.run.result : null, evidence: loaded && algorithmsReady ? example.evidence : undefined};
  const drawing: Drawing = useMemo(() => ({document_id: "local-replay", page: example.run.source.page_index + 1,
    image_url: imageUrl, width: example.run.source.width, height: example.run.source.height, source: example.run.source}), [example]);
  const activity = liveActivity(run, drawing.source);
  useEffect(() => {
    if (follow && activity?.regions.length) {setFocus(activity.regions[0]); setSelected(null);}
  }, [follow, activity?.focusKey, exampleIndex]);
  useEffect(() => {
    if (!playing || finished) return;
    const previous = events[cursor - 1]?.elapsed || 0;
    const delay = speed === "guided" ? 2000 : Math.max(40, (events[cursor].elapsed - previous) * 1000 / Number(speed));
    const timer = setTimeout(() => setCursor(c => c + 1), delay);
    return () => clearTimeout(timer);
  }, [playing, cursor, speed, exampleIndex, finished]);
  useEffect(() => {if (finished) setPlaying(false);}, [finished]);
  useEffect(() => {
    const pane = conversation.current;
    if (pane && followScroll.current) pane.scrollTo({top: finished ? 0 : pane.scrollHeight, behavior: "instant"});
  }, [cursor, finished, inspector]);
  function closeInspector() {setInspector(null); setShapes([]); setSplit(null);}
  function seek(next: number) {
    setPlaying(false); setCursor(next); setFocus(null); setSelected(null); setLoaded(false);
    closeInspector(); followScroll.current = true;
  }
  function showRegion(region: Region) {setFocus(region); setSelected(null); setMobileTab("drawing");}
  function jump(stage: string) {
    const index = events.findIndex(e => e.stage === stage && e.state === "running");
    if (index >= 0) seek(index + 1);
  }
  return <div className="product product-loaded replay-shell">
    <header className="replay-header"><div className="brand"><BrandMark /><span>Planproof</span></div>
      <div><strong>Recorded tool replay</strong><span>Real saved execution · offline · no new model calls</span></div>
      <label>Example<select aria-label="Replay example" value={exampleIndex} onChange={e => {
        const index = Number(e.target.value); setExampleIndex(index); setZoom(100); setFollow(false);
        seek(Math.max(0, payload.examples[index].run.events.findIndex(event => event.stage === "get_algorithms")) + 1);
      }}>{payload.examples.map((item, i) => <option key={i} value={i}>{item.title}</option>)}</select></label>
    </header>
    <section className="replay-controls" aria-label="Replay controls">
      <div className="replay-transport">
        <button className="icon-button" aria-label="Restart replay" onClick={() => seek(0)}><RotateCcw size={16} /></button>
        <button className="icon-button" aria-label="Previous event" disabled={!cursor} onClick={() => seek(cursor - 1)}><SkipBack size={16} /></button>
        <button className="replay-play" onClick={() => {if (finished) seek(0); setPlaying(p => !p);}}>{playing ? <Pause size={15} /> : <Play size={15} />}{playing ? "Pause" : "Play"}</button>
        <button className="icon-button" aria-label="Next event" disabled={finished} onClick={() => seek(cursor + 1)}><SkipForward size={16} /></button>
        <label className="replay-speed"><span className="visually-hidden">Playback pace</span><select aria-label="Playback pace" value={speed} onChange={e => setSpeed(e.target.value)}>
          <option value="guided">Guided · 2s / event</option><option value="1">Original timing · 1×</option><option value="4">Original timing · 4×</option><option value="12">Original timing · 12×</option>
        </select></label>
      </div>
      <div className="replay-scrubber"><input type="range" aria-label="Recorded event" min="0" max={events.length} value={cursor} onChange={e => seek(Number(e.target.value))} />
        <span data-testid="replay-position">Event {cursor}/{events.length} · recorded +{clock(revealed.at(-1)?.elapsed || 0)}</span></div>
      <nav className="replay-jumps" aria-label="Jump to recorded work"><button onClick={() => jump("get_algorithms")}>ML + rules</button><button onClick={() => jump("inspect_nemotron")}>Nemotron</button><button onClick={() => seek(events.length)}>Recorded result</button></nav>
    </section>
    <nav className="mobile-tabs" aria-label="Workspace view"><button aria-label="Show drawing" className={mobileTab === "drawing" ? "active" : ""} onClick={() => setMobileTab("drawing")}><Scan size={15} />Drawing</button><button aria-label="Show conversation" className={mobileTab === "ask" ? "active" : ""} onClick={() => setMobileTab("ask")}><MessageCircle size={15} />Activity</button></nav>
    <main className="workspace" data-mobile-tab={mobileTab}>
      <div className="drawing-column"><PlanCanvas2D drawing={drawing} name={payload.image.name} hasTask={true} taskOpen={!!inspector}
        onTaskDetails={() => {if (inspector) closeInspector(); else {setInspector("steps"); setMobileTab("ask");}}}
        liveRun={run} followLive={follow} onFollowLive={setFollow} onManualView={() => setFollow(false)}
        result={run.result || undefined} focusRegion={focus} shapes={shapes} comparisonSplit={split}
        selectedIndex={selected} onSelectEvidence={index => {setSelected(index); setFocus(null);}}
        zoom={zoom} onZoomChange={setZoom} isPdf={false} disabled={false} loadingPage={false} onPageChange={() => {}} /></div>
      {inspector ? <EvidenceInspector key={`${exampleIndex}-${inspector}`} initialView={inspector} run={run} drawing={drawing} frame={drawing.source}
        connection="Recorded replay" onBack={closeInspector} onLoadLayers={() => {if (algorithmsReady) setLoaded(true);}}
        loading={false} layerError="" onFocus={showRegion} onShapes={setShapes} onCompare={setSplit} follow={follow} onFollow={setFollow} />
        : <section className="assistant-panel replay-panel" aria-label="Recorded activity">
          <header className="assistant-header"><h2>{example.title}</h2><span className="replay-badge">Replay</span></header>
          <div className="conversation" ref={conversation} onScroll={() => {const pane = conversation.current!; followScroll.current = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 80;}}>
            <aside className="replay-note">{example.note}</aside>
            {example.question ? <p className="user-message">{example.question}</p> : <p className="replay-original-note">The original prompt was not saved with this older run. Tool events and the recorded answer are preserved.</p>}
            {run.result && <AnswerBriefing result={run.result} drawing={drawing} messageId={run.request_id} selectedEvidence={selected === null ? null : {message: run.request_id, index: selected}}
              onHighlight={(_, index) => {setSelected(index); setFocus(null); setMobileTab("drawing");}} />}
            <RunActivity key={exampleIndex} run={run} drawing={drawing} connection="Recorded activity"
              onFocus={showRegion} onInspectCandidates={() => {setInspector("candidates"); setMobileTab("ask");}} />
          </div>
        </section>}
    </main>
    <footer className="replay-footer">Original pixels and recorded outputs. Playback changes presentation time only. No inference, uploads or changes to app history.</footer>
  </div>;
}
createRoot(document.getElementById("root")!).render(<ToolReplay />);
