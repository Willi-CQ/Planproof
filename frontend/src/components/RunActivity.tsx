import { useState } from "react";
import { ChevronDown, Focus, LoaderCircle, CircleAlert, ScanSearch, Layers, Crop, Calculator, Workflow, ArrowDownLeft, Clock3 } from "lucide-react";
import { CropPreview } from "./EvidenceInspector";
import { activityDetail, activityTitle, boundRegions, eventTime, needsAttention, toolFamily, toolLabels, workSteps, type WorkStep } from "../lib/activity";
import { durationSuffix, sameSource, terminal, type AgentRun, type Region } from "../lib/runs";
import type { Drawing } from "../lib/api";

const icons = {nemotron: ScanSearch, algorithms: Layers, source: Crop, measurement: Calculator, coordinator: Workflow};
function RequestText({text}: {text: string}) {
  if (text.length <= 240) return <p className="work-request">{text}</p>;
  return <details className="work-request"><summary>{text.slice(0, 210)}… <span>Read full task</span></summary><p>{text}</p></details>;
}
function StepCard({step, done, drawing, bound, onFocus, onInspectCandidates}: {step: WorkStep; done: boolean; drawing: Drawing; bound: boolean; onFocus: (region: Region) => void; onInspectCandidates: () => void}) {
  const family = toolFamily(step.stage), Icon = icons[family];
  const running = !done && step.state === "running";
  const pending = step.state === "pending" || step.state === "busy";
  const interrupted = done && step.state === "running";
  const attention = needsAttention(step, done);
  const reused = step.cached || step.state === "cached";
  const unavailable = step.state === "unavailable" || interrupted;
  const status = interrupted ? "Interrupted" : running ? "Running" : step.state === "partial" ? "Partial result"
    : unavailable ? "Unavailable" : pending ? "Pending at return" : reused ? "Reused" : "Returned";
  const regions = bound ? boundRegions(step, drawing.source) : [];
  const findings = step.advisory?.findings || [];
  const start = eventTime(step.startedAt), end = eventTime(step.returnedAt);
  const count = new Set(step.delivered_ids || []).size;
  return <li className={`${running ? "work-current" : ""} ${attention || pending ? "work-unavailable" : ""}`} data-tool={family}>
    <span className="work-icon"><Icon size={15} /></span>
    <div className="work-content">
      <div className="work-card-heading"><span className="work-tool">{toolLabels[family]}{step.detailNumber ? ` · Detail ${step.detailNumber}` : ""}</span>
        <span className={`work-status ${running ? "is-running" : attention || pending ? "is-caution" : ""}`}>
          {running ? <LoaderCircle className="spinning" size={11} /> : attention ? <CircleAlert size={11} /> : <ArrowDownLeft size={11} />}{status}
        </span>
      </div>
      <strong>{interrupted ? "Step interrupted" : done && pending ? "Algorithm evidence was not ready for this request" : activityTitle(step)}</strong>
      <RequestText text={activityDetail(step)} />
      {family === "algorithms" && <div className="algorithm-roles">
        <div><ScanSearch size={14} /><span><b>ML detection</b>Proposes rooms and symbols from the image.</span></div>
        <div><Layers size={14} /><span><b>Deterministic rules</b>Check pixel and geometry cues; some depend on ML proposals.</span></div>
        <p>{pending || running ? "Combined evidence request. No candidates returned by this request yet." : unavailable
          ? "No usable evidence returned from this request." : Array.isArray(step.delivered_ids)
            ? `${count} evidence ${count === 1 ? "item" : "items"} supplied by this request. Candidates, not verified objects.`
            : "Evidence returned. Item counts were not recorded for this request."}</p>
        {bound && !pending && !running && !unavailable && <button className="text-button work-candidates" onClick={onInspectCandidates}>Explore raw ML + rule candidates</button>}
      </div>}
      {regions.length > 0 && <div className="work-crops">{regions.map((region, j) => {
        const size = region.sent_size;
        const resolution = Array.isArray(size) && size.length === 2 && size.every(n => Number.isFinite(n) && n > 0)
          ? `Sent ${size[0]} × ${size[1]} px` : "Source region";
        return <button key={j} onClick={() => onFocus(region)} aria-label={`Inspect work step ${step.startSequence} source ${j + 1}`}>
          <CropPreview region={region} drawing={drawing} />
          <span><Focus size={12} />{step.state === "running" ? "Requested view" : "Source detail"}{regions.length > 1 ? ` ${j + 1}` : ""}</span>
          <small>{region.image_id && <>{region.image_id} · </>}{resolution} · page {region.page_index + 1}</small>
        </button>;
      })}</div>}
      {family === "nemotron" && step.stage !== "inspect_regions" && regions.length > 0 && <p className="work-scope">Highlighted regions show the requested view, not detected objects.</p>}
      {findings.length > 0 && <div className="work-findings"><span className="work-section-label">Reported findings{attention ? " · uncertainty remains" : ""}</span>
        {findings.slice(0, 2).map((finding, j) => <p className="work-finding" key={j}>{finding}</p>)}
        {findings.length > 2 && <details><summary>Show {findings.length - 2} more findings</summary>{findings.slice(2).map((finding, j) => <p className="work-finding" key={j}>{finding}</p>)}</details>}
      </div>}
      {attention && !unavailable && step.advisory && <p className="work-caution">This check returned an uncertain or disputed observation. It is not a verified finding.</p>}
      {step.unresolved?.map((issue, j) => <p className="work-caution" key={j}>{issue}</p>)}
      {unavailable && <p className="work-caution">This step did not establish a result. Other checks may continue.</p>}
      <small>{pending ? "No evidence supplied by this request" : unavailable ? "No result established" : running ? "In progress" : reused ? "Reused evidence" : "Returned"}{durationSuffix(step.seconds)}</small>
      {(start || end) && <div className="work-timing"><Clock3 size={11} />{start && <span>Started {start}</span>}{end && <span>Returned {end}</span>}</div>}
    </div>
  </li>;
}

interface Props { run: AgentRun; drawing: Drawing; onFocus: (region: Region) => void; onInspectCandidates: () => void; connection?: string }
export function RunActivity({ run, drawing, onFocus, onInspectCandidates, connection }: Props) {
  const done = terminal(run), steps = workSteps(run);
  const [expanded, setExpanded] = useState(false);
  const active = steps.filter(s => s.state === "running" && !done);
  const returned = steps.filter(s => ["completed", "cached"].includes(s.state)).length;
  const attention = steps.filter(s => needsAttention(s, done)).length;
  const recent = new Set(steps.slice(-3).map(s => s.startSequence));
  const shown = expanded || done ? steps : steps.filter(s => recent.has(s.startSequence) || s.state === "running" || needsAttention(s, done) || s.state === "pending" || s.state === "busy");
  const families = [...new Set(steps.map(s => toolFamily(s.stage)))];
  const content = <>
    {!done && <div className="work-heading"><span className="live-dot" /><strong>{connection || "Agent activity"}</strong><span>{active.length} active steps · {returned} returned</span></div>}
    {!steps.length && <p className="work-empty">Question queued. Waiting for the first execution event.</p>}
    {families.length > 0 && <div className="work-families" aria-label="Tools used in this task">{families.map(family => {
      const Icon = icons[family];
      const count = active.filter(s => toolFamily(s.stage) === family).length;
      return <span key={family} data-active={count > 0}><Icon size={12} />{toolLabels[family]}{count > 0 && <b>{count} active</b>}</span>;
    })}</div>}
    {!done && steps.length > shown.length && <button className="text-button earlier-steps" onClick={() => setExpanded(true)}>Show {steps.length - shown.length} earlier steps</button>}
    {!done && expanded && steps.length > 3 && <button className="text-button earlier-steps" onClick={() => setExpanded(false)}>Show active and recent steps</button>}
    <ol className="work-log" aria-label="Work on this question">{shown.map(step => <StepCard key={`${step.stage}-${step.startSequence}`} step={step} done={done} drawing={drawing} bound={sameSource(run.source, drawing.source)} onFocus={onFocus} onInspectCandidates={onInspectCandidates} />)}</ol>
    <p className="work-footnote">Times are relative to this task. A returned tool result is evidence to assess, not a verified answer.</p>
  </>;
  if (done) return <details className="work-recap"><summary><Workflow size={14} />Tools and source checks<span>{steps.length} steps{attention > 0 ? ` · ${attention} need attention` : ""}</span><ChevronDown size={14} /></summary>{content}</details>;
  return <div className="run-activity" data-testid="live-work-log">{content}</div>;
}
