import { useState } from "react";
import { ChevronDown, Focus, LoaderCircle, CircleAlert, ScanSearch, Layers, Crop, Calculator, Workflow, ArrowDownLeft, Clock3 } from "lucide-react";
import { CropPreview } from "./EvidenceInspector";
import { activityDetail, activityTitle, boundRegions, eventTime, needsAttention, toolFamily, toolLabels, workSteps, inspectionColor, type WorkStep } from "../lib/activity";
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
  const [detailsOpen, setDetailsOpen] = useState(false);
  const active = steps.filter(s => s.state === "running" && !done);
  const inspections = active.filter(s => toolFamily(s.stage) === "nemotron");
  const attention = steps.filter(s => needsAttention(s, done)).length;
  const bound = sameSource(run.source, drawing.source);
  const label = inspections.length ? "Calling Nemotron"
    : active.some(s => s.stage === "get_algorithms") ? "Finding rooms"
    : active.some(s => s.stage === "inspect_regions") ? "Preparing detail checks"
    : active.some(s => toolFamily(s.stage) === "source") ? "Opening a detail"
    : active.some(s => toolFamily(s.stage) === "measurement") ? "Measuring"
    : steps.length ? "Reviewing results" : "Starting";
  const details = <details className={done ? "work-recap" : "activity-details"}
    onToggle={e => setDetailsOpen(e.currentTarget.open)}>
    <summary>{done ? <Workflow size={13} /> : null}{done ? "Activity" : "Details"}
      {done && attention > 0 && <span>{attention} checks need attention</span>}<ChevronDown size={13} />
    </summary>
    {detailsOpen && <>
      <ol className="work-log" aria-label="Work on this question">{steps.map(step => <StepCard
        key={`${step.stage}-${step.startSequence}`} step={step} done={done} drawing={drawing}
        bound={bound} onFocus={onFocus} onInspectCandidates={onInspectCandidates} />)}</ol>
      <p className="work-footnote">Inspection views are not detections. Returned observations may need verification.</p>
    </>}
  </details>;
  if (done) return details;
  return <div className="run-activity compact-activity" data-testid="live-work-log">
    <div className="compact-activity-heading" role="status"><LoaderCircle size={14} className="spinning" />
      <strong>{connection || label}</strong>{inspections.length > 1 && <span>{inspections.length} regions</span>}
    </div>
    {inspections.length > 0 && <div className="compact-inspections" aria-label="Active inspections">
      {inspections.flatMap(step => bound ? boundRegions(step, drawing.source).map((region, i) =>
        <button key={`${step.startSequence}-${i}`} onClick={() => onFocus(region)}
          style={{borderColor: inspectionColor(step.detailNumber), color: inspectionColor(step.detailNumber)}}
          aria-label={`Show Nemotron ${step.detailNumber} view ${i + 1}`}>
          <Focus size={12} />{step.detailNumber}{boundRegions(step, drawing.source).length > 1 ? `.${i + 1}` : ""}
        </button>) : [])}
    </div>}
    {details}
  </div>;
}
