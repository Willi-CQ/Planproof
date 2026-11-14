import { useState } from "react";
import { Check, ChevronDown, Focus, LoaderCircle, CircleAlert } from "lucide-react";
import { CropPreview } from "./EvidenceInspector";
import { activityDetail, activityTitle, boundRegions, workSteps } from "../lib/activity";
import { durationSuffix, sameSource, terminal, type AgentRun, type Region } from "../lib/runs";
import type { Drawing } from "../lib/api";

interface Props { run: AgentRun; drawing: Drawing; onFocus: (region: Region) => void; connection?: string }
export function RunActivity({ run, drawing, onFocus, connection }: Props) {
  const done = terminal(run);
  const steps = workSteps(run);
  const [expanded, setExpanded] = useState(false);
  const shown = expanded || done ? steps : steps.slice(-5);
  const content = <>
    {!done && <div className="work-heading"><span className="live-dot" />{connection || "Working on your question"}</div>}
    {!steps.length && <p className="work-empty">Question queued. Waiting for the first execution event.</p>}
    {!done && steps.length > 5 && <button className="text-button earlier-steps" onClick={() => setExpanded(!expanded)}>{expanded ? "Show recent steps" : `Show ${steps.length - 5} earlier steps`}</button>}
    <ol className="work-log" aria-label="Work on this question">
      {shown.map((step, i) => {
        const running = !done && step.state === "running";
        const pending = step.state === "pending" || step.state === "busy";
        const interrupted = done && step.state === "running";
        const unavailable = step.state === "unavailable" || step.state === "partial" || interrupted || pending;
        const regions = sameSource(run.source, drawing.source) ? boundRegions(step, drawing.source) : [];
        return <li key={`${step.stage}-${step.sequence}`} className={running ? "work-current" : unavailable ? "work-unavailable" : ""}>
          <span className="work-icon">{running ? <LoaderCircle className="spinning" size={14} /> : unavailable ? <CircleAlert size={14} /> : <Check size={14} />}</span>
          <div className="work-content">
            <strong>{interrupted ? "Step interrupted" : activityTitle(step)}</strong>
            {(running || pending || expanded || done || i === shown.length - 1 || step.question || step.checks?.length) && <p>{activityDetail(step)}</p>}
            {step.advisory?.findings?.slice(0, 3).map((finding, j) => <p className="work-finding" key={j}>{finding}</p>)}
            {step.unresolved?.map((issue, j) => <p className="work-caution" key={j}>{issue}</p>)}
            {regions.length > 0 && <div className="work-crops">{regions.slice(0, 3).map((region, j) => <button key={j} onClick={() => onFocus(region)} aria-label={`Inspect work step ${step.sequence} source ${j + 1}`}>
              <CropPreview region={region} drawing={drawing} /><span><Focus size={12} />{step.state === "running" ? "Requested view" : "Source detail"}</span>
            </button>)}</div>}
            <small>{pending ? "Not supplied to this answer yet" : unavailable ? "No result established" : running ? "In progress" : step.state === "cached" ? "Reused evidence" : "Returned"}{durationSuffix(step.seconds)}</small>
          </div>
        </li>;
      })}
    </ol>
  </>;
  if (done) return <details className="work-recap"><summary><Check size={14} />{run.status === "completed" ? "Work behind this answer" : "Recorded work"}<span>{steps.length} steps</span><ChevronDown size={14} /></summary>{content}</details>;
  return <div className="run-activity" data-testid="live-work-log">{content}</div>;
}
