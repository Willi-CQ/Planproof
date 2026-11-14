import { ArrowUpRight } from "lucide-react";
import { CropPreview } from "./EvidenceInspector";
import { isAnswered, resultText, visibleCitations, type AgentResult, type Drawing } from "../lib/api";

interface Props {
  result: AgentResult;
  drawing: Drawing;
  messageId: string;
  selectedEvidence: { message: string; index: number } | null;
  onHighlight: (message: string, index: number) => void;
}
export function AnswerBriefing({ result, drawing, messageId, selectedEvidence, onHighlight }: Props) {
  const citations = visibleCitations(result, drawing);
  const resolved = isAnswered(result);
  const details = result.answer?.citation_details || [];
  const paragraphs = result.answer?.briefing || [];
  const referenced = new Set(paragraphs.flatMap(p => p.citations));
  const detailFor = (source: number) => details.find(d => d.citation === source);
  function reference(source: number) {
    const index = citations.findIndex(c => c.source_index === source);
    if (index < 0) return null;
    const c = citations[index];
    return <button key={source} className="inline-citation" title={detailFor(source)?.label || `Source ${source}`}
      aria-label={`View evidence ${source}, page ${c.page_index + 1}`}
      aria-pressed={selectedEvidence?.message === messageId && selectedEvidence.index === index}
      onClick={() => onHighlight(messageId, index)}>{source}</button>;
  }
  const detections = citations.filter(c => detailFor(c.source_index!)?.kind === "detection");
  const unlinked = citations.filter(c => !referenced.has(c.source_index!));
  const approximate = result.measurement?.approximate === true;
  const limitations = <ul>{result.answer?.uncertainties?.map((u, i) => <li key={i}>{u}</li>)}</ul>;
  const display = resultText(result).replace(/\b(m2|ft2|in2|cm2|mm2)\b/g, unit => unit.slice(0, -1) + "²");
  return <div className={`answer answer-briefing ${resolved ? "" : "answer-unresolved"}`} data-testid="agent-answer">
    <span className="briefing-eyebrow">{resolved ? approximate ? "Estimated from the drawing" : "Your answer" : "Evidence incomplete"}</span>
    <p className="answer-value" data-testid="answer-value">{approximate ? "≈ " : ""}{display}</p>
    <div className="briefing-prose">
      {paragraphs.length ? paragraphs.map((paragraph, i) => <p key={i} className={paragraph.section === "limitation" ? "briefing-limitation" : ""}>
        {paragraph.text}{paragraph.citations.map(reference)}
      </p>) : result.answer?.explanation && <p>{result.answer.explanation}{citations.map(c => reference(c.source_index!))}</p>}
    </div>
    {!!paragraphs.length && !!unlinked.length && <p className="additional-sources">Additional source evidence {unlinked.map(c => reference(c.source_index!))}</p>}
    {!!result.answer?.uncertainties?.length && (paragraphs.some(p => p.section === "limitation")
      ? <details className="briefing-limits"><summary>Verification notes · {result.answer.uncertainties.length}</summary>{limitations}</details>
      : <aside className="briefing-limits"><strong>What to keep in mind</strong>{limitations}</aside>)}
    {!!detections.length && <div className="detection-evidence" aria-label="Detected features in the source">
      <h3>Located on your drawing</h3>
      <div>{detections.map(c => {
        const index = citations.indexOf(c);
        return <button key={c.source_index} onClick={() => onHighlight(messageId, index)} className="detection-card">
          <CropPreview drawing={drawing} region={c} />
          <span><b>{c.source_index}</b>{detailFor(c.source_index!)?.label || "Located feature"}<ArrowUpRight size={13} /></span>
        </button>;
      })}</div>
    </div>}
    {result.answer?.coverage && <details className="answer-notes"><summary>Scope of this answer</summary><p>{result.answer.coverage}</p></details>}
    {!resolved && <p className="answer-help">{result.reliability?.retry_after_seconds ? `Please wait ${result.reliability.retry_after_seconds} seconds before trying again.` : "A clearer detail or a more specific question may help resolve this."}</p>}
  </div>;
}
