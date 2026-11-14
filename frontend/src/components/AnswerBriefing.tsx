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
  const countEvidence = result.count_evidence;
  const visibleLimit = !countEvidence && !paragraphs.slice(0, 2).some(p => p.section === "limitation") ? result.answer?.uncertainties?.[0] : undefined;
  const notes = result.answer?.uncertainties?.slice(visibleLimit ? 1 : 0) || [];
  const limitations = <ul>{notes.map((u, i) => <li key={i}>{u}</li>)}</ul>;
  const display = resultText(result).replace(/\b(m2|ft2|in2|cm2|mm2)\b/g, unit => unit.slice(0, -1) + "²");
  return <div className={`answer answer-briefing ${resolved ? "" : "answer-unresolved"}`} data-testid="agent-answer">
    <span className="briefing-eyebrow">{resolved ? approximate ? "Estimated from the drawing" : "Your answer" : "Evidence incomplete"}</span>
    <p className="answer-value" data-testid="answer-value">{approximate ? "≈ " : ""}{display}</p>
    <div className="briefing-prose">
      {paragraphs.length ? paragraphs.slice(0, 2).map((paragraph, i) => <p key={i} className={paragraph.section === "limitation" ? "briefing-limitation" : ""}>
        {paragraph.text}{paragraph.citations.map(reference)}
      </p>) : result.answer?.explanation && <p>{result.answer.explanation}{citations.map(c => reference(c.source_index!))}</p>}
    </div>
    {paragraphs.length > 2 && <details className="answer-notes"><summary>More detail</summary>
      {paragraphs.slice(2).map((paragraph, i) => <p key={i}>{paragraph.text}{paragraph.citations.map(reference)}</p>)}
    </details>}
    {!!paragraphs.length && !!unlinked.length && <p className="additional-sources">Additional source evidence {unlinked.map(c => reference(c.source_index!))}</p>}
    {visibleLimit && <p className="briefing-limitation">{visibleLimit}</p>}
    {!!notes.length && <details className="briefing-limits"><summary>Verification notes · {notes.length}</summary>{limitations}</details>}
    {countEvidence && <aside className="briefing-limits" data-testid="count-inventory">
      <strong>{countEvidence.complete ? "Located door inventory" : "Provisional door inventory"}</strong>
      {countEvidence.objects.length ? <>
        <p>{countEvidence.supported} supported · {countEvidence.uncertain} uncertain</p>
        <p>Green: supported. Dashed amber: uncertain. Not human-verified.</p>
      </> : <p>No door locations returned. This does not mean there are no doors.</p>}
    </aside>}
    {!!detections.length && <details className="detection-evidence" aria-label="Detected features in the source">
      <summary>{detections.length} locations highlighted on the drawing</summary>
      <div>{detections.slice(0, countEvidence ? 3 : detections.length).map(c => {
        const index = citations.indexOf(c);
        return <button key={c.source_index} onClick={() => onHighlight(messageId, index)} className="detection-card">
          <CropPreview drawing={drawing} region={c} />
          <span><b>{c.source_index}</b>{detailFor(c.source_index!)?.label || "Located feature"}<ArrowUpRight size={13} /></span>
        </button>;
      })}</div>
    </details>}
    {result.answer?.coverage && <details className="answer-notes"><summary>Scope of this answer</summary><p>{result.answer.coverage}</p></details>}
    {!resolved && !result.answer?.clarification_question && <p className="answer-help">{result.reliability?.retry_after_seconds ? `Please wait ${result.reliability.retry_after_seconds} seconds before trying again.` : result.completion_reason === "inspection_incomplete" ? "The inspection did not establish a complete answer. This does not mean the drawing lacks the information." : "A clearer detail or a more specific question may help resolve this."}</p>}
  </div>;
}
