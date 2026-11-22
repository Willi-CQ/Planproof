import { ArrowUpRight, ChevronDown } from "lucide-react";
import { CropPreview } from "./EvidenceInspector";
import { RasterEvidenceDetail } from "./RasterEvidenceDetail";
import { isAnswered, resultText, visibleCitations, type AgentResult, type Drawing } from "../lib/api";

interface Props {
  result: AgentResult;
  drawing: Drawing;
  messageId: string;
  rasterSource?: Blob;
  selectedEvidence: { message: string; index: number } | null;
  onHighlight: (message: string, index: number) => void;
}
export function AnswerBriefing({ result, drawing, messageId, selectedEvidence, onHighlight, rasterSource }: Props) {
  const citations = visibleCitations(result, drawing);
  const selected = selectedEvidence?.message === messageId ? citations[selectedEvidence.index] : undefined;
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
  const notes = [...new Set(result.answer?.uncertainties || [])].filter(note => !paragraphs.some(p => p.text === note));
  const moreInformation = paragraphs.length > 2 || notes.length > 0 || !!result.answer?.coverage || !!countEvidence || !!result.reader_input;
  const display = resultText(result).replace(/\b(m2|ft2|in2|cm2|mm2)\b/g, unit => unit.slice(0, -1) + "²");
  return <div className={`answer answer-briefing ${resolved ? "" : "answer-unresolved"}`} data-testid="agent-answer">
    <span className="briefing-eyebrow">{result.answer?.status === "partial" ? "Partial result" : resolved ? approximate ? "Estimated from the drawing" : "Your answer" : result.reader_reason === "unsupported_question" ? "Question outside reader scope" : "Evidence incomplete"}</span>
    <p className="answer-value" data-testid="answer-value">{approximate ? "≈ " : ""}{display}</p>
    <div className="briefing-prose">
      {paragraphs.length ? paragraphs.slice(0, 2).map((paragraph, i) => <p key={i} className={paragraph.section === "limitation" ? "briefing-limitation" : ""}>
        {paragraph.text}{paragraph.citations.map(reference)}
      </p>) : result.answer?.explanation && <p>{result.answer.explanation}{citations.map(c => reference(c.source_index!))}</p>}
    </div>
    {!!paragraphs.length && !!unlinked.length && <p className="additional-sources">Additional source evidence {unlinked.map(c => reference(c.source_index!))}</p>}
    {rasterSource && selected && <RasterEvidenceDetail source={rasterSource} drawing={drawing} bounds={selected.canonical_bbox} />}
    {moreInformation && <details className="answer-notes answer-information"><summary>More information<ChevronDown size={13} /></summary>
      <div className="answer-information-content">
      {paragraphs.slice(2).map((paragraph, i) => <p key={i}>{paragraph.text}{paragraph.citations.map(reference)}</p>)}
      {!!notes.length && <ul>{notes.map((note, i) => <li key={i}>{note}</li>)}</ul>}
      {result.answer?.coverage && <p className="answer-scope"><strong>Scope</strong><span>{result.answer.coverage}</span></p>}
      {result.reader_input && <p className="answer-scope reader-source-receipt">
        <strong>Submitted source</strong><span>{({original:"Original file", page:"Extracted PDF page", image:"Rendered page image"})[result.reader_input.representation]}
          {" · "}{result.reader_input.width} × {result.reader_input.height} {result.reader_input.coordinates === "image_pixels" ? "px" : "PDF points"}
          {" · page "}{result.reader_input.page}</span><span>SHA-256 {result.reader_input.sha256}</span>
      </p>}
      {countEvidence && <aside className="count-inventory" data-testid="count-inventory">
      <strong>{countEvidence.complete ? "Located door inventory" : "Provisional door inventory"}</strong>
      {countEvidence.objects.length ? <>
        <p>{countEvidence.supported} supported · {countEvidence.uncertain} uncertain</p>
        <p>Green: supported. Dashed amber: uncertain. Not human-verified.</p>
      </> : <p>No door locations returned. This does not mean there are no doors.</p>}
      </aside>}
      </div>
    </details>}
    {!resolved && result.answer?.status !== "partial" && !result.answer?.clarification_question && <p className="answer-help">{result.reader_reason === "unsupported_question" || result.reader_reason === "visual_count_required" || result.reader_reason === "geometry_on_large_raster"
      ? 'To locate printed text, try “Find the label \"North Hall\"” and replace the quoted words with a label on your drawing.'
      : result.reliability?.retry_after_seconds ? `Please wait ${result.reliability.retry_after_seconds} seconds before trying again.` : result.completion_reason === "inspection_incomplete" ? "The inspection did not establish a complete answer. This does not mean the drawing lacks the information." : "A clearer detail or a more specific question may help resolve this."}</p>}
    {!!detections.length && <section className="detection-evidence" aria-label="Detected features in the source">
      <h3>On the drawing <span>{detections.length}</span></h3>
      <div>{detections.map(c => {
        const index = citations.indexOf(c);
        return <button key={c.source_index} onClick={() => onHighlight(messageId, index)} className="detection-card">
          <CropPreview drawing={drawing} region={c} />
          <span><b>{c.source_index}</b>{detailFor(c.source_index!)?.label || "Located feature"}<ArrowUpRight size={13} /></span>
        </button>;
      })}</div>
    </section>}
  </div>;
}
