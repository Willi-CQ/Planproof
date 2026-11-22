import { useEffect, useRef, type ReactNode } from "react";
import {
  ArrowUp,
  Plus,
  ArrowUpRight,
  RotateCcw,
  CornerDownLeft,
  Download,
  LoaderCircle,
} from "lucide-react";
import { BrandMark } from "./BrandMark";
import type { AgentResult, Drawing } from "../lib/api";
import { terminal, type AgentRun, type Region } from "../lib/runs";
import { RunActivity } from "./RunActivity";
import { AnswerBriefing } from "./AnswerBriefing";
export interface Message {
  run?: AgentRun;
  id: string;
  role: "user" | "assistant";
  content: string;
  result?: AgentResult;
}
interface Props {
  draft: string;
  onDraftChange: (draft: string) => void;
  onNewDrawing: () => void;
  onDownload: () => void;
  latestTaskId?: string;
  downloading: boolean;
  downloadError?: string;
  onFocusWork: (messageId: string, region: Region) => void;
  onInspectCandidates: (messageId: string) => void;
  connection?: string;
  messages: Message[];
  drawing: Drawing;
  busy: boolean;
  disabled?: boolean;
  reader?: boolean;
  spatialSample?: boolean;
  canSend?: boolean;
  readerNotice?: ReactNode;
  rasterSource?: Blob;
  onSendMessage: (question: string) => Promise<boolean>;
  onClearChat: () => void;
  selectedEvidence: { message: string; index: number } | null;
  onHighlight: (message: string, index: number) => void;
}
export function PlanChatbotPanel({
  draft,
  onDraftChange: setDraft,
  onNewDrawing,
  onDownload, latestTaskId, downloading, downloadError,
  onFocusWork,
  onInspectCandidates,
  connection,
  messages,
  drawing,
  busy,
  disabled = false,
  reader = false,
  spatialSample = false,
  canSend = true,
  readerNotice,
  rasterSource,
  onSendMessage,
  onClearChat,
  selectedEvidence,
  onHighlight,
}: Props) {
  const scroll = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const followWork = useRef(true);
  const lastMessage = useRef<string | null>(null);
  const lastAnswer = useRef<string | null>(null);
  useEffect(() => {
    const pane = scroll.current;
    if (!pane) return;
    const latest = [...messages].reverse().find(m => m.role === "assistant");
    const newMessage = latest?.id !== lastMessage.current;
    const completed = latest?.result && latest.id !== lastAnswer.current;
    lastMessage.current = latest?.id || null;
    if (!latest) lastAnswer.current = null;
    const article = completed ? pane.querySelector<HTMLElement>(`[data-message-id="${CSS.escape(latest.id)}"]`) : null;
    if (completed) lastAnswer.current = latest.id;
    if (!completed && !newMessage && (!busy || !followWork.current)) return;
    pane.scrollTo({
      top: article ? pane.scrollTop + article.getBoundingClientRect().top - pane.getBoundingClientRect().top - 16 : pane.scrollHeight,
      behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "instant"
        : "smooth",
    });
  }, [messages, busy]);
  useEffect(() => {
    if (input.current) {
      input.current.style.height = "auto";
      input.current.style.height =
        Math.min(input.current.scrollHeight, 132) + "px";
    }
  }, [draft]);
  async function submit() {
    if (busy || disabled || !canSend || !draft.trim()) return;
    const sent = draft.trim();
    const completed = await onSendMessage(sent);
    if (completed) setDraft("");
  }
  function suggestion(text: string) {
    setDraft(text);
    input.current?.focus();
  }
  return (
    <section className="assistant-panel" aria-label="Planproof assistant">
      <header className="assistant-header">
        <div>
          <BrandMark small />
          <h2>Planproof</h2>
        </div>
        <div className="assistant-actions">
          <button
            className="subtle-button"
            aria-label="New drawing"
            disabled={busy || disabled}
            onClick={onNewDrawing}
          >
            <Plus size={15} />
            <span>New drawing</span>
          </button>
          <button
            className="icon-button"
            aria-label="Clear conversation"
            title="New conversation"
            disabled={!messages.length || busy || disabled}
            onClick={() => {
              onClearChat();
              setDraft("");
            }}
          >
            <RotateCcw size={16} />
          </button>
        </div>
      </header>
      <div
        ref={scroll}
        onScroll={() => {
          const pane = scroll.current!;
          followWork.current = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 80;
        }}
        className="conversation"
        aria-live="polite"
        data-testid="plan-chat-messages"
      >
        {!messages.length && (
          <div className="chat-welcome">
            {!reader && <span className="eyebrow">A little clarity goes a long way.</span>}
            <h3>
              What would you
              <br />
              like to know?
            </h3>
            <p>
              {reader ? "Search for a printed label in English or German. Replace the words in quotes with text from your drawing." : "Ask about a detail, a dimension, or the whole drawing."}
            </p>
            <div className="suggestions">
              {(spatialSample ? [
                'Find the label "HALLWAY".',
                'Wo steht „DEMO“?',
              ] : reader ? [
                'Find the label "North Hall".',
                'Wo steht „North Hall“?',
                "Where is door D-01?",
              ] : [
                "What is the drawing title?",
                "How many rooms are shown?",
                "Where are the stairs?",
              ]).map((text) => (
                <button key={text} onClick={() => suggestion(text)}>
                  {text}
                  <ArrowUpRight size={14} />
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((message) => (
          <article
            key={message.id}
            data-message-id={message.id}
            className={`message message-${message.role}`}
          >
            {message.role === "user" ? (
              <p className="user-message">{message.content}</p>
            ) : (
              <>
                <div className="assistant-label">
                  <BrandMark small />
                  <span>Planproof</span>
                </div>
                {message.result && (
                  <AnswerBriefing
                    result={message.result}
                    rasterSource={rasterSource}
                    drawing={drawing}
                    messageId={message.id}
                    selectedEvidence={selectedEvidence}
                    onHighlight={onHighlight}
                  />
                )}
                {message.run && !terminal(message.run) && <RunActivity run={message.run} drawing={drawing} connection={connection}
                  onInspectCandidates={() => onInspectCandidates(message.id)}
                  onFocus={(region) => onFocusWork(message.id, region)} />}
                {message.run && terminal(message.run) && !message.result && <p className="answer-help">{message.run.message || "This task could not finish. Its recorded steps are available in Task details."}</p>}
                {message.id === latestTaskId && message.run && terminal(message.run) && <div className="task-download">
                  <button className="inspect-button" onClick={onDownload} disabled={downloading}>
                    {downloading ? <LoaderCircle size={14} className="spinning" /> : <Download size={14} />}
                    {downloading ? "Preparing report…" : "Download task report"}
                    <span className="report-format">PDF</span>
                  </button>
                  {downloadError && <p role="alert" className="report-error">{downloadError}</p>}
                </div>}
              </>
            )}
          </article>
        ))}
        {busy && !messages.some(m => m.run && !m.result) && (
          <div role="status" className="thinking">
            <span className="thinking-dots">
              <i />
              <i />
              <i />
            </span>
            <span>
              {connection || "Taking a closer look"}
              <span className="thinking-note">
                Detailed drawings can take a few minutes.
              </span>
            </span>
          </div>
        )}
      </div>
      <div className="composer-area">
        {readerNotice && <div className="reader-upload-note">{readerNotice}</div>}
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <label htmlFor="question" className="visually-hidden">
            Question about this drawing
          </label>
          <textarea
            ref={input}
            id="question"
            rows={2}
            value={draft}
            maxLength={2000}
            disabled={busy || disabled}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Ask anything about this drawing…"
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.nativeEvent.isComposing
              ) {
                e.preventDefault();
                void submit();
              }
            }}
          />
          <div className="composer-bottom">
            <span className="composer-hint">
              {draft.length > 1800 ? (
                `${draft.length}/2000`
              ) : (
                <>
                  <CornerDownLeft size={11} /> to send
                </>
              )}
            </span>
            <button
              className="send-button"
              disabled={busy || disabled || !canSend || !draft.trim()}
              aria-label="Ask Planproof"
              title="Send question"
            >
              <ArrowUp size={19} />
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}
