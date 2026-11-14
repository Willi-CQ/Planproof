import { useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  ArrowUpRight,
  ChevronDown,
  RotateCcw,
  CornerDownLeft,
} from "lucide-react";
import { BrandMark } from "./BrandMark";
import {
  isAnswered,
  resultText,
  visibleCitations,
  type AgentResult,
  type Drawing,
} from "../lib/api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  result?: AgentResult;
}
interface Props {
  messages: Message[];
  drawing: Drawing;
  busy: boolean;
  disabled?: boolean;
  onSendMessage: (question: string) => Promise<boolean>;
  onClearChat: () => void;
  selectedEvidence: { message: string; index: number } | null;
  onHighlight: (message: string, index: number) => void;
}
function Answer({
  result,
  drawing,
  messageId,
  selectedEvidence,
  onHighlight,
}: { result: AgentResult; drawing: Drawing; messageId: string } & Pick<
  Props,
  "selectedEvidence" | "onHighlight"
>) {
  const citations = visibleCitations(result, drawing);
  const resolved = isAnswered(result);
  const hasNotes = !!(
    result.answer?.coverage || result.answer?.uncertainties?.length
  );
  return (
    <div
      className={`answer ${resolved ? "" : "answer-unresolved"}`}
      data-testid="agent-answer"
    >
      <p className="answer-value" data-testid="answer-value">
        {resultText(result)}
      </p>
      {result.answer?.explanation && (
        <p className="answer-explanation">{result.answer.explanation}</p>
      )}
      {citations.length > 0 && (
        <div className="citations">
          {citations.map((c, i) => (
            <button
              key={i}
              className="citation"
              aria-label={`View evidence ${i + 1}, page ${c.page_index + 1}`}
              onClick={() => onHighlight(messageId, i)}
              aria-pressed={
                selectedEvidence?.message === messageId &&
                selectedEvidence.index === i
              }
            >
              <span className="citation-number">{i + 1}</span>View on drawing
              <ArrowUpRight size={13} />
            </button>
          ))}
        </div>
      )}
      {hasNotes && (
        <details className="answer-notes">
          <summary>
            About this answer
            <ChevronDown size={13} />
          </summary>
          <p>{result.answer?.coverage}</p>
          <ul>
            {result.answer?.uncertainties?.map((u, i) => (
              <li key={i}>{u}</li>
            ))}
          </ul>
        </details>
      )}
      {!resolved && (
        <p className="answer-help">
          {result.reliability?.retry_after_seconds
            ? `Please wait ${result.reliability.retry_after_seconds} seconds before trying again.`
            : "Try a more specific question, or send it again."}
        </p>
      )}
    </div>
  );
}
export function PlanChatbotPanel({
  messages,
  drawing,
  busy,
  disabled = false,
  onSendMessage,
  onClearChat,
  selectedEvidence,
  onHighlight,
}: Props) {
  const [draft, setDraft] = useState("");
  const scroll = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    scroll.current?.scrollTo({
      top: scroll.current.scrollHeight,
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
    if (busy || disabled || !draft.trim()) return;
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
      </header>
      <div
        ref={scroll}
        className="conversation"
        aria-live="polite"
        data-testid="plan-chat-messages"
      >
        {!messages.length && (
          <div className="chat-welcome">
            <span className="eyebrow">A little clarity goes a long way.</span>
            <h3>
              What would you
              <br />
              like to know?
            </h3>
            <p>
              Ask about a detail, a dimension,
              <br />
              or the whole drawing.
            </p>
            <div className="suggestions">
              {[
                "What is the drawing title?",
                "How many rooms are shown?",
                "Where are the stairs?",
              ].map((text) => (
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
                  <Answer
                    result={message.result}
                    drawing={drawing}
                    messageId={message.id}
                    selectedEvidence={selectedEvidence}
                    onHighlight={onHighlight}
                  />
                )}
              </>
            )}
          </article>
        ))}
        {busy && (
          <div role="status" className="thinking">
            <span className="thinking-dots">
              <i />
              <i />
              <i />
            </span>
            <span>
              Taking a closer look
              <span className="thinking-note">
                Detailed drawings can take a few minutes.
              </span>
            </span>
          </div>
        )}
      </div>
      <div className="composer-area">
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
              disabled={busy || disabled || !draft.trim()}
              aria-label="Ask Planproof"
              title="Send question"
            >
              <ArrowUp size={19} />
            </button>
          </div>
        </form>
        <p className="processing-note">
          Sending lets Planproof’s AI services analyze this drawing.
        </p>
        <details className="chat-help">
          <summary>Good to know</summary>
          <p>
            Each question is answered independently against the loaded page.
            Include the room or detail you mean. Answers can be imperfect; check
            the highlighted source.
          </p>
        </details>
      </div>
    </section>
  );
}
