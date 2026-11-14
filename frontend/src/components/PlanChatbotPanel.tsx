import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Compass,
  MapPin,
  RotateCcw,
  SendHorizontal,
  X,
} from "lucide-react";
import {
  isAnswered,
  resultText,
  visibleCitations,
  type AgentResult,
  type Drawing,
  type Permissions,
} from "../lib/api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  result?: AgentResult;
}
interface Props {
  messages: Message[];
  drawing: Drawing | null;
  name?: string;
  busy: "upload" | "ask" | null;
  error: string | null;
  onSendMessage: (question: string) => Promise<void>;
  onClearChat: () => void;
  onDismissError: () => void;
  permissions: Permissions;
  onPermissionsChange: (value: Permissions) => void;
  selectedEvidence: { message: string; index: number } | null;
  onHighlight: (message: string, index: number) => void;
}
function Answer({
  result,
  drawing,
  messageId,
  selectedEvidence,
  onHighlight,
}: { result: AgentResult; drawing: Drawing | null; messageId: string } & Pick<
  Props,
  "selectedEvidence" | "onHighlight"
>) {
  const citations = drawing ? visibleCitations(result, drawing) : [];
  const recovered =
    result.reliability?.recovery_succeeded ||
    result.reliability?.token_repair_succeeded;
  return (
    <div className="space-y-3" data-testid="agent-answer">
      <div
        className={`text-xs font-medium ${isAnswered(result) ? "text-blue-700" : "text-amber-700"}`}
      >
        {isAnswered(result) ? "Answer · verify on source" : "Unresolved"}
      </div>
      <p
        className="whitespace-pre-wrap break-words font-display text-xl font-semibold text-slate-900"
        data-testid="answer-value"
      >
        {resultText(result)}
      </p>
      {result.answer?.explanation && (
        <p className="whitespace-pre-wrap break-words">
          {result.answer.explanation}
        </p>
      )}
      {recovered && (
        <p className="rounded bg-blue-50 px-2 py-1 text-xs text-blue-800">
          Completed after provider recovery.
        </p>
      )}
      {!!result.reliability?.retry_after_seconds && !isAnswered(result) && (
        <p className="text-xs">
          Wait at least {result.reliability.retry_after_seconds} seconds before
          trying again.
        </p>
      )}
      {citations.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {citations.map((c, i) => (
            <button
              key={i}
              onClick={() => onHighlight(messageId, i)}
              aria-pressed={
                selectedEvidence?.message === messageId &&
                selectedEvidence.index === i
              }
              className="inline-flex items-center gap-1 rounded border border-blue-200 bg-white px-2 py-1 text-xs font-medium text-blue-800 aria-pressed:bg-blue-800 aria-pressed:text-white"
            >
              <MapPin size={12} />
              Evidence {i + 1} · p{c.page_index + 1}
            </button>
          ))}
        </div>
      )}
      <div
        className="flex flex-wrap gap-1.5 text-[11px] text-slate-600"
        data-testid="actual-tools"
      >
        {typeof result.algorithm_used === "boolean" && (
          <span className="rounded bg-slate-200/60 px-2 py-1">
            {result.algorithm_used ? "Algorithms used" : "Algorithms not used"}
          </span>
        )}
        {typeof result.nemotron_calls === "number" && (
          <span className="rounded bg-slate-200/60 px-2 py-1">
            {result.nemotron_calls} Nemotron calls
          </span>
        )}
        {typeof result.reflection_calls === "number" && (
          <span className="rounded bg-slate-200/60 px-2 py-1">
            {result.reflection_calls} reflections
          </span>
        )}
        {typeof result.seconds === "number" && (
          <span className="rounded bg-slate-200/60 px-2 py-1">
            {result.seconds.toFixed(1)}s
          </span>
        )}
      </div>
      {(result.answer?.coverage || result.answer?.uncertainties?.length) && (
        <details className="border-t border-slate-200 pt-2 text-xs">
          <summary className="cursor-pointer font-medium">
            Coverage and uncertainty
          </summary>
          <p className="mt-2 whitespace-pre-wrap">{result.answer?.coverage}</p>
          <ul className="mt-2 list-disc space-y-1 pl-4">
            {result.answer?.uncertainties?.map((u, i) => (
              <li key={i}>{u}</li>
            ))}
          </ul>
        </details>
      )}
      {!isAnswered(result) && (
        <p className="text-xs text-slate-500">
          You can ask again explicitly or clarify the question. No automatic
          resubmission.
        </p>
      )}
    </div>
  );
}
export function PlanChatbotPanel({
  messages,
  drawing,
  name,
  busy,
  error,
  onSendMessage,
  onClearChat,
  onDismissError,
  permissions,
  onPermissionsChange,
  selectedEvidence,
  onHighlight,
}: Props) {
  const [draft, setDraft] = useState("");
  const scroll = useRef<HTMLDivElement>(null);
  const canAsk = !!drawing && permissions.allow_gemini && !busy;
  useEffect(() => {
    scroll.current?.scrollTo({
      top: scroll.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, busy]);
  return (
    <section
      aria-label="Planproof drawing assistant"
      className="flex flex-col overflow-hidden rounded-lg border border-slate-200 bg-white md:h-full"
    >
      <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
        <div className="min-w-0">
          <h2 className="truncate font-display text-base font-semibold">
            Ask your drawing
          </h2>
          <p className="truncate text-xs text-slate-500">
            {drawing ? name : "Evidence you can inspect"}
          </p>
        </div>
        <button
          onClick={() => {
            onClearChat();
            setDraft("");
          }}
          disabled={!messages.length || !!busy}
          aria-label="Clear conversation"
          className="viewer-button gap-1 text-xs"
        >
          <RotateCcw size={14} />
          Clear
        </button>
      </div>
      <div
        ref={scroll}
        aria-live="polite"
        data-testid="plan-chat-messages"
        className="min-h-[240px] flex-1 space-y-4 overflow-y-auto px-4 py-4"
      >
        {!messages.length && (
          <div className="flex min-h-[220px] h-full flex-col items-center justify-center px-3 text-center">
            <Compass size={30} className="text-slate-400" />
            <p className="mt-3 font-display text-lg font-semibold">
              From question to visible evidence
            </p>
            <p className="mt-2 max-w-sm text-sm leading-relaxed text-slate-500">
              Ask about a count, a label, or a detail. The agent chooses whether
              to inspect the drawing, use detection, or check a closer view.
            </p>
          </div>
        )}
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex flex-col ${message.role === "user" ? "items-end" : "items-start"}`}
          >
            <span className="mb-1 text-xs text-slate-500">
              {message.role === "user" ? "You" : "Planproof"}
            </span>
            <div
              className={
                message.role === "user"
                  ? "max-w-[92%] whitespace-pre-wrap break-words rounded-lg bg-[#0F172A] px-3.5 py-2.5 text-sm text-white"
                  : "w-full rounded-lg border border-slate-200 bg-[#F8FAFC] px-3.5 py-3 text-sm leading-relaxed text-slate-800"
              }
            >
              {message.result ? (
                <Answer
                  result={message.result}
                  drawing={drawing}
                  messageId={message.id}
                  selectedEvidence={selectedEvidence}
                  onHighlight={onHighlight}
                />
              ) : (
                message.content
              )}
            </div>
          </div>
        ))}
        {busy && (
          <div
            role="status"
            className="rounded-lg border border-slate-200 bg-[#F8FAFC] p-4 text-sm"
          >
            <span className="mr-2 inline-block h-2 w-2 animate-pulse rounded-full bg-blue-700" />
            {busy === "ask"
              ? "Inspecting your question… This may take several minutes."
              : "Uploading the original drawing…"}
            {busy === "ask" && (
              <p className="mt-1 text-xs text-slate-500">
                Tool activity and evidence appear when the response is complete.
              </p>
            )}
          </div>
        )}
      </div>
      {error && (
        <div
          role="alert"
          className="mx-4 mb-2 flex items-start gap-2 rounded border border-red-200 bg-red-50 p-3 text-xs text-red-800"
        >
          <AlertCircle size={16} className="shrink-0" />
          <span className="flex-1">{error}</span>
          <button onClick={onDismissError} aria-label="Dismiss error">
            <X size={14} />
          </button>
        </div>
      )}
      <div className="shrink-0 space-y-3 border-t border-slate-200 bg-[#F8FAFC] p-4">
        <fieldset
          disabled={!!busy}
          className="flex flex-wrap gap-x-4 gap-y-2 text-xs text-slate-600"
        >
          <legend className="mb-2 text-xs font-medium text-slate-700">
            Allow this drawing to be sent to
          </legend>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={permissions.allow_gemini}
              onChange={(e) =>
                onPermissionsChange({
                  ...permissions,
                  allow_gemini: e.target.checked,
                })
              }
            />
            Gemini (answer agent)
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={permissions.allow_nemotron}
              onChange={(e) =>
                onPermissionsChange({
                  ...permissions,
                  allow_nemotron: e.target.checked,
                })
              }
            />
            Nemotron (detail crops)
          </label>
        </fieldset>
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (canAsk && draft.trim()) void onSendMessage(draft.trim());
          }}
        >
          <label htmlFor="question" className="sr-only">
            Question about this drawing
          </label>
          <input
            id="question"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            maxLength={2000}
            disabled={!!busy}
            placeholder="How many…? Where is…?"
            className="min-h-11 min-w-0 flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm focus:border-blue-700 focus:outline-none"
          />
          <button
            disabled={!canAsk || !draft.trim()}
            aria-label="Ask drawing"
            className="inline-flex min-h-11 shrink-0 items-center gap-2 rounded-md bg-[#1E40AF] px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            <span>Ask</span>
            <SendHorizontal size={16} />
          </button>
        </form>
        <p className="text-[11px] text-slate-500">
          Each question independently inspects the loaded page. Earlier messages
          are displayed for your reference.
        </p>
      </div>
    </section>
  );
}
