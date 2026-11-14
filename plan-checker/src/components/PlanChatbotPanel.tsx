import { useRef, useState, type FormEvent, type RefObject } from 'react';
import {
  AlertCircle,
  Compass,
  MapPin,
  RotateCcw,
  SendHorizontal,
  X,
} from 'lucide-react';
import type { PlanChatMessage, PlanSheet, UploadedPlanFile } from '../lib/plansData';
import { parsePlanMessage, type InlineRun } from '../lib/stoaEngine';

interface PlanChatbotPanelProps {
  plan: PlanSheet;
  uploadedPlan: UploadedPlanFile | null;
  messages: PlanChatMessage[];
  onSendMessage: (query: string) => Promise<void>;
  onClearChat: () => void;
  selectedZoneId: string | null;
  onHighlightZone: (zoneId: string) => void;
  isLoading: boolean;
  errorMessage: string | null;
  onDismissError: () => void;
  inputRef: RefObject<HTMLInputElement | null>;
}

export function PlanChatbotPanel({
  plan,
  uploadedPlan,
  messages,
  onSendMessage,
  onClearChat,
  selectedZoneId,
  onHighlightZone,
  isLoading,
  errorMessage,
  onDismissError,
  inputRef,
}: PlanChatbotPanelProps) {
  const [draft, setDraft] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = draft.trim();
    if (!trimmed || isLoading) return;
    setDraft('');
    await onSendMessage(trimmed);
    requestAnimationFrame(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    });
  };

  const handleStarter = async (promptText: string) => {
    if (isLoading) return;
    setDraft('');
    await onSendMessage(promptText);
    requestAnimationFrame(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    });
  };

  return (
    <section
      aria-label="Stoa 2D Plan Assistant"
      className="flex flex-col rounded-lg border border-slate-200 bg-white overflow-hidden md:h-full"
    >
      {/* Chat Header */}
      <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
        <div className="min-w-0">
          <h2 className="truncate font-display text-base font-semibold text-slate-900">
            {uploadedPlan ? `Ask about ${uploadedPlan.name}` : `Ask Stoa · ${plan.code}`}
          </h2>
          <p className="truncate text-xs text-slate-500">
            {uploadedPlan
              ? 'Custom uploaded 2D sheet · Visual drawing inspection'
              : `${plan.project} · ${plan.zones.length} tagged zones indexed`}
          </p>
        </div>

        <button
          type="button"
          onClick={onClearChat}
          disabled={messages.length === 0 || isLoading}
          aria-label="Clear chat"
          className="inline-flex min-h-9 items-center gap-1.5 rounded border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-600 whitespace-nowrap shrink-0 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
        >
          <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
          <span>Clear</span>
        </button>
      </div>

      {/* Conversation Stream */}
      <div
        ref={scrollRef}
        aria-live="polite"
        data-testid="plan-chat-messages"
        className="flex-1 space-y-4 overflow-y-auto px-4 py-4 min-h-[280px] bg-white"
      >
        {messages.length === 0 ? (
          <div className="flex h-full min-h-[220px] flex-col items-center justify-center px-4 text-center">
            <Compass className="h-7 w-7 text-slate-400" aria-hidden="true" />
            <p className="mt-3 font-display text-base font-semibold text-slate-900">
              Ask about any dimension, tag, or detail
            </p>
            <p className="mt-1 max-w-sm text-xs leading-relaxed text-slate-500">
              {uploadedPlan
                ? 'Ask Stoa to read dimensions, room tags, or notes on your uploaded 2D drawing.'
                : 'Select a zone on the 2D drawing or ask a question below. Citations highlight directly on the plan.'}
            </p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${
                msg.role === 'user' ? 'items-end' : 'items-start'
              }`}
            >
              <div className="mb-1 flex items-center gap-1.5 text-xs text-slate-400 tabular-nums">
                <span className="font-medium text-slate-600">
                  {msg.role === 'user' ? 'You' : 'Stoa'}
                </span>
                <span aria-hidden="true">·</span>
                <span>{msg.timestamp}</span>
              </div>

              {msg.role === 'user' ? (
                <div className="max-w-[88%] rounded-lg bg-[#0F172A] px-3.5 py-2.5 text-sm leading-relaxed text-white">
                  {msg.content}
                </div>
              ) : (
                <div className="max-w-[94%] space-y-2 rounded-lg border border-slate-200 bg-[#F8FAFC] px-3.5 py-3 text-sm leading-relaxed text-slate-800">
                  {parsePlanMessage(msg.content, uploadedPlan ? [] : plan.zones).map(
                    (block, blockIdx) => {
                      const renderRuns = (runs: InlineRun[]) =>
                        runs.map((run, runIdx) => {
                          if (run.type === 'text') {
                            return run.bold ? (
                              <strong key={runIdx} className="font-semibold text-slate-900">
                                {run.text}
                              </strong>
                            ) : (
                              <span key={runIdx}>{run.text}</span>
                            );
                          }
                          const { citation } = run;
                          const isActive = selectedZoneId === citation.zone?.id;
                          return (
                            <button
                              key={runIdx}
                              type="button"
                              onClick={() => {
                                if (citation.zone) {
                                  onHighlightZone(citation.zone.id);
                                }
                              }}
                              title={`Highlight ${citation.label} on 2D plan`}
                              className={`ml-1.5 mr-0.5 inline-flex items-center gap-1 rounded border px-1.5 py-0.5 align-baseline font-mono text-xs font-medium tabular-nums transition-colors cursor-pointer ${
                                isActive
                                  ? 'border-[#1E40AF] bg-[#1E40AF] text-white'
                                  : 'border-[#1E40AF]/30 bg-white text-[#1E40AF] hover:border-[#1E40AF] hover:bg-[#1E40AF]/8'
                              }`}
                            >
                              <MapPin className="h-3 w-3 shrink-0" aria-hidden="true" />
                              <span>{citation.label}</span>
                            </button>
                          );
                        });

                      if (block.type === 'bullet') {
                        return (
                          <div key={blockIdx} className="flex items-start gap-2 pl-0.5">
                            <span
                              aria-hidden="true"
                              className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-slate-400"
                            />
                            <p className="flex-1">{renderRuns(block.runs)}</p>
                          </div>
                        );
                      }

                      return <p key={blockIdx}>{renderRuns(block.runs)}</p>;
                    }
                  )}
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div
            aria-label="Stoa is checking the 2D plan"
            className="max-w-[85%] space-y-2 rounded-lg border border-slate-200 bg-[#F8FAFC] p-3.5"
          >
            <div className="h-3 w-1/3 animate-pulse rounded bg-slate-200" />
            <div className="h-3 w-5/6 animate-pulse rounded bg-slate-200" />
            <div className="h-3 w-2/3 animate-pulse rounded bg-slate-200" />
          </div>
        )}
      </div>

      {/* Error Alert */}
      {errorMessage && (
        <div
          role="alert"
          className="mx-4 mb-2 flex items-center justify-between gap-2 rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800"
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4 shrink-0 text-red-700" aria-hidden="true" />
            <span>{errorMessage}</span>
          </div>
          <button
            type="button"
            onClick={onDismissError}
            aria-label="Dismiss error"
            className="rounded p-1 text-red-700 hover:bg-red-100 cursor-pointer"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* Starter Prompts & Input */}
      <div className="border-t border-slate-200 bg-[#F8FAFC] p-4">
        {!uploadedPlan && (
          <div className="mb-2.5 flex flex-wrap gap-1.5">
            {plan.starterPrompts.map((promptText) => (
              <button
                key={promptText}
                type="button"
                disabled={isLoading}
                onClick={() => handleStarter(promptText)}
                className="rounded border border-slate-200 bg-white px-2.5 py-1 text-left text-xs font-medium text-slate-700 hover:border-[#1E40AF]/50 hover:text-[#1E40AF] disabled:opacity-50 transition-colors cursor-pointer"
              >
                {promptText}
              </button>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <label htmlFor="stoa-plan-input" className="sr-only">
            Ask a question about {uploadedPlan ? uploadedPlan.name : plan.code}
          </label>
          <input
            ref={inputRef}
            id="stoa-plan-input"
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={
              uploadedPlan
                ? `Ask about ${uploadedPlan.name}...`
                : `Ask about ${plan.code} (${plan.discipline})...`
            }
            className="min-h-11 flex-1 rounded-md border border-slate-300 bg-white px-3.5 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-[#1E40AF] focus:outline-none"
          />
          <button
            type="submit"
            disabled={isLoading || !draft.trim()}
            aria-label="Send question about 2D plan"
            className="inline-flex min-h-11 items-center gap-1.5 rounded-md bg-[#1E40AF] px-4 py-2 text-sm font-medium text-white whitespace-nowrap shrink-0 hover:bg-[#1E3A8A] disabled:opacity-45 disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            <span>Ask</span>
            <SendHorizontal className="h-4 w-4" aria-hidden="true" />
          </button>
        </form>
      </div>
    </section>
  );
}
