import { useRef, useState, type FormEvent, type RefObject } from 'react';
import { AlertCircle, MapPin, Send, Sparkles, Trash2, X } from 'lucide-react';
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

  const ask = async (query: string) => {
    await onSendMessage(query);
    requestAnimationFrame(() => {
      if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    });
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = draft.trim();
    if (!trimmed || isLoading) return;
    setDraft('');
    await ask(trimmed);
  };

  const suggestionLabels = plan.id === 'plan-a101'
    ? ['Largeur du couloir', 'Porte D-04']
    : plan.id === 'plan-m301'
      ? ['Dégagement CH-01', 'Réseau hydraulique']
      : ['Portée du plancher', 'Assemblage B-2'];
  const suggestions = plan.starterPrompts.slice(0, 2);

  return (
    <section
      aria-label="Assistant Stoa"
      className="flex min-h-[440px] flex-col overflow-hidden rounded-2xl border border-black/[0.06] bg-white shadow-[0_2px_12px_rgba(0,0,0,0.035)] md:h-full md:min-h-0"
    >
      <div className="flex items-center justify-between border-b border-black/[0.06] px-4 py-3.5">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-[#eef1ff] text-[#4968e8]">
            <Sparkles className="h-4 w-4" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold tracking-[-0.02em]">Assistant Stoa</h2>
            <p className="truncate text-[11px] text-[#86868b]">
              {uploadedPlan ? uploadedPlan.name : plan.code}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={onClearChat}
          disabled={messages.length === 0 || isLoading}
          aria-label="Effacer la conversation"
          title="Effacer la conversation"
          className="flex h-8 w-8 items-center justify-center rounded-full text-[#86868b] transition-colors hover:bg-[#f5f5f5] hover:text-[#1d1d1f] disabled:opacity-30 cursor-pointer disabled:cursor-default"
        >
          <Trash2 className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>

      <div
        ref={scrollRef}
        aria-live="polite"
        data-testid="plan-chat-messages"
        className="flex-1 space-y-4 overflow-y-auto px-4 py-4"
      >
        {messages.length === 0 ? (
          <div className="flex h-full min-h-[250px] flex-col items-center justify-center px-3 text-center">
            <span className="mb-3 flex h-11 w-11 items-center justify-center rounded-full bg-[#f5f5f7] text-[#6e6e73]">
              <Sparkles className="h-5 w-5" aria-hidden="true" />
            </span>
            <p className="text-sm font-medium tracking-[-0.02em]">Que souhaitez-vous vérifier&nbsp;?</p>
            <p className="mt-1 text-xs text-[#86868b]">Sélectionnez une zone ou posez une question.</p>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'user' ? (
                <div className="max-w-[88%] rounded-2xl rounded-br-md bg-[#f2f2f2] px-3.5 py-2.5 text-sm leading-relaxed text-[#1d1d1f]">
                  {msg.content}
                </div>
              ) : (
                <div className="max-w-[96%] space-y-2 text-sm leading-relaxed text-[#3a3a3c]">
                  {parsePlanMessage(msg.content, uploadedPlan ? [] : plan.zones).map((block, blockIdx) => {
                    const renderRuns = (runs: InlineRun[]) => runs.map((run, runIdx) => {
                      if (run.type === 'text') {
                        return run.bold ? (
                          <strong key={runIdx} className="font-semibold text-[#1d1d1f]">{run.text}</strong>
                        ) : <span key={runIdx}>{run.text}</span>;
                      }
                      const { citation } = run;
                      const isActive = selectedZoneId === citation.zone?.id;
                      return (
                        <button
                          key={runIdx}
                          type="button"
                          onClick={() => citation.zone && onHighlightZone(citation.zone.id)}
                          title={`Afficher ${citation.label} sur le plan`}
                          className={`mx-1 inline-flex items-center gap-1 rounded-full px-2 py-0.5 align-baseline text-[11px] font-medium transition-colors cursor-pointer ${
                            isActive ? 'bg-[#4968e8] text-white' : 'bg-[#eef1ff] text-[#405fd5] hover:bg-[#e2e8ff]'
                          }`}
                        >
                          <MapPin className="h-3 w-3" aria-hidden="true" />
                          {citation.label}
                        </button>
                      );
                    });

                    return block.type === 'bullet' ? (
                      <p key={blockIdx} className="pl-2">· {renderRuns(block.runs)}</p>
                    ) : <p key={blockIdx}>{renderRuns(block.runs)}</p>;
                  })}
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div aria-label="Analyse du plan en cours" className="max-w-[85%] space-y-2 py-1">
            <div className="h-2.5 w-1/3 animate-pulse rounded-full bg-[#e9e9eb]" />
            <div className="h-2.5 w-5/6 animate-pulse rounded-full bg-[#e9e9eb]" />
            <div className="h-2.5 w-2/3 animate-pulse rounded-full bg-[#e9e9eb]" />
          </div>
        )}
      </div>

      {errorMessage && (
        <div role="alert" className="mx-3 mb-2 flex items-start justify-between gap-2 rounded-xl bg-[#fff1f0] px-3 py-2.5 text-xs text-[#a12622]">
          <div className="flex items-start gap-2">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <span>{errorMessage}</span>
          </div>
          <button type="button" onClick={onDismissError} aria-label="Fermer" className="rounded p-0.5 hover:bg-red-100 cursor-pointer">
            <X className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </div>
      )}

      <div className="border-t border-black/[0.06] p-3.5">
        {messages.length === 0 && !uploadedPlan && (
          <div className="mb-2.5 flex flex-wrap gap-1.5">
            {suggestions.map((promptText, index) => (
              <button
                key={promptText}
                type="button"
                disabled={isLoading}
                onClick={() => void ask(promptText)}
                className="rounded-full bg-[#f5f5f7] px-3 py-1.5 text-[11px] text-[#6e6e73] transition-colors hover:bg-[#ededf0] hover:text-[#1d1d1f] disabled:opacity-50 cursor-pointer"
              >
                {suggestionLabels[index]}
              </button>
            ))}
          </div>
        )}
        <form onSubmit={handleSubmit} className="flex items-center gap-2 rounded-full bg-[#f5f5f7] p-1.5 pl-4 focus-within:ring-2 focus-within:ring-[#4968e8]/25">
          <label htmlFor="stoa-plan-input" className="sr-only">Votre question sur le plan</label>
          <input
            ref={inputRef}
            id="stoa-plan-input"
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Poser une question…"
            className="min-w-0 flex-1 bg-transparent py-1.5 text-sm text-[#1d1d1f] outline-none placeholder:text-[#a1a1a6]"
          />
          <button
            type="submit"
            disabled={isLoading || !draft.trim()}
            aria-label="Envoyer"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#4968e8] text-white transition-colors hover:bg-[#3654ce] disabled:bg-[#d2d2d7] cursor-pointer disabled:cursor-default"
          >
            <Send className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </form>
      </div>
    </section>
  );
}
