import { useRef, useState, type ChangeEvent } from 'react';
import { Upload } from 'lucide-react';
import {
  COMPANY_PLANS,
  type PlanChatMessage,
  type PlanZone,
  type UploadedPlanFile,
} from './lib/plansData';
import { validatePlanImageFile } from './lib/stoaEngine';
import { PlanCanvas2D } from './components/PlanCanvas2D';
import { PlanChatbotPanel } from './components/PlanChatbotPanel';
import { stoaApiUrl } from './lib/api';

function formatNowTime(): string {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function App() {
  const [activePlanId, setActivePlanId] = useState<string>(COMPANY_PLANS[0].id);
  const [selectedZoneId, setSelectedZoneId] = useState<string | null>('Z1');
  const [zoom, setZoom] = useState<number>(100);
  const [messagesByPlan, setMessagesByPlan] =
    useState<Record<string, PlanChatMessage[]>>({});
  const [uploadedPlan, setUploadedPlan] = useState<UploadedPlanFile | null>(null);
  const [uploadedMessages, setUploadedMessages] = useState<PlanChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const chatInputRef = useRef<HTMLInputElement | null>(null);

  const activePlan =
    COMPANY_PLANS.find((p) => p.id === activePlanId) ?? COMPANY_PLANS[0];

  const activeMessages = uploadedPlan
    ? uploadedMessages
    : (messagesByPlan[activePlan.id] ?? []);

  const handleSelectPlan = (planId: string) => {
    setUploadedPlan(null);
    setErrorMessage(null);
    setActivePlanId(planId);
    setSelectedZoneId('Z1');
    setZoom(100);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;

    const validation = validatePlanImageFile({ type: file.type, size: file.size });
    if (!validation.valid) {
      setErrorMessage(validation.error ?? 'Format de plan non pris en charge.');
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const result = typeof reader.result === 'string' ? reader.result : '';
      const base64Data = result.split(',')[1] ?? '';
      if (!base64Data) {
        setErrorMessage('Impossible de lire ce fichier.');
        return;
      }
      setErrorMessage(null);
      setUploadedPlan({
        name: file.name,
        mimeType: file.type,
        base64Data,
        previewUrl: result,
      });
      setUploadedMessages([]);
      setZoom(100);
      setTimeout(() => chatInputRef.current?.focus(), 100);
    };
    reader.readAsDataURL(file);
  };

  const sendMessage = async (query: string) => {
    const userMsg: PlanChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: query,
      timestamp: formatNowTime(),
    };

    setErrorMessage(null);
    const currentHistory = activeMessages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    if (uploadedPlan) {
      setUploadedMessages((prev) => [...prev, userMsg]);
    } else {
      setMessagesByPlan((prev) => ({
        ...prev,
        [activePlan.id]: [...(prev[activePlan.id] ?? []), userMsg],
      }));
    }

    setIsLoading(true);
    try {
      if (!stoaApiUrl) {
        throw new Error('L’assistant n’est pas encore connecté.');
      }
      const response = await fetch(`${stoaApiUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          plan: uploadedPlan ? undefined : activePlan,
          history: currentHistory,
          uploadedImage: uploadedPlan
            ? {
                name: uploadedPlan.name,
                mimeType: uploadedPlan.mimeType,
                base64Data: uploadedPlan.base64Data,
              }
            : null,
        }),
      });

      const data = (await response.json()) as {
        reply?: string;
        error?: string;
      };

      if (!response.ok || !data.reply) {
        throw new Error(data.error || 'Unable to inspect 2D plan.');
      }

      const assistantMsg: PlanChatMessage = {
        id: `a-${Date.now()}`,
        role: 'assistant',
        content: data.reply,
        timestamp: formatNowTime(),
      };

      if (uploadedPlan) {
        setUploadedMessages((prev) => [...prev, assistantMsg]);
      } else {
        setMessagesByPlan((prev) => ({
          ...prev,
          [activePlan.id]: [...(prev[activePlan.id] ?? []), assistantMsg],
        }));
      }
    } catch (err: unknown) {
      setErrorMessage(
        err instanceof Error ? err.message : 'Network error contacting Stoa.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    setErrorMessage(null);
    if (uploadedPlan) {
      setUploadedMessages([]);
    } else {
      setMessagesByPlan((prev) => ({
        ...prev,
        [activePlan.id]: [],
      }));
    }
  };

  const handleAskAboutZone = (zone: PlanZone) => {
    void sendMessage(`Check specifications and clearances for ${zone.tag} (${zone.title}).`);
  };

  return (
    <div className="flex min-h-screen flex-col bg-[#f5f5f3] text-[#1d1d1f] md:h-screen md:overflow-hidden">
      <header className="shrink-0 border-b border-black/[0.06] bg-white/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4 px-5 py-3 sm:px-8">
          <a
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              handleSelectPlan(COMPANY_PLANS[0].id);
            }}
            className="font-display text-[19px] font-semibold tracking-[-0.06em] text-[#1d1d1f]"
          >
            stoa<span className="text-[#4968e8]">.</span>
          </a>

          <nav aria-label="Plans" className="flex items-center gap-1 rounded-full bg-[#f2f2f2] p-1">
            {COMPANY_PLANS.map((plan) => {
              const isActive = !uploadedPlan && plan.id === activePlan.id;
              return (
                <button
                  key={plan.id}
                  type="button"
                  aria-current={isActive ? 'page' : undefined}
                  onClick={() => handleSelectPlan(plan.id)}
                  className={`min-h-8 rounded-full px-3.5 text-xs font-medium transition-colors cursor-pointer ${
                    isActive
                      ? 'bg-white text-[#1d1d1f] shadow-sm'
                      : 'text-[#6e6e73] hover:text-[#1d1d1f]'
                  }`}
                >
                    {plan.code}
                </button>
              );
            })}
          </nav>

          <div className="shrink-0">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={handleFileChange}
              aria-label="Importer un plan 2D"
              className="sr-only"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex min-h-9 items-center gap-2 rounded-full bg-[#1d1d1f] px-4 text-xs font-medium text-white transition-colors hover:bg-[#3a3a3c] cursor-pointer"
            >
              <Upload className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Importer</span>
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-[1600px] flex-1 flex-col gap-5 px-5 py-6 sm:px-8 md:min-h-0 md:py-8">
        <div className="flex shrink-0 items-end justify-between gap-4">
          <div>
            <p className="mb-1 text-[11px] font-medium uppercase tracking-[0.16em] text-[#86868b]">Espace de travail</p>
            <h1 className="font-display text-2xl font-semibold tracking-[-0.045em] sm:text-[30px]">
              {uploadedPlan ? uploadedPlan.name : activePlan.title}
            </h1>
          </div>
          <span className="mb-1 hidden rounded-full bg-white px-3 py-1.5 text-xs text-[#6e6e73] shadow-sm sm:inline-flex">
            {uploadedPlan ? 'Nouveau plan' : `${activePlan.code} · ${activePlan.revision}`}
          </span>
        </div>

        <div className="grid flex-1 grid-cols-1 gap-4 md:min-h-0 md:grid-cols-[minmax(0,1fr)_360px] xl:grid-cols-[minmax(0,1fr)_390px]">
          <PlanCanvas2D
            plan={activePlan}
            uploadedPlan={uploadedPlan}
            onClearUpload={() => {
              setUploadedPlan(null);
              setErrorMessage(null);
            }}
            selectedZoneId={selectedZoneId}
            onSelectZone={setSelectedZoneId}
            zoom={zoom}
            onZoomChange={setZoom}
            onAskAboutZone={handleAskAboutZone}
          />

          <PlanChatbotPanel
            plan={activePlan}
            uploadedPlan={uploadedPlan}
            messages={activeMessages}
            onSendMessage={sendMessage}
            onClearChat={handleClearChat}
            selectedZoneId={selectedZoneId}
            onHighlightZone={(id) => setSelectedZoneId(id)}
            isLoading={isLoading}
            errorMessage={errorMessage}
            onDismissError={() => setErrorMessage(null)}
            inputRef={chatInputRef}
          />
        </div>
      </main>
    </div>
  );
}
