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

const INITIAL_PLAN_MESSAGES: Record<string, PlanChatMessage[]> = {
  'plan-a101': [
    {
      id: 'seed-a101-u',
      role: 'user',
      content: 'What is the clear corridor width along Gridline B?',
      timestamp: '09:10',
    },
    {
      id: 'seed-a101-a',
      role: 'assistant',
      content:
        'Main Egress Corridor (Grid B–C) maintains a minimum clear unobstructed width of **1,850 mm** between finished acoustic wall linings, with honed terrazzo over 40 mm acoustic screed [[Z1|COR-01 · Grid B-2]].',
      timestamp: '09:10',
    },
  ],
  'plan-m301': [
    {
      id: 'seed-m301-u',
      role: 'user',
      content: 'What maintenance clearance is required around CH-01?',
      timestamp: '10:05',
    },
    {
      id: 'seed-m301-a',
      role: 'assistant',
      content:
        'Chiller Skid **CH-01** requires a minimum **1,200 mm** service and tube-pull clearance on all four sides of its 6.4 m x 2.2 m C40/50 inertia plinth [[Z1|CH-01 · Grid A-1]].',
      timestamp: '10:05',
    },
  ],
  'plan-s204': [
    {
      id: 'seed-s204-u',
      role: 'user',
      content: 'What span and thickness apply to the CLT deck panels?',
      timestamp: '11:15',
    },
    {
      id: 'seed-s204-a',
      role: 'assistant',
      content:
        'Level 03 uses **240 mm** 7-ply spruce CLT panels spanning **7,200 mm** clear between primary glulam beams, with core moisture kept below 15.5% prior to enclosure [[Z2|CLT-03 · Grid B-1]].',
      timestamp: '11:15',
    },
  ],
};

export default function App() {
  const [activePlanId, setActivePlanId] = useState<string>(COMPANY_PLANS[0].id);
  const [selectedZoneId, setSelectedZoneId] = useState<string | null>('Z1');
  const [zoom, setZoom] = useState<number>(100);
  const [messagesByPlan, setMessagesByPlan] =
    useState<Record<string, PlanChatMessage[]>>(INITIAL_PLAN_MESSAGES);
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
      setErrorMessage(validation.error ?? 'Invalid 2D plan image.');
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const result = typeof reader.result === 'string' ? reader.result : '';
      const base64Data = result.split(',')[1] ?? '';
      if (!base64Data) {
        setErrorMessage('Could not read the selected image file.');
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
        throw new Error('STOA chat API is not configured. Set VITE_STOA_API_BASE_URL in the Pages workflow variables.');
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
    <div className="flex min-h-screen flex-col bg-[#F5F6F8] text-[#0F172A] md:h-screen md:overflow-hidden">
      {/* Compact 3-Zone Top Bar */}
      <header className="border-b border-slate-200 bg-white shrink-0">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-3 px-4 py-2.5 md:flex-nowrap md:gap-6 md:px-5 md:py-3">
          {/* Zone 1: Brand Wordmark */}
          <a
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              handleSelectPlan(COMPANY_PLANS[0].id);
            }}
            className="font-display text-xl font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
          >
            Stoa
          </a>

          {/* Zone 2: 2D Plan Sheet Tabs */}
          <nav
            aria-label="Company 2D plan sheets"
            className="flex flex-wrap items-center gap-1 rounded-md border border-slate-200 bg-[#F8FAFC] p-1"
          >
            {COMPANY_PLANS.map((plan) => {
              const isActive = !uploadedPlan && plan.id === activePlan.id;
              return (
                <button
                  key={plan.id}
                  type="button"
                  aria-current={isActive ? 'page' : undefined}
                  onClick={() => handleSelectPlan(plan.id)}
                  className={`min-h-8 rounded px-2.5 py-1 font-mono text-xs font-medium whitespace-nowrap shrink-0 transition-colors cursor-pointer ${
                    isActive
                      ? 'bg-[#0F172A] text-white'
                      : 'text-slate-600 hover:bg-white hover:text-slate-900'
                  }`}
                >
                  {plan.shortLabel}
                </button>
              );
            })}
          </nav>

          {/* Zone 3: Primary Action (Upload 2D Plan) */}
          <div className="shrink-0">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={handleFileChange}
              aria-label="Upload 2D plan image file"
              className="sr-only"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex min-h-10 items-center gap-2 rounded-md bg-[#1E40AF] px-3.5 py-2 text-xs font-medium text-white whitespace-nowrap shrink-0 hover:bg-[#1E3A8A] transition-colors cursor-pointer"
            >
              <Upload className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Upload 2D Plan</span>
            </button>
          </div>
        </div>
      </header>

      {/* Single-Page Split Workspace: Chatbot Left, Scanned 2D Plan Right */}
      <main className="mx-auto grid w-full max-w-[1440px] flex-1 grid-cols-1 gap-4 p-4 md:min-h-0 md:grid-cols-12">
        <div className="md:col-span-5 md:min-h-0">
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

        <div className="md:col-span-7 md:min-h-0">
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
        </div>
      </main>
    </div>
  );
}
