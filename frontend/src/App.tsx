import { useEffect, useRef, useState } from "react";
import { Upload, FileText } from "lucide-react";
import { PlanCanvas2D } from "./components/PlanCanvas2D";
import { PlanChatbotPanel, type Message } from "./components/PlanChatbotPanel";
import {
  agentStatus,
  ApiError,
  askDrawing,
  uploadDrawing,
  validateFile,
  type Drawing,
  type Permissions,
} from "./lib/api";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [page, setPage] = useState(1);
  const [drawing, setDrawing] = useState<Drawing | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [selectedEvidence, setSelectedEvidence] = useState<{
    message: string;
    index: number;
  } | null>(null);
  const [zoom, setZoom] = useState(100);
  const [busy, setBusy] = useState<"upload" | "ask" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [configuration, setConfiguration] = useState(
    "Checking agent configuration…",
  );
  const [permissions, setPermissions] = useState<Permissions>({
    allow_gemini: false,
    allow_nemotron: false,
  });
  const input = useRef<HTMLInputElement>(null);
  const pending = useRef(false);
  useEffect(() => {
    agentStatus()
      .then((s) =>
        setConfiguration(
          s.configured ? "Agent configured" : "Server configuration required",
        ),
      )
      .catch(() => setConfiguration("Backend unavailable"));
  }, []);
  async function loadFile(nextFile: File, nextPage: number) {
    if (pending.current) return;
    const validation = validateFile(nextFile);
    if (validation) {
      setError(validation);
      return;
    }
    pending.current = true;
    setBusy("upload");
    setError(null);
    setDrawing(null);
    setMessages([]);
    setSelectedEvidence(null);
    setZoom(100);
    setFile(nextFile);
    setPage(nextPage);
    try {
      setDrawing(await uploadDrawing(nextFile, nextPage));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed. Please retry.");
    } finally {
      pending.current = false;
      setBusy(null);
    }
  }
  async function sendMessage(question: string) {
    if (pending.current || !drawing || !permissions.allow_gemini) return;
    pending.current = true;
    setBusy("ask");
    setError(null);
    setSelectedEvidence(null);
    const id = crypto.randomUUID();
    setMessages((previous) => [
      ...previous,
      { id: id + "-question", role: "user", content: question },
    ]);
    try {
      const result = await askDrawing(
        drawing.document_id,
        question,
        permissions,
      );
      setMessages((previous) => [
        ...previous,
        { id, role: "assistant", content: "", result },
      ]);
    } catch (e) {
      setError(
        e instanceof TypeError
          ? "Connection interrupted. The server may still be processing; no automatic retry was made."
          : e instanceof Error
            ? e.message
            : "Connection interrupted. The server may still be processing.",
      );
      setMessages((previous) => [
        ...previous,
        {
          id,
          role: "assistant",
          content: "",
          result: { status: "unavailable" },
        },
      ]);
      if (e instanceof ApiError && e.status === 404) {
        setDrawing(null);
        setMessages([]);
      }
    } finally {
      pending.current = false;
      setBusy(null);
    }
  }
  const evidenceMessage = selectedEvidence
    ? messages.find((m) => m.id === selectedEvidence.message)
    : [...messages].reverse().find((m) => m.result);
  const reset = () => {
    if (pending.current) return;
    setFile(null);
    setDrawing(null);
    setMessages([]);
    setError(null);
    setPage(1);
    setZoom(100);
    setSelectedEvidence(null);
  };
  return (
    <div className="flex min-h-screen flex-col bg-[#F5F6F8] text-[#0F172A] md:h-screen md:overflow-hidden">
      <header className="shrink-0 border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-3 px-4 py-3 md:px-5">
          <a
            href="#"
            onClick={(e) => {
              e.preventDefault();
              reset();
            }}
            className="font-display text-xl font-bold tracking-tight"
          >
            Planproof<span className="text-blue-700">.</span>
          </a>
          <div className="flex min-w-0 items-center gap-2 text-xs text-slate-600">
            <FileText size={15} />
            <span className="max-w-72 truncate">
              {file?.name || "Drawing workspace"}
            </span>
            {drawing && (
              <span className="rounded bg-slate-100 px-2 py-1">
                Page {drawing.page}
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <span
              className="text-xs text-slate-500"
              title="Configuration presence only; not a live provider connection test."
            >
              {configuration}
            </span>
            <input
              ref={input}
              type="file"
              accept="application/pdf,image/png,image/jpeg"
              aria-label="Upload drawing"
              className="sr-only"
              disabled={!!busy}
              onChange={(e) => {
                const selected = e.target.files?.[0];
                e.target.value = "";
                if (selected) void loadFile(selected, 1);
              }}
            />
            <button
              disabled={!!busy}
              onClick={() => input.current?.click()}
              className="inline-flex min-h-10 items-center gap-2 rounded-md bg-[#1E40AF] px-3.5 py-2 text-xs font-medium text-white hover:bg-[#1E3A8A] disabled:opacity-40"
            >
              <Upload size={15} />
              Upload plan
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto grid w-full max-w-[1600px] flex-1 grid-cols-1 gap-4 p-4 md:min-h-0 md:grid-cols-12">
        <div className="min-w-0 md:col-span-5 md:min-h-0">
          <PlanChatbotPanel
            messages={messages}
            drawing={drawing}
            name={file?.name}
            onSendMessage={sendMessage}
            busy={busy}
            error={error}
            onDismissError={() => setError(null)}
            permissions={permissions}
            onPermissionsChange={setPermissions}
            onClearChat={() => {
              if (!pending.current) {
                setMessages([]);
                setSelectedEvidence(null);
              }
            }}
            selectedEvidence={selectedEvidence}
            onHighlight={(message, index) =>
              setSelectedEvidence({ message, index })
            }
          />
        </div>
        <div className="flex min-w-0 flex-col gap-2 md:col-span-7 md:min-h-0">
          {file?.type === "application/pdf" && (
            <form
              className="flex items-center gap-2 text-xs"
              onSubmit={(e) => {
                e.preventDefault();
                if (file) void loadFile(file, page);
              }}
            >
              <label htmlFor="page-number">PDF page</label>
              <input
                id="page-number"
                aria-label="PDF page"
                type="number"
                min="1"
                step="1"
                required
                value={page}
                disabled={!!busy}
                onChange={(e) => setPage(Number(e.target.value))}
                className="w-20 rounded border border-slate-300 bg-white p-2"
              />
              <button
                disabled={!!busy || !Number.isInteger(page) || page < 1}
                className="rounded border border-slate-300 bg-white px-3 py-2 disabled:opacity-40"
              >
                Load page
              </button>
              <span className="text-slate-500">
                Questions use the loaded page only.
              </span>
            </form>
          )}
          {file && !drawing && !busy && (
            <button
              className="rounded border border-slate-300 bg-white p-2 text-sm"
              onClick={() => void loadFile(file, page)}
            >
              Re-upload {file.name}
            </button>
          )}
          <PlanCanvas2D
            drawing={drawing}
            name={file?.name}
            result={busy === "ask" ? undefined : evidenceMessage?.result}
            selectedIndex={selectedEvidence?.index ?? null}
            onSelectEvidence={(index) => {
              if (evidenceMessage)
                setSelectedEvidence({ message: evidenceMessage.id, index });
            }}
            zoom={zoom}
            onZoomChange={setZoom}
            onReset={reset}
            disabled={!!busy}
            uploading={busy === "upload"}
            onUpload={() => input.current?.click()}
          />
        </div>
      </main>
    </div>
  );
}
