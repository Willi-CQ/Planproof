import { useEffect, useRef, useState, type DragEvent } from "react";
import {
  ArrowUpRight,
  FileText,
  FolderOpen,
  LoaderCircle,
  Plus,
  X,
  MessageCircle,
  Scan,
} from "lucide-react";
import { DrawingLibrary } from "./components/DrawingLibrary";
import {
  readLibrary,
  getSavedDrawing,
  saveDrawing,
  removeDrawing,
  leaveDrawing,
  drawingId,
  type LibraryEntry,
  type SavedDrawing,
} from "./lib/library";
import { BrandMark } from "./components/BrandMark";
import { PlanCanvas2D } from "./components/PlanCanvas2D";
import { PlanChatbotPanel, type Message } from "./components/PlanChatbotPanel";
import {
  ApiError,
  askDrawing,
  uploadDrawing,
  validateFile,
  type Drawing,
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
  const [dragging, setDragging] = useState(false);
  const [mobileTab, setMobileTab] = useState<"drawing" | "ask">("ask");
  const input = useRef<HTMLInputElement>(null);
  const pending = useRef(false);
  const [ready, setReady] = useState(false);
  const [savedId, setSavedId] = useState<string | null>(null);
  const [entries, setEntries] = useState<LibraryEntry[]>([]);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [cacheError, setCacheError] = useState<string | null>(null);
  const cacheFailure = () =>
    setCacheError(
      "Your browser couldn't save these changes. Keep this tab open, or free browser storage and try again.",
    );

  async function restore(saved: SavedDrawing) {
    if (pending.current) return;
    pending.current = true;
    setBusy("upload");
    setLibraryOpen(false);
    setError(null);
    try {
      // Renew the ephemeral backend document using the unchanged original. No inference.
      const fresh = await uploadDrawing(saved.file, saved.drawing.page);
      const sameFrame =
        fresh.width === saved.drawing.width &&
        fresh.height === saved.drawing.height &&
        fresh.page === saved.drawing.page;
      setFile(saved.file);
      setPage(fresh.page);
      setDrawing(fresh);
      setSavedId(saved.id);
      setMessages(saved.messages);
      setDraft(saved.draft);
      setZoom(saved.zoom);
      setSelectedEvidence(sameFrame ? saved.selectedEvidence : null);
      if (saved.interrupted)
        setError(
          "The previous question was interrupted by a refresh. It may still be processing; it has not been sent again.",
        );
      else if (!sameFrame)
        setError(
          "This page rendered differently. Your conversation is saved, but previous highlights may no longer align.",
        );
    } catch {
      setError(
        "Couldn't reopen the saved drawing. Your original and conversation are still saved. Open Your drawings to try again when the server is available.",
      );
    } finally {
      pending.current = false;
      setBusy(null);
    }
  }
  useEffect(() => {
    void readLibrary()
      .then(async (library) => {
        setEntries(library.entries);
        if (library.active) await restore(library.active);
      })
      .catch(cacheFailure)
      .finally(() => setReady(true));
  }, []);
  useEffect(() => {
    if (!ready || !savedId || !file || !drawing || busy === "upload") return;
    void saveDrawing({
      id: savedId,
      file,
      drawing,
      messages,
      draft,
      zoom,
      selectedEvidence,
      interrupted: busy === "ask",
      updated: Date.now(),
    })
      .then(() => {
        setEntries((previous) => [
          {
            id: savedId,
            name: file.name,
            page: drawing.page,
            updated: Date.now(),
          },
          ...previous.filter((entry) => entry.id !== savedId),
        ]);
        setCacheError(null);
      })
      .catch(cacheFailure);
  }, [
    ready,
    savedId,
    file,
    drawing,
    messages,
    draft,
    zoom,
    selectedEvidence,
    busy,
  ]);
  async function openSaved(id: string) {
    try {
      const saved = await getSavedDrawing(id);
      if (saved) await restore(saved);
    } catch {
      cacheFailure();
    }
  }
  async function deleteSaved(id: string) {
    try {
      await removeDrawing(id, id === savedId);
      if (id === savedId) reset();
      setEntries((previous) => previous.filter((entry) => entry.id !== id));
    } catch {
      cacheFailure();
    }
  }

  async function loadFile(nextFile: File, nextPage: number) {
    if (pending.current) return;
    const validation = validateFile(nextFile);
    if (validation) {
      setError(validation);
      return;
    }
    pending.current = true;
    const changingPage = nextFile === file && drawing !== null;
    setBusy("upload");
    setError(null);
    if (!changingPage) {
      setDrawing(null);
      setSavedId(null);
    }
    setFile(nextFile);
    setPage(nextPage);
    if (!changingPage) setMobileTab("ask");
    try {
      const fresh = await uploadDrawing(nextFile, nextPage);
      const id = changingPage && savedId ? savedId : await drawingId(nextFile);
      setDrawing(fresh);
      setSavedId(id);
      setDraft("");
      setMessages([]);
      setSelectedEvidence(null);
      setZoom(100);
    } catch (e) {
      if (changingPage) setPage(drawing.page);
      setError(
        e instanceof Error
          ? e.message
          : "We couldn’t open that drawing. Please try again.",
      );
    } finally {
      pending.current = false;
      setBusy(null);
    }
  }
  async function sendMessage(question: string): Promise<boolean> {
    if (pending.current || !drawing) return false;
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
      // Only explicit send calls the existing Planproof agent.
      // The same agent chooses all tools; no model choice or new routing is introduced.
      const result = await askDrawing(drawing.document_id, question, {
        allow_gemini: true,
        allow_nemotron: true,
      });
      setMessages((previous) => [
        ...previous,
        { id, role: "assistant", content: "", result },
      ]);
      return result.status === "ok";
    } catch (e) {
      setError(
        e instanceof TypeError
          ? "The connection was interrupted. Your question may still be processing. Please wait before sending it again."
          : e instanceof Error
            ? e.message
            : "Something went wrong. Please try again.",
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
      return false;
    } finally {
      pending.current = false;
      setBusy(null);
    }
  }
  const reset = () => {
    if (pending.current) return;
    setSavedId(null);
    setDraft("");
    void leaveDrawing().catch(cacheFailure);
    setFile(null);
    setDrawing(null);
    setMessages([]);
    setError(null);
    setPage(1);
    setZoom(100);
    setSelectedEvidence(null);
  };
  const highlight = (message: string, index: number) => {
    setSelectedEvidence({ message, index });
    setMobileTab("drawing");
  };
  const evidenceMessage = selectedEvidence
    ? messages.find((m) => m.id === selectedEvidence.message)
    : [...messages].reverse().find((m) => m.result);
  function drop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (pending.current || !ready) return;
    if (e.dataTransfer.files.length !== 1) {
      setError("Choose one drawing at a time.");
      return;
    }
    void loadFile(e.dataTransfer.files[0], 1);
  }

  return (
    <div
      className={`product ${drawing ? "product-loaded" : ""} ${dragging ? "product-dragging" : ""}`}
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes("Files")) {
          e.preventDefault();
          if (!pending.current) setDragging(true);
        }
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node))
          setDragging(false);
      }}
      onDrop={drop}
    >
      <header className="app-header">
        <button
          className="brand"
          onClick={reset}
          disabled={!!busy || !ready}
          aria-label="Planproof home"
        >
          <BrandMark />
          <span>Planproof</span>
        </button>
        <div className="header-context">
          {drawing ? (
            <>
              <span className="header-divider" />
              <FileText size={14} />
              <span className="filename" title={file?.name}>
                {file?.name}
              </span>
            </>
          ) : (
            <span>Clarity, built in.</span>
          )}
        </div>
        <div className="header-actions">
          <button
            className="subtle-button"
            aria-label="Your drawings"
            title="Your drawings"
            disabled={!!busy || !ready}
            onClick={() => setLibraryOpen(true)}
          >
            <FolderOpen size={16} />
            <span>Your drawings</span>
          </button>
          {drawing && (
            <button
              className="subtle-button mobile-new"
              aria-label="New drawing"
              disabled={!!busy || !ready}
              onClick={() => input.current?.click()}
            >
              <Plus size={16} />
            </button>
          )}
        </div>
        <input
          ref={input}
          className="visually-hidden"
          type="file"
          aria-label="Upload drawing"
          accept="application/pdf,image/png,image/jpeg"
          disabled={!!busy || !ready}
          onChange={(e) => {
            const selected = e.target.files?.[0];
            e.target.value = "";
            if (selected) void loadFile(selected, 1);
          }}
        />
      </header>

      {(error || cacheError) && (
        <div role="alert" className="error-banner">
          <p>{error || cacheError}</p>
          <button
            className="icon-button"
            onClick={() => {
              setError(null);
              setCacheError(null);
            }}
            aria-label="Dismiss error"
          >
            <X size={16} />
          </button>
        </div>
      )}

      {!drawing ? (
        <main className="welcome">
          <div className="welcome-content">
            <div className="welcome-emblem" aria-hidden="true">
              <FileText strokeWidth={1.1} />
              <span>
                <Scan size={15} />
              </span>
            </div>
            <h1>
              Your plans.
              <br />
              <span>Clearly understood.</span>
            </h1>
            <p className="welcome-description">
              Ask a question. Find the answer in your drawing.
            </p>
            <button
              className={`upload-card ${dragging ? "upload-card-active" : ""}`}
              disabled={!!busy || !ready}
              onClick={() =>
                file && !busy
                  ? void loadFile(file, page)
                  : input.current?.click()
              }
            >
              <span className="upload-card-icon">
                {busy === "upload" ? (
                  <LoaderCircle className="spinning" size={22} />
                ) : (
                  <Plus size={24} strokeWidth={1.5} />
                )}
              </span>
              <span className="upload-card-copy">
                <strong>
                  {busy === "upload"
                    ? "Opening your drawing…"
                    : file
                      ? "Try opening this drawing again"
                      : "Add your drawing"}
                </strong>
                <span>
                  {busy === "upload"
                    ? file?.name
                    : dragging
                      ? "Drop it here"
                      : file
                        ? file.name
                        : "Drop a file here, or browse"}
                </span>
              </span>
              {!busy && (
                <ArrowUpRight size={18} className="upload-card-arrow" />
              )}
            </button>
            {file && !busy && (
              <button
                className="text-button"
                onClick={() => input.current?.click()}
              >
                Choose a different drawing
              </button>
            )}
            <p className="file-hint">
              PDF, PNG or JPG <span>·</span> Up to 10 MB
            </p>
          </div>
          <p className="welcome-footer">
            <span className="mini-rule" />
            From the big picture to the smallest detail.
          </p>
        </main>
      ) : (
        <>
          <nav className="mobile-tabs" aria-label="Workspace view">
            <button
              aria-label="Show drawing"
              aria-pressed={mobileTab === "drawing"}
              onClick={() => setMobileTab("drawing")}
            >
              <Scan size={16} />
              Drawing
            </button>
            <button
              aria-label="Show conversation"
              aria-pressed={mobileTab === "ask"}
              onClick={() => setMobileTab("ask")}
            >
              <MessageCircle size={16} />
              Ask Planproof
            </button>
          </nav>
          <main className="workspace" data-mobile-tab={mobileTab}>
            <div className="drawing-column">
              <PlanCanvas2D
                drawing={drawing}
                name={file?.name}
                result={busy === "ask" ? undefined : evidenceMessage?.result}
                selectedIndex={selectedEvidence?.index ?? null}
                onSelectEvidence={(index) => {
                  if (evidenceMessage) highlight(evidenceMessage.id, index);
                }}
                zoom={zoom}
                onZoomChange={setZoom}
                isPdf={file?.type === "application/pdf"}
                disabled={!!busy || !ready}
                loadingPage={busy === "upload"}
                onPageChange={(nextPage) => {
                  if (file && Number.isInteger(nextPage) && nextPage > 0)
                    void loadFile(file, nextPage);
                }}
              />
            </div>
            <PlanChatbotPanel
              key={drawing.document_id}
              draft={draft}
              onDraftChange={setDraft}
              onNewDrawing={() => input.current?.click()}
              messages={messages}
              drawing={drawing}
              busy={busy === "ask"}
              disabled={!!busy || !ready}
              onSendMessage={sendMessage}
              onClearChat={() => {
                if (!pending.current) {
                  setMessages([]);
                  setSelectedEvidence(null);
                }
              }}
              selectedEvidence={selectedEvidence}
              onHighlight={highlight}
            />
          </main>
        </>
      )}
      {libraryOpen && (
        <DrawingLibrary
          entries={entries}
          active={savedId}
          onClose={() => setLibraryOpen(false)}
          onOpen={(id) => void openSaved(id)}
          onRemove={(id) => void deleteSaved(id)}
          onAdd={() => {
            setLibraryOpen(false);
            input.current?.click();
          }}
        />
      )}
      {dragging && (
        <div className="drop-overlay" aria-hidden="true">
          <div>
            <Plus size={28} />
            <strong>Drop your drawing</strong>
          </div>
        </div>
      )}
    </div>
  );
}
