import { useEffect, useRef, useState, type DragEvent } from "react";
import {
  ArrowUpRight,
  Box,
  FileText,
  FolderOpen,
  LoaderCircle,
  Plus,
  X,
  MessageCircle,
  Scan,
  Ruler,
} from "lucide-react";
import { DrawingLibrary } from "./components/DrawingLibrary";
import {
  readLibrary,
  getSavedDrawing,
  saveDrawing,
  removeDrawing,
  leaveDrawing,
  drawingId,
  readerWorkspace,
  type LibraryEntry,
  type SavedDrawing,
} from "./lib/library";
import { BrandMark } from "./components/BrandMark";
import { PlanCanvas2D } from "./components/PlanCanvas2D";
import { PlanChatbotPanel, type Message } from "./components/PlanChatbotPanel";
import { ApiError, uploadDrawing, validateFile, type Drawing } from "./lib/api";
import { askReader, prepareReaderPage, validateReaderFile, type ReaderPage } from "./lib/reader";
import { loadPublicSheet, publicSheets, type PublicSheet } from "./lib/publicSheets";
import spatialSampleUrl from "./assets/plan-sample.png";
import { TakeoffPanel } from "./components/Takeoff";
import { readerTakeoff } from "./lib/takeoff";
import { useTakeoff } from "./lib/useTakeoff";

import { liveActivity } from "./lib/activity";
import { downloadTaskReport } from "./lib/report";
import { EvidenceInspector } from "./components/EvidenceInspector";
import {
  startRun,
  getRun,
  getEvidence,
  mergeRun,
  terminal,
  sameSource,
  type AgentRun,
  type EvidenceShape,
  type Region,
} from "./lib/runs";

const spatialSampleName = "Planproof 3D sample.png";
const spatialSampleHash = "687dce98c60de863d2b9eafe75d0e02d9eb0df44940ca292d43d3a1a2b7a572c";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [page, setPage] = useState(1);
  const [drawing, setDrawing] = useState<Drawing | null>(null);
  const [readerPage, setReaderPage] = useState<ReaderPage | null>(null);
  const [allowPageImage, setAllowPageImage] = useState(false);
  const [loadingExample, setLoadingExample] = useState("");
  useEffect(() => () => {
    if (readerPage) {
      URL.revokeObjectURL(readerPage.drawing.image_url);
      if (readerPage.drawing.original_url) URL.revokeObjectURL(readerPage.drawing.original_url);
    }
  }, [readerPage]);
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
  const [takeoffOpen,setTakeoffOpen]=useState(false);
  const takeoff=useTakeoff(readerWorkspace ? savedId : null,drawing);
  const [entries, setEntries] = useState<LibraryEntry[]>([]);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [cacheError, setCacheError] = useState<string | null>(null);
  const [inspectionId, setInspectionId] = useState<string | null>(null);
  const [inspectionView, setInspectionView] = useState<"steps" | "candidates">("steps");
  const [focusRegion, setFocusRegion] = useState<Region | null>(null);
  const [roomPick, setRoomPick] = useState<{question:string; german:boolean} | null>(null);
  const [shapes, setShapes] = useState<EvidenceShape[]>([]);
  const [follow, setFollow] = useState(false);
  const [followLive, setFollowLive] = useState(false);
  const [comparisonSplit, setComparisonSplit] = useState<number | null>(null);
  const [connection, setConnection] = useState("");
  const [layersLoading, setLayersLoading] = useState(false);
  const [layerError, setLayerError] = useState("");
  const [reportState, setReportState] = useState<{id?: string; busy: boolean; error?: string}>({busy: false});
  const reportPending = useRef(false);
  const mounted = useRef(true);
  useEffect(
    () => () => {
      mounted.current = false;
    },
    [],
  );
  const activeMessage = [...messages].reverse().find(m => m.run && !terminal(m.run));
  const activeRun = activeMessage?.run;
  const activeView = liveActivity(activeRun, drawing?.source);
  useEffect(() => {
    if (followLive && activeView?.regions.length) {
      setSelectedEvidence(null);
      setFocusRegion(activeView.regions[0]);
    }
  }, [activeRun?.request_id, activeView?.focusKey, followLive]);
  const closeInspector = () => {
    setComparisonSplit(null);
    setInspectionId(null);
    setInspectionView("steps");
    setFocusRegion(null);
    setShapes([]);
    setFollow(false);
  };
  const cacheFailure = () =>
    setCacheError(
      "Your browser couldn't save these changes. Keep this tab open, or free browser storage and try again.",
    );

  async function restore(saved: SavedDrawing) {
    if (pending.current) return;
    pending.current = true;
    setTakeoffOpen(false);
    setBusy("upload");
    setLibraryOpen(false);
    setError(null);
    try {
      // Renew the ephemeral backend document using the unchanged original. No inference.
      const prepared = readerWorkspace ? await prepareReaderPage(saved.file, saved.drawing.page) : null;
      const fresh = prepared?.drawing ?? await uploadDrawing(saved.file, saved.drawing.page);
      setReaderPage(prepared);
      setAllowPageImage(false);
      const sameFrame =
        fresh.width === saved.drawing.width &&
        fresh.height === saved.drawing.height &&
        fresh.page === saved.drawing.page;
      closeInspector();
      setFile(saved.file);
      setPage(fresh.page);
      setDrawing(fresh);
      setSavedId(saved.id);
      setMessages(saved.messages);
      setDraft(saved.draft);
      setZoom(saved.zoom);
      setSelectedEvidence(sameFrame ? saved.selectedEvidence : null);
      const activeRun = saved.messages.find((m) => m.run && !terminal(m.run));
      if (!readerWorkspace && activeRun?.run)
        setTimeout(() => {
          void resumeRun(activeRun.id, activeRun.run!);
        }, 0);
      if (saved.interrupted && !activeRun)
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
    setRoomPick(null);
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

  function openExample(sheet: PublicSheet) {
    return openExampleFile(sheet.name, () => loadPublicSheet(sheet));
  }
  function openSpatialSample() {
    return openExampleFile(spatialSampleName, async () => {
      const response = await fetch(spatialSampleUrl);
      if (!response.ok) throw new Error("The 3D sample could not be opened.");
      return new File([await response.blob()], spatialSampleName, {type:"image/png"});
    }, true);
  }
  async function openExampleFile(name: string, getFile: () => Promise<File>, showDrawing = false) {
    if (pending.current) return;
    pending.current = true;
    setBusy("upload");
    setLoadingExample(name);
    setError(null);
    try {
      const source = await getFile();
      pending.current = false;
      await loadFile(source, 1);
      if (showDrawing) setMobileTab("drawing");
    } catch (e) {
      setError(e instanceof Error ? e.message : "The reference sheet could not be downloaded.");
    } finally {
      pending.current = false;
      setLoadingExample("");
      setBusy(null);
    }
  }
  async function loadFile(nextFile: File, nextPage: number) {
    if (pending.current) return;
    const validation = readerWorkspace ? validateReaderFile(nextFile) : validateFile(nextFile);
    if (validation) {
      setError(validation);
      return;
    }
    pending.current = true;
    setRoomPick(null);
    setTakeoffOpen(false);
    const changingPage = nextFile === file && drawing !== null;
    setBusy("upload");
    closeInspector();
    setError(null);
    if (!changingPage) {
      setDrawing(null);
      setSavedId(null);
    }
    setFile(nextFile);
    setPage(nextPage);
    if (!changingPage) setMobileTab("ask");
    try {
      const prepared = readerWorkspace ? await prepareReaderPage(nextFile, nextPage,
        changingPage ? readerPage?.originalHash : undefined) : null;
      const fresh = prepared?.drawing ?? await uploadDrawing(nextFile, nextPage);
      const id = prepared?.drawing.document_id ?? (changingPage && savedId ? savedId : await drawingId(nextFile));
      const saved = prepared ? await getSavedDrawing(id).catch(() => undefined) : undefined;
      setReaderPage(prepared);
      setAllowPageImage(false);
      setDrawing(fresh);
      setSavedId(id);
      setDraft(saved?.draft ?? "");
      setMessages(saved?.messages ?? []);
      setSelectedEvidence(saved?.selectedEvidence ?? null);
      setZoom(saved?.zoom ?? 100);
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
  function updateRun(id: string, run: AgentRun) {
    setMessages((previous) =>
      previous.map((m) =>
        m.id === id
          ? {
              ...m,
              run: {
                ...run,
                evidence: run.evidence ?? m.run?.evidence,
              },
              result:
                run.result ||
                (["failed", "interrupted"].includes(run.status)
                  ? { status: "unavailable" }
                  : undefined),
            }
          : m,
      ),
    );
  }
  async function watchRun(id: string, initial: AgentRun): Promise<boolean> {
    let current = initial;
    updateRun(id, current);
    let missing = 0;
    while (mounted.current && !terminal(current)) {
      try {
        const next = await getRun(current.request_id, current.last_sequence);
        current = mergeRun(current, next);
        updateRun(id, current);
        setConnection("");
        missing = 0;
      } catch (e) {
        if (e instanceof ApiError && e.status === 404 && ++missing >= 3) {
          current = {
            ...current,
            status: "interrupted",
            message:
              "Activity is unavailable. This question has not been sent again.",
          };
          updateRun(id, current);
          setError(current.message!);
          break;
        }
        setConnection("Reconnecting to activity");
      }
      if (!terminal(current))
        await new Promise((resolve) =>
          setTimeout(resolve, document.hidden ? 3000 : 1000),
        );
    }
    if (current.status === "failed" || current.status === "interrupted")
      setError(
        current.message ||
          "The run was interrupted; it has not been resubmitted.",
      );
    if (current.result?.status === "ok") setDraft("");
    return current.result?.status === "ok";
  }
  async function resumeRun(id: string, run: AgentRun) {
    if (pending.current || !mounted.current) return;
    pending.current = true;
    setBusy("ask");
    try {
      await watchRun(id, run);
    } finally {
      pending.current = false;
      setBusy(null);
      setConnection("");
    }
  }
  async function sendMessage(question: string, roomPoint?: [number,number]): Promise<boolean> {
    if (pending.current || !drawing || !file) return false;
    if (readerWorkspace && (!readerPage || (readerPage.representation === "image" && !allowPageImage))) return false;
    pending.current = true;
    setRoomPick(null);
    setBusy("ask");
    setError(null);
    setSelectedEvidence(null);
    closeInspector();
    const id = crypto.randomUUID();
    const run: AgentRun = {
      request_id: id,
      status: "queued",
      source: drawing.source || {
        raster_sha256: "",
        page_index: drawing.page - 1,
        width: drawing.width,
        height: drawing.height,
      },
      events: [],
      last_sequence: 0,
    };
    const nextMessages: Message[] = [
      ...messages,
      { id: id + "-question", role: "user", content: question },
      ...(!readerWorkspace ? [{ id, role: "assistant" as const, content: "", run }] : []),
    ];
    setMessages(nextMessages);
    try {
      // Save the ID before dispatch so refresh cannot accidentally submit twice.
      if (savedId)
        await saveDrawing({
          id: savedId,
          file,
          drawing,
          messages: nextMessages,
          draft: question,
          zoom,
          selectedEvidence: null,
          interrupted: true,
          updated: Date.now(),
        }).catch(cacheFailure);
      if (readerWorkspace && readerPage) {
        const result = await askReader(readerPage, question, roomPoint);
        setMessages(previous => [...previous, {id, role: "assistant", content: "", result}]);
        if (result.answer?.citation_details?.[0]?.kind === "measurement")
          setSelectedEvidence({message:id, index:0});
        if (roomPoint && (result.answer?.status === "abstain" || result.reader_reason === "room_scale_unverified"))
          setMobileTab("ask");
        setDraft("");
        return true;
      }
      let accepted: AgentRun;
      try {
        accepted = await startRun(drawing.document_id, id, question);
      } catch (e) {
        if (!(e instanceof ApiError)) {
          setConnection("Reconnecting to activity");
          return await watchRun(id, run);
        }
        updateRun(id, { ...run, status: "failed", message: e.message });
        if (e.status === 404) {
          setDrawing(null);
          setMessages([]);
          throw new ApiError(
            "This uploaded page has expired. Reopen the saved drawing before asking again.",
            404,
          );
        }
        throw e;
      }
      return await watchRun(id, accepted);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "This question could not be started.",
      );
      return false;
    } finally {
      pending.current = false;
      setBusy(null);
      setConnection("");
    }
  }
  const inspected = messages.find((m) => m.id === inspectionId);
  const latestTask = [...messages].reverse().find(m => m.role === "assistant");
  async function downloadReport() {
    if (!drawing || !latestTask?.run || !terminal(latestTask.run) || reportPending.current) return;
    const task = latestTask;
    const question = messages.slice(0, messages.indexOf(task)).reverse().find(m => m.role === "user")?.content || "";
    reportPending.current = true;
    setReportState({id: task.id, busy: true});
    try {
      await downloadTaskReport(task.run!.request_id, drawing.document_id, question, file?.name || "Drawing");
      setReportState({id: task.id, busy: false});
    } catch (error) {
      setReportState({id: task.id, busy: false, error: error instanceof Error && error.name !== "TimeoutError"
        ? error.message : "The report took too long to prepare. Please try again."});
    } finally { reportPending.current = false; }
  }
  async function loadLayers() {
    const run = inspected?.run;
    if (!run || layersLoading) return;
    setLayersLoading(true);
    setLayerError("");
    try {
      const evidence = await getEvidence(run.request_id);
      setMessages((previous) =>
        previous.map((m) =>
          m.id === inspectionId && m.run
            ? { ...m, run: { ...m.run, evidence } }
            : m,
        ),
      );
    } catch {
      setLayerError("Recorded layers could not be loaded.");
    } finally {
      setLayersLoading(false);
    }
  }
  function focusWork(messageId: string, region: Region) {
    const message = messages.find(m => m.id === messageId);
    if (!sameSource(message?.run?.source, drawing?.source) || region.page_index !== drawing!.page - 1) return;
    setFocusRegion(region);
    setSelectedEvidence(null);
    setMobileTab("drawing");
  }
  function focus(region: Region) {
    if (
      !sameSource(inspected?.run?.source, drawing?.source) ||
      region.page_index !== drawing!.page - 1
    )
      return;
    setFocusRegion(region);
    setSelectedEvidence(null);
    setMobileTab("drawing");
  }
  const reset = () => {
    setTakeoffOpen(false);
    setRoomPick(null);
    if (pending.current) return;
    closeInspector();
    setSavedId(null);
    setDraft("");
    void leaveDrawing().catch(cacheFailure);
    setFile(null);
    setReaderPage(null);
    setAllowPageImage(false);
    setDrawing(null);
    setMessages([]);
    setError(null);
    setPage(1);
    setZoom(100);
    setSelectedEvidence(null);
  };
  const highlight = (message: string, index: number) => {
    setRoomPick(null);
    setFocusRegion(null);
    setSelectedEvidence({ message, index });
    setMobileTab("drawing");
  };
  function pickRoom(messageId: string) {
    if (!readerPage || busy || inspectionId || (readerPage.representation === "image" && !allowPageImage)) return;
    const answer = messages.findIndex(m => m.id === messageId);
    if (answer < 1) return;
    const question = messages.slice(0,answer).reverse().find(m => m.role === "user")?.content;
    if (!question) return;
    const explanation = messages[answer].result?.answer?.explanation ?? "";
    setRoomPick({question,german:/^(Wähle|Mehrere|Um diesen|Ich|Die Beschriftung)/.test(explanation)});
    setSelectedEvidence(null);
    setFocusRegion(null);
    setMobileTab("drawing");
  }
  const evidenceMessage =
    inspected ||
    (selectedEvidence
      ? messages.find((m) => m.id === selectedEvidence.message)
      : [...messages].reverse().find((m) => m.role === "assistant"));
  const planResult = busy === "ask" || (drawing && evidenceMessage?.run &&
    !sameSource(evidenceMessage.run.source,drawing.source)) ? undefined : evidenceMessage?.result;
  function openTakeoff() {
    if (pending.current || !drawing || !savedId) return;
    closeInspector();
    setRoomPick(null);
    setFollowLive(false);
    setTakeoffOpen(true);
    setMobileTab("drawing");
  }
  function closeTakeoff() {
    takeoff.cancel();
    setTakeoffOpen(false);
  }
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
                    ? loadingExample || file?.name
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
            {readerWorkspace && <section className="public-sheets" aria-labelledby="public-sheets-title">
              <h2 id="public-sheets-title">Try a public Stoa reference sheet</h2>
              {publicSheets.map(sheet => <div className="public-sheet-row" key={sheet.id}>
                <button disabled={!!busy || !ready} onClick={() => void openExample(sheet)}>
                  <FileText size={15} />{sheet.name}
                </button>
                <a href={sheet.sourceUrl} target="_blank" rel="noreferrer" aria-label={"Source for " + sheet.name}>
                  Source <ArrowUpRight size={12} />
                </a>
              </div>)}
              {publicSheets[0] && <p>{publicSheets[0].attribution} · <a href={publicSheets[0].licenseUrl}
                target="_blank" rel="noreferrer">CC BY-SA 4.0</a></p>}
            </section>}
            {readerWorkspace && <button className="open-spatial-sample" type="button"
              disabled={!!busy || !ready} onClick={() => void openSpatialSample()}>
              <Box size={18} /> Explore the 3D sample <ArrowUpRight size={13} />
            </button>}
          </div>
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
              aria-label={takeoffOpen ? "Show takeoff" : "Show conversation"}
              aria-pressed={mobileTab === "ask"}
              onClick={() => setMobileTab("ask")}
            >
              {takeoffOpen ? <><Ruler size={16}/> Takeoff</> : <><MessageCircle size={16}/> Ask Planproof</>}
            </button>
          </nav>
          <main className="workspace" data-mobile-tab={mobileTab}>
            <div className="drawing-column">
              <PlanCanvas2D
                key={drawing.document_id + ":" + drawing.page}
                spatialSample={readerPage?.originalHash === spatialSampleHash}
                hasTask={!!latestTask?.run}
                taskOpen={!!inspectionId}
                onTaskDetails={() => {
                  closeInspector();
                  if (!inspectionId && latestTask?.run) {
                    setInspectionId(latestTask.id);
                    setLayerError("");
                    setMobileTab("ask");
                  }
                }}
                drawing={drawing}
                name={file?.name}
                liveRun={activeRun}
                followLive={followLive}
                onFollowLive={setFollowLive}
                shapes={shapes}
                comparisonSplit={comparisonSplit}
                focusRegion={focusRegion}
                onManualView={() => {
                  setFollow(false);
                  setFollowLive(false);
                  setFocusRegion(null);
                }}
                result={planResult}
                selectedIndex={selectedEvidence?.index ?? null}
                onSelectEvidence={(index) => {
                  if (evidenceMessage) highlight(evidenceMessage.id, index);
                }}
                zoom={zoom}
                onZoomChange={setZoom}
                isPdf={file?.type === "application/pdf"}
                disabled={!!busy || !ready}
                roomPick={roomPick ? {
                  german:roomPick.german,
                  onPoint: point => void sendMessage(roomPick.question,point),
                  onCancel: () => setRoomPick(null),
                } : undefined}
                onOpenTakeoff={readerWorkspace && savedId && !busy ? openTakeoff : undefined}
                takeoff={takeoffOpen ? takeoff : undefined}
                onCloseTakeoff={closeTakeoff}
                loadingPage={busy === "upload"}
                onPageChange={(nextPage) => {
                  if (file && Number.isInteger(nextPage) && nextPage > 0)
                    void loadFile(file, nextPage);
                }}
              />
            </div>
            {takeoffOpen ? <TakeoffPanel key={drawing.document_id+":"+drawing.page}
              drawing={drawing} filename={file?.name??"drawing"} controls={takeoff}
              suggested={readerTakeoff(planResult,drawing)} onShowPlan={()=>setMobileTab("drawing")} onClose={closeTakeoff}/> : inspectionId ? (
              <EvidenceInspector
                key={inspectionId}
                initialView={inspectionView}
                run={inspected?.run}
                drawing={drawing}
                frame={drawing.source}
                connection={connection}
                onBack={closeInspector}
                onLoadLayers={() => void loadLayers()}
                loading={layersLoading}
                layerError={layerError}
                onFocus={focus}
                onShapes={setShapes}
                onCompare={setComparisonSplit}
                follow={follow}
                onFollow={setFollow}
              />
            ) : (
              <PlanChatbotPanel
                reader={readerWorkspace}
                spatialSample={readerPage?.originalHash === spatialSampleHash}
                rasterSource={readerPage?.rasterPreviewSource}
                canSend={!(readerPage?.representation === "image" && !allowPageImage)}
                readerNotice={readerPage?.representation === "image" ? (
                  <label><input type="checkbox" checked={allowPageImage} disabled={!!busy}
                    onChange={e => setAllowPageImage(e.target.checked)} />
                    Use the displayed {readerPage.payloadWidth} × {readerPage.payloadHeight} page image for questions.
                    Small labels and vector detail can be lost. Highlights map back to page {drawing.page}.
                  </label>
                ) : readerPage?.representation === "page"
                  ? `Page ${drawing.page} will be sent as a single-page PDF with its text and vectors.` :
                    readerPage?.rasterPreviewSource && drawing.width * drawing.height > 16_000_000
                      ? `Questions use the original ${drawing.width} × ${drawing.height} image. Select a source reference to see its original pixels.` :
                    readerPage && (drawing.page_count ?? 1) > 1
                      ? `Asking sends the original PDF to Planproof. Only page ${drawing.page} is read.` : undefined}
                key={drawing.document_id}
                draft={draft}
                onDraftChange={setDraft}
                onNewDrawing={() => input.current?.click()}
                connection={connection}
                onFocusWork={focusWork}
                onInspectCandidates={(id) => {
                  closeInspector();
                  setInspectionId(id);
                  setInspectionView("candidates");
                  setLayerError("");
                  setMobileTab("ask");
                }}
                onDownload={() => void downloadReport()}
                latestTaskId={latestTask?.id}
                downloading={reportState.busy}
                downloadError={reportState.id === latestTask?.id ? reportState.error : undefined}
                messages={messages}
                drawing={drawing}
                busy={busy === "ask"}
                disabled={!!busy || !ready}
                onSendMessage={sendMessage}
                onClearChat={() => {
                  if (!pending.current) {
                    setMessages([]);
                    setSelectedEvidence(null);
                    setRoomPick(null);
                  }
                }}
                selectedEvidence={selectedEvidence}
                onHighlight={highlight}
                onPickRoom={readerWorkspace && (readerPage?.representation !== "image" || allowPageImage) ? pickRoom : undefined}
              />
            )}
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
