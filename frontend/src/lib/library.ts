import type { Message } from "../components/PlanChatbotPanel";
import type { Drawing } from "./api";
import { emptyTakeoff, type SheetTakeoff } from "./takeoff";

// Keep the local backend's saved sessions alongside the product reader's per-page histories.
export const readerWorkspace = !(["localhost", "127.0.0.1", "[::1]"].includes(location.hostname) &&
  new URLSearchParams(location.search).get("workspace") === "agent");
const activeKey = readerWorkspace ? "reader-active" : "active";

export interface SavedDrawing {
  id: string;
  file: File;
  drawing: Drawing;
  messages: Message[];
  draft: string;
  zoom: number;
  selectedEvidence: { message: string; index: number } | null;
  interrupted: boolean;
  updated: number;
}
export interface LibraryEntry {
  id: string;
  name: string;
  page: number;
  updated: number;
}

// Originals and answers stay in this browser. No credentials or model calls.
let connection: Promise<IDBDatabase> | undefined;
function database() {
  return (connection ??= new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open("planproof-library", 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore("drawings", { keyPath: "id" });
      request.result.createObjectStore("state");
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => {
      connection = undefined;
      reject(request.error);
    };
    request.onblocked = () =>
      reject(new Error("Close other Planproof tabs to open the library."));
  }));
}
async function write(action: (tx: IDBTransaction) => void) {
  const db = await database();
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(["drawings", "state"], "readwrite");
    tx.oncomplete = () => resolve();
    tx.onabort = tx.onerror = () =>
      reject(tx.error ?? new Error("Local save failed"));
    action(tx);
  });
}
export async function readLibrary(): Promise<{
  entries: LibraryEntry[];
  active: SavedDrawing | null;
}> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["drawings", "state"]);
    const records = tx.objectStore("drawings").getAll();
    const active = tx.objectStore("state").get(activeKey);
    tx.oncomplete = () => {
      const all = (records.result as SavedDrawing[]).filter(r => r.id.startsWith("reader-") === readerWorkspace);
      resolve({
        entries: all
          .map((r) => ({
            id: r.id,
            name: r.file.name,
            page: r.drawing.page,
            updated: r.updated,
          }))
          .sort((a, b) => b.updated - a.updated),
        active: all.find((r) => r.id === active.result) ?? null,
      });
    };
    tx.onabort = tx.onerror = () => reject(tx.error);
  });
}
export async function getSavedDrawing(
  id: string,
): Promise<SavedDrawing | undefined> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const request = db.transaction("drawings").objectStore("drawings").get(id);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
export function saveDrawing(record: SavedDrawing) {
  return write((tx) => {
    tx.objectStore("drawings").put(record);
    tx.objectStore("state").put(record.id, activeKey);
  });
}
export function leaveDrawing() {
  return write((tx) => {
    tx.objectStore("state").delete(activeKey);
  });
}
export function removeDrawing(id: string, active: boolean) {
  return write((tx) => {
    tx.objectStore("drawings").delete(id);
    tx.objectStore("state").delete(takeoffKey(id));
    if (active) tx.objectStore("state").delete(activeKey);
  });
}
const takeoffKey = (id:string) => "takeoff:"+id;

export async function readTakeoff(id:string):Promise<SheetTakeoff> {
  const db=await database();
  return new Promise((resolve,reject)=>{
    const get=db.transaction("state").objectStore("state").get(takeoffKey(id));
    get.onsuccess=()=>resolve(get.result ?? emptyTakeoff(id));
    get.onerror=()=>reject(get.error);
  });
}

export async function updateTakeoff(id:string, change:(current:SheetTakeoff)=>SheetTakeoff):Promise<SheetTakeoff> {
  const db=await database();
  return new Promise((resolve,reject)=>{
    const tx=db.transaction(["drawings","state"],"readwrite");
    const state=tx.objectStore("state");
    let saved:SheetTakeoff;
    const drawing=tx.objectStore("drawings").getKey(id);
    drawing.onsuccess=()=>{
      if (drawing.result===undefined) {tx.abort();return;}
      const get=state.get(takeoffKey(id));
      get.onsuccess=()=>{
        try {
          saved=change(get.result ?? emptyTakeoff(id));
          state.put(saved,takeoffKey(id));
        } catch (error) {
          reject(error);
          tx.abort();
        }
      };
    };
    tx.oncomplete=()=>resolve(saved);
    tx.onabort=()=>reject(tx.error ?? new Error("The drawing could not be saved"));
  });
}
export async function drawingId(file: File) {
  const bytes = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(bytes), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}
