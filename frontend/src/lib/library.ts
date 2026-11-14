import type { Message } from "../components/PlanChatbotPanel";
import type { Drawing } from "./api";

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
    const active = tx.objectStore("state").get("active");
    tx.oncomplete = () => {
      const all = records.result as SavedDrawing[];
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
    tx.objectStore("state").put(record.id, "active");
  });
}
export function leaveDrawing() {
  return write((tx) => {
    tx.objectStore("state").delete("active");
  });
}
export function removeDrawing(id: string, active: boolean) {
  return write((tx) => {
    tx.objectStore("drawings").delete(id);
    if (active) tx.objectStore("state").delete("active");
  });
}
export async function drawingId(file: File) {
  const bytes = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(bytes), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}
