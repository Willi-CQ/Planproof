import { useEffect, useRef } from "react";
import { FileText, Plus, Trash2, X } from "lucide-react";
import type { LibraryEntry } from "../lib/library";

export function DrawingLibrary({
  entries,
  active,
  onClose,
  onOpen,
  onRemove,
  onAdd,
}: {
  entries: LibraryEntry[];
  active: string | null;
  onClose: () => void;
  onOpen: (id: string) => void;
  onRemove: (id: string) => void;
  onAdd: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  return (
    <dialog
      ref={dialog}
      className="library-dialog"
      aria-labelledby="library-title"
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="library-content">
        <header>
          <div>
            <h2 id="library-title">Your drawings</h2>
            <p>Saved in this browser.</p>
          </div>
          <button
            className="icon-button"
            aria-label="Close drawings"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </header>
        <div className="library-list">
          {!entries.length && (
            <p className="library-empty">Drawings you open will appear here.</p>
          )}
          {entries.map((entry) => (
            <div className="library-row" key={entry.id}>
              <button
                className="library-open"
                onClick={() => onOpen(entry.id)}
                aria-label={`Open ${entry.name}`}
                aria-current={entry.id === active ? "true" : undefined}
              >
                <FileText size={21} />
                <span>
                  <strong>{entry.name}</strong>
                  <small>
                    Page {entry.page}
                    {entry.id === active ? " · Current" : ""}
                  </small>
                </span>
              </button>
              <button
                className="icon-button"
                aria-label={`Remove ${entry.name} from saved drawings`}
                title="Remove local copy"
                onClick={() => onRemove(entry.id)}
              >
                <Trash2 size={16} />
              </button>
            </div>
          ))}
        </div>
        <button className="subtle-button library-add" onClick={onAdd}>
          <Plus size={16} />
          New drawing
        </button>
      </div>
    </dialog>
  );
}
