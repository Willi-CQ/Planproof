from __future__ import annotations
import os, sys
from pathlib import Path

def _candidate_roots() -> list[Path]:
    cands: list[Path] = []

    # 1) PyInstaller
    if hasattr(sys, "_MEIPASS"):
        cands.append(Path(sys._MEIPASS))

    # 2) Ordner der aktuellen Datei (src/ai/.. -> hoch)
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        cands.append(p)

    # 3) Ordner des gestarteten Scripts / exe
    try:
        cands.append(Path(sys.argv[0]).resolve().parent)
    except Exception:
        pass

    # 4) CWD als letzter Fallback
    cands.append(Path.cwd().resolve())

    # duplicates entfernen
    out = []
    seen = set()
    for p in cands:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def app_root_dir() -> Path:
    # Optionaler Override für Debug/Support
    env = os.environ.get("PLAN_APP_ROOT")
    if env:
        return Path(env).resolve()

    for root in _candidate_roots():
        exe = root / "resources" / "tesseract" / "tesseract.exe"
        if exe.exists():
            return root

        # falls resources unter src/ liegt (manche Projekte machen das)
        exe2 = root / "src" / "resources" / "tesseract" / "tesseract.exe"
        if exe2.exists():
            return root / "src"

    # letzter fallback: bisherige Logik
    return Path(__file__).resolve().parents[2]

_CACHED_TESS_SETUP: dict | None = None

def configure_tesseract() -> dict:
    global _CACHED_TESS_SETUP
    if _CACHED_TESS_SETUP is not None:
        return _CACHED_TESS_SETUP

    try:
        import pytesseract
    except Exception as e:
        _CACHED_TESS_SETUP = {"ok": False, "reason": "pytesseract_missing", "error": str(e)}
        return _CACHED_TESS_SETUP

    root = app_root_dir()
    tess_root = root / "resources" / "tesseract"
    if not tess_root.exists():
        tess_root = root / "src" / "resources" / "tesseract"

    exe = tess_root / "tesseract.exe"
    tessdata = tess_root / "tessdata"

    if not exe.exists():
        _CACHED_TESS_SETUP = {"ok": False, "reason": "tesseract_exe_missing", "path": str(exe)}
        return _CACHED_TESS_SETUP
    if not tessdata.exists():
        _CACHED_TESS_SETUP = {"ok": False, "reason": "tessdata_missing", "path": str(tessdata)}
        return _CACHED_TESS_SETUP

    # ---- NEW: traineddata wirklich prüfen (mindestens eng) ----
    eng = tessdata / "eng.traineddata"
    deu = tessdata / "deu.traineddata"
    available_langs = []
    if eng.exists():
        available_langs.append("eng")
    if deu.exists():
        available_langs.append("deu")

    if not eng.exists():
        _CACHED_TESS_SETUP = {
            "ok": False,
            "reason": "traineddata_missing",
            "missing": ["eng.traineddata"],
            "tessdata": str(tessdata),
        }
        return _CACHED_TESS_SETUP

    # Windows: DLL search path + PATH absichern
    if os.name == "nt":
        try:
            os.add_dll_directory(str(tess_root))
        except Exception:
            pass
        os.environ["PATH"] = str(tess_root) + os.pathsep + os.environ.get("PATH", "")

    pytesseract.pytesseract.tesseract_cmd = str(exe)

    # IMPORTANT:
    # TESSDATA_PREFIX soll ohne Quotes/Trailing-Slashes gesetzt sein
    os.environ["TESSDATA_PREFIX"] = str(tessdata).rstrip("\\/")

    _CACHED_TESS_SETUP = {
        "ok": True,
        "tesseract_cmd": str(exe),
        "tessdata": str(tessdata),
        "root": str(root),
        "available_langs": available_langs,
    }
    
    return _CACHED_TESS_SETUP
