"""Step 1: Validate scan input file."""

from __future__ import annotations

import zipfile
from pathlib import Path

from ..point_cloud_io import SUPPORTED_EXTENSIONS

MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

# Issue #47: identisch zu ``planproof_worker/runner._validate_zip_safe``, das
# nur das AEUSSERE Archiv sieht. Werte bewusst gleich gehalten.
MAX_ZIP_TOTAL_UNCOMPRESSED = 4 * 1024 * 1024 * 1024  # 4 GiB
MAX_ZIP_MEMBERS = 10_000


class ValidateInputStep:
    name = "Validate input"

    def run(self, ctx, state):
        p = Path(state.input_path)

        if not p.exists():
            raise FileNotFoundError(f"Scan file not found: {state.input_path}")
        if not p.is_file():
            raise ValueError(f"Not a file: {state.input_path}")

        ext = p.suffix.lower()

        # Handle ZIP: extract and find first supported file
        if ext == ".zip":
            state.input_path = _extract_scan_from_zip(p, state.work_dir, ctx)
            p = Path(state.input_path)
            ext = p.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS - {'.zip'}))}"
            )

        file_size = p.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File too large: {file_size / 1e9:.1f} GB "
                f"(max {MAX_FILE_SIZE_BYTES / 1e9:.0f} GB)"
            )

        state.metrics["input_file_size_mb"] = round(file_size / 1e6, 2)
        state.metrics["input_format"] = ext
        ctx.log(f"Input: {p.name} ({state.metrics['input_file_size_mb']} MB, {ext})")
        return state


def _assert_zip_safe(zf: zipfile.ZipFile) -> None:
    """ZIP-Bomben-Schutz fuer das INNERE Archiv (Issue #47).

    Der Worker prueft das AEUSSERE Archiv bereits
    (``planproof_worker/runner._validate_zip_safe``, mit Regressionstest). Ein
    ``outer.zip`` mit genau einem 10-MB-Eintrag ``a.zip`` passiert dort
    unauffaellig — und weil ``.zip`` nicht in ``_SCAN_EXTENSIONS`` steht, faellt
    die innere Datei auf ``input_paths[0]`` durch und landet hier. Dieses zweite
    ``extractall`` hatte weder Member- noch Groessen- noch Ratio-Grenze und
    schrieb, bis der Ephemeral-Storage der Replica voll war.

    Bewusst dieselben Zahlen wie im Worker: zwei Deckel, die auseinanderlaufen,
    sind schlimmer als einer.
    """
    infos = zf.infolist()
    if len(infos) > MAX_ZIP_MEMBERS:
        raise ValueError(
            f"ZIP enthaelt {len(infos)} Eintraege (max {MAX_ZIP_MEMBERS})."
        )
    gesamt = sum(i.file_size for i in infos)
    if gesamt > MAX_ZIP_TOTAL_UNCOMPRESSED:
        raise ValueError(
            f"ZIP entpackt zu {gesamt / 1e9:.1f} GB "
            f"(max {MAX_ZIP_TOTAL_UNCOMPRESSED / 1e9:.0f} GB)."
        )


def _extract_scan_from_zip(zip_path: Path, work_dir: str, ctx) -> str:
    """Extract ZIP and return path to first supported scan file."""
    extract_dir = Path(work_dir) / "zip_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        _assert_zip_safe(zf)
        zf.extractall(extract_dir)

    scan_exts = SUPPORTED_EXTENSIONS - {".zip"}
    for f in sorted(extract_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in scan_exts:
            ctx.log(f"Extracted scan from ZIP: {f.name}")
            return str(f)

    raise ValueError(
        f"ZIP contains no supported scan files. "
        f"Expected: {', '.join(sorted(scan_exts))}"
    )
