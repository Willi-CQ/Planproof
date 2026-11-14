"""DWG nach DXF: austauschbare Schnittstelle zu externen Konvertern.

planproof-cad liest DXF selbst (ezdxf) und enthaelt keinen DWG-Parser. Fuer ``.dwg`` ruft es einen
externen Konverter als eigenen Prozess auf. Welcher, bestimmt der Aufrufer:

* ``NoConverter``       Standard. Kein Konverter; die Fehlermeldung nennt die Auswahlmoeglichkeiten.
* ``OdaFileConverter``  ODA File Converter (Kommandozeile). Pfad aus dem Parameter, aus der
                        Umgebungsvariable ``ODA_FILE_CONVERTER`` oder von ``PATH``.
* ``LibreDwgConverter`` ``dwg2dxf`` aus LibreDWG (GPL). Wird nicht mitgeliefert und nur ueber
                        einen Unterprozess aufgerufen. Pfad aus dem Parameter, ``DWG2DXF`` oder
                        ``PATH``.

Auswahl: Konfigurationsschluessel ``converter`` der Pipeline (``"none"``, ``"oda"``,
``"libredwg"`` oder eine Instanz mit ``convert``), sonst die Umgebungsvariable
``CQPLAN_DWG_CONVERTER`` mit denselben Namen.
Rechtlicher Hintergrund: ``docs/ODA_AND_DWG.md``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, Union, runtime_checkable

ENV_CONVERTER = "CQPLAN_DWG_CONVERTER"
ENV_ODA = "ODA_FILE_CONVERTER"
ENV_LIBREDWG = "DWG2DXF"
TIMEOUT_S = 300
_TAIL = 600  # so viele Zeichen der Ausgabe des Konverters kommen in die Fehlermeldung

Executable = Union[str, os.PathLike, Sequence[str]]


class DwgConversionError(Exception):
    """DWG->DXF-Konvertierung nicht moeglich oder fehlgeschlagen."""


@runtime_checkable
class DwgConverter(Protocol):
    """Schnittstelle: eine DWG-Datei in ein Ausgabeverzeichnis nach DXF umsetzen."""

    def convert(self, dwg_path: Path, out_dir: Path) -> Path:
        """Gibt den Pfad der erzeugten DXF zurueck oder wirft ``DwgConversionError``."""
        ...


def _command_prefix(exe: Executable) -> list[str]:
    """Programm als Kommandoanfang. Eine ``.py``-Datei laeuft mit dem aktuellen Python (Tests,
    Skripte)."""
    if isinstance(exe, (str, os.PathLike)):
        s = os.fspath(exe)
        return [sys.executable, s] if s.lower().endswith(".py") else [s]
    return [str(x) for x in exe]


def _tail(text: Optional[str]) -> str:
    t = (text or "").strip()
    return t[-_TAIL:] if len(t) > _TAIL else t


def _run(name: str, cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise DwgConversionError(f"{name}: Programm nicht gefunden ({cmd[0]}): {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise DwgConversionError(
            f"{name}: Zeitüberschreitung nach {TIMEOUT_S} s ({cmd[0]})"
        ) from exc
    except OSError as exc:
        raise DwgConversionError(f"{name}: Programm nicht startbar ({cmd[0]}): {exc}") from exc


def _fail(name: str, proc: subprocess.CompletedProcess, hint: str = "") -> DwgConversionError:
    detail = _tail(proc.stderr) or _tail(proc.stdout) or "keine Ausgabe"
    return DwgConversionError(
        f"{name} lieferte keine DXF (Rückgabecode {proc.returncode}). Ausgabe: {detail}{hint}"
    )


class NoConverter:
    """Kein Konverter ausgewaehlt (Standard): DXF funktioniert, DWG nicht."""

    name = "none"

    def convert(self, dwg_path: Path, out_dir: Path) -> Path:
        raise DwgConversionError(
            f"DWG-Datei {Path(dwg_path).name} kann nicht gelesen werden: Es ist kein "
            "DWG-Konverter ausgewählt. DXF-Dateien werden ohne Zusatzsoftware gelesen. "
            "Für DWG gibt es drei Wege: "
            "(a) die Zeichnung im CAD-Programm als DXF speichern; "
            "(b) den ODA File Converter installieren und mit converter='oda' bzw. "
            f"{ENV_CONVERTER}=oda wählen (Pfad über {ENV_ODA}); "
            "(c) LibreDWG dwg2dxf installieren und mit converter='libredwg' bzw. "
            f"{ENV_CONVERTER}=libredwg wählen (Pfad über {ENV_LIBREDWG}). "
            "Lizenzlage beider Konverter: docs/ODA_AND_DWG.md."
        )


class OdaFileConverter:
    """ODA File Converter (Kommandozeile). Die Lizenz des Programms liegt beim Betreiber."""

    name = "oda"

    def __init__(
        self, executable: Optional[Executable] = None, out_version: str = "ACAD2018"
    ) -> None:
        self.executable = executable
        self.out_version = out_version

    def _resolve(self) -> Optional[list[str]]:
        if self.executable:
            return _command_prefix(self.executable)
        env = os.environ.get(ENV_ODA)
        if env:
            return _command_prefix(env)
        found = shutil.which("ODAFileConverter")
        return [found] if found else None

    def convert(self, dwg_path: Path, out_dir: Path) -> Path:
        dwg_path, out_dir = Path(dwg_path), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd0 = self._resolve()
        if cmd0 is None:
            raise DwgConversionError(
                "ODA File Converter nicht gefunden: weder Parameter noch Umgebungsvariable "
                f"{ENV_ODA} noch PATH nennen das Programm (docs/ODA_AND_DWG.md)."
            )
        out_dxf = out_dir / (dwg_path.stem + ".dxf")
        # ODAFileConverter <Eingabeordner> <Ausgabeordner> <Version> <Typ> <rekursiv> <audit>
        # [Filter]
        cmd = [
            *cmd0,
            str(dwg_path.parent),
            str(out_dir),
            self.out_version,
            "DXF",
            "0",
            "1",
            dwg_path.name,
        ]
        proc = _run("ODA File Converter", cmd)
        if proc.returncode != 0 or not out_dxf.exists():
            raise _fail("ODA File Converter", proc)
        return out_dxf


class LibreDwgConverter:
    """``dwg2dxf`` aus LibreDWG (GPL). Wird nur als eigener Prozess gestartet, nicht gelinkt."""

    name = "libredwg"

    def __init__(self, executable: Optional[Executable] = None) -> None:
        self.executable = executable

    def _resolve(self) -> Optional[list[str]]:
        if self.executable:
            return _command_prefix(self.executable)
        env = os.environ.get(ENV_LIBREDWG)
        if env:
            return _command_prefix(env)
        found = shutil.which("dwg2dxf")
        return [found] if found else None

    def convert(self, dwg_path: Path, out_dir: Path) -> Path:
        dwg_path, out_dir = Path(dwg_path), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd0 = self._resolve()
        if cmd0 is None:
            raise DwgConversionError(
                "LibreDWG dwg2dxf nicht gefunden: weder Parameter noch Umgebungsvariable "
                f"{ENV_LIBREDWG} noch PATH nennen das Programm (docs/ODA_AND_DWG.md)."
            )
        out_dxf = out_dir / (dwg_path.stem + ".dxf")
        proc = _run("LibreDWG dwg2dxf", [*cmd0, "-y", "-o", str(out_dxf), str(dwg_path)])
        if proc.returncode != 0 or not out_dxf.exists():
            raise _fail("LibreDWG dwg2dxf", proc)
        return out_dxf


_BY_NAME = {
    "": NoConverter,
    "none": NoConverter,
    "no": NoConverter,
    "oda": OdaFileConverter,
    "odafc": OdaFileConverter,
    "libredwg": LibreDwgConverter,
    "dwg2dxf": LibreDwgConverter,
}


def resolve_converter(spec: Any = None) -> DwgConverter:
    """Konverter aus Parameter, sonst aus ``CQPLAN_DWG_CONVERTER``, sonst ``NoConverter``.

    ``spec``: Name (``"none"``, ``"oda"``, ``"libredwg"``), Instanz mit ``convert`` oder ``None``.
    """
    if spec is None:
        spec = os.environ.get(ENV_CONVERTER)
    if spec is None:
        return NoConverter()
    if hasattr(spec, "convert"):
        return spec  # type: ignore[no-any-return]
    key = str(spec).strip().lower()
    cls = _BY_NAME.get(key)
    if cls is None:
        raise DwgConversionError(
            f"Unbekannter DWG-Konverter {spec!r}; erlaubt: none, oda, libredwg "
            f"(Parameter converter bzw. Umgebungsvariable {ENV_CONVERTER})."
        )
    return cls()


def convert_dwg(dwg_path: Path, out_dir: Path, converter: Any = None) -> Path:
    """Wählt den Konverter (siehe ``resolve_converter``), setzt um und prueft das Ergebnis."""
    conv = resolve_converter(converter)
    out = Path(conv.convert(Path(dwg_path), Path(out_dir)))
    if not out.exists():
        raise DwgConversionError(
            f"Konverter {type(conv).__name__} meldete {out}, die Datei existiert nicht."
        )
    return out


__all__ = [
    "DwgConversionError",
    "DwgConverter",
    "NoConverter",
    "OdaFileConverter",
    "LibreDwgConverter",
    "resolve_converter",
    "convert_dwg",
    "ENV_CONVERTER",
]
