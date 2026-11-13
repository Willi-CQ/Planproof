"""Swiss Dwellings v3.0.0: geometries.csv streamen und indexieren.

Die Rohdaten (ZIP, CSV) liegen ausserhalb des Repos. Alles hier liest sie nur als
Daten (csv-Modul, keine Ausfuehrung). Die Datei ist ca. 1,1 GB gross und wird nie
komplett in den Speicher geladen.

Quelle: Standfest et al., Archilyse AG, "Swiss Dwellings", Zenodo, DOI
10.5281/zenodo.7788422, Lizenz CC BY 4.0.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

EXPECTED_COLUMNS = [
    "apartment_id",
    "site_id",
    "building_id",
    "plan_id",
    "floor_id",
    "unit_id",
    "area_id",
    "unit_usage",
    "entity_type",
    "entity_subtype",
    "geometry",
    "elevation",
    "height",
]

FloorKey = Tuple[str, str]


def iter_rows(csv_path: str | Path) -> Iterator[Dict[str, str]]:
    """Liefert die Zeilen der geometries.csv als Dicts (Streaming)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"unerwartete Spalten: {reader.fieldnames}")
        yield from reader


def _row_hash(row: Dict[str, str]) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(row["entity_type"].encode())
    h.update(b"\0")
    h.update(row["entity_subtype"].encode())
    h.update(b"\0")
    h.update(row["geometry"].encode())
    return int.from_bytes(h.digest(), "big")


def _geometry_kind(wkt: str) -> str:
    if not wkt:
        return "EMPTY"
    head = wkt.lstrip()[:20].upper()
    for name in (
        "MULTIPOLYGON",
        "POLYGON",
        "MULTILINESTRING",
        "LINESTRING",
        "MULTIPOINT",
        "POINT",
        "GEOMETRYCOLLECTION",
        "LINEARRING",
    ):
        if head.startswith(name):
            return name
    return "OTHER"


def scan_index(csv_path: str | Path, progress: bool = False) -> Dict[str, Any]:
    """Ein Durchlauf ueber die CSV: Zaehlungen und ein Index je Geschoss.

    Rueckgabe:
      stats  - Rohdaten-Zaehlungen (Zeilen, Typen, Geometrietypen, IDs, Nutzung, Hoehen)
      floors - {(plan_id, floor_id): {"n": {typ/subtyp: Anzahl}, "hash": int}}
        "hash" ist die ordnungsunabhaengige Summe der Zeilen-Hashes (Typ, Subtyp,
        WKT) und dient dem Nachweis, ob zwei Geschosse desselben plan_id identisch sind.
    """
    type_counts: Counter = Counter()
    geom_counts: Counter = Counter()
    usage_counts: Counter = Counter()
    height_counts: Dict[str, Counter] = defaultdict(Counter)
    elevation_counts: Counter = Counter()
    ids: Dict[str, set] = {k: set() for k in ("site_id", "building_id", "plan_id", "floor_id")}
    floors: Dict[FloorKey, Dict[str, Any]] = {}
    n_rows = 0
    mask = (1 << 64) - 1
    for row in iter_rows(csv_path):
        n_rows += 1
        et, es = row["entity_type"], row["entity_subtype"]
        type_counts[(et, es)] += 1
        geom_counts[(et, _geometry_kind(row["geometry"]))] += 1
        usage_counts[row["unit_usage"]] += 1
        for k in ids:
            ids[k].add(row[k])
        if et in ("separator", "opening"):
            height_counts[f"{et}/{es}"][row["height"]] += 1
        if et == "separator" and es == "WALL":
            elevation_counts[row["elevation"]] += 1
        key = (row["plan_id"], row["floor_id"])
        fl = floors.get(key)
        if fl is None:
            fl = {"n": Counter(), "hash": 0}
            floors[key] = fl
        fl["n"][f"{et}/{es}"] += 1
        fl["hash"] = (fl["hash"] + _row_hash(row)) & mask
        if progress and n_rows % 500000 == 0:
            print(f"  {n_rows} Zeilen", file=sys.stderr, flush=True)

    stats = {
        "rows": n_rows,
        "entity_counts": {
            f"{a}/{b}": c for (a, b), c in sorted(type_counts.items(), key=lambda kv: -kv[1])
        },
        "geometry_kinds": {f"{a}/{b}": c for (a, b), c in sorted(geom_counts.items())},
        "unit_usage": dict(usage_counts.most_common()),
        "unique": {k: len(v) for k, v in ids.items()},
        "wall_elevation_top": elevation_counts.most_common(5),
        "heights_top": {k: c.most_common(5) for k, c in sorted(height_counts.items())},
    }
    return {"stats": stats, "floors": floors}


def plan_floor_table(
    floors: Dict[FloorKey, Dict[str, Any]],
) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """Gruppiert die Geschosse nach plan_id, aufsteigend nach floor_id (numerisch)."""
    by_plan: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for (pid, fid), v in floors.items():
        by_plan[pid].append((fid, v))
    for pid in by_plan:
        by_plan[pid].sort(key=lambda t: int(t[0]))
    return dict(by_plan)


def repetition_stats(floors: Dict[FloorKey, Dict[str, Any]]) -> Dict[str, Any]:
    """Wie oft teilen sich mehrere Geschosse einen plan_id, und sind sie dann identisch?"""
    by_plan = plan_floor_table(floors)
    multi = {p: fl for p, fl in by_plan.items() if len(fl) > 1}
    identical = sum(1 for fl in multi.values() if len({v["hash"] for _, v in fl}) == 1)
    extra_floors = sum(len(fl) - 1 for fl in multi.values())
    # identische Plaene unter VERSCHIEDENEN plan_id (gleiche Geometrie-Summe)
    sig_to_plans: Dict[int, set] = defaultdict(set)
    for p, fl in by_plan.items():
        sig_to_plans[fl[0][1]["hash"]].add(p)
    dup_across = sum(len(s) - 1 for s in sig_to_plans.values() if len(s) > 1)
    return {
        "plans": len(by_plan),
        "floors": len(floors),
        "plans_with_multiple_floors": len(multi),
        "plans_with_multiple_floors_all_identical": identical,
        "floors_that_repeat_another_floor_of_same_plan": extra_floors,
        "plan_ids_identical_to_an_other_plan_id_geometry": dup_across,
    }


def load_floor_rows(
    csv_path: str | Path, wanted: Dict[str, str]
) -> Dict[str, List[Dict[str, str]]]:
    """Laedt in einem Durchlauf alle Zeilen der gewuenschten Geschosse.

    wanted: {plan_id: floor_id}. Rueckgabe: {plan_id: [Zeilen]}.
    """
    out: Dict[str, List[Dict[str, str]]] = {p: [] for p in wanted}
    for row in iter_rows(csv_path):
        p = row["plan_id"]
        if p in wanted and row["floor_id"] == wanted[p]:
            out[p].append(row)
    return out
