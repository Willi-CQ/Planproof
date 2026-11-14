"""Baut das DXF-Goldset aus Swiss Dwellings v3.0.0 (reproduzierbar).

    python build_goldset.py --csv C:/tmp/swiss-dwellings/raw/geometries.csv \
        --index-cache C:/tmp/swiss-dwellings/floor_index.json

Ablauf (Details in GROUND_TRUTH.md):
  1. Index je Geschoss (ein Durchlauf ueber die CSV, ca. 1,5 Minuten, wird zwischengespeichert)
  2. je plan_id das Geschoss mit der kleinsten floor_id (Wiederholungen entfallen), Plaene mit
     identischer Geometrie unter anderer plan_id entfallen
  3. Schichtung nach Groesse (Zahl der Wandzeilen) in drei Klassen, innerhalb der Klasse
     feste Zufallsreihenfolge (Seed), Pruefung in dieser Reihenfolge mit sd_plan
  4. Auswahl aus den zulaessigen Plaenen, je Klasse ausgewogen nach Raumzahl
  5. Rendern in zwei Zeichenkonventionen je Plan, Groessenbudget, Schreiben von DXF, Schwester-
     JSON, manifest.json, ATTRIBUTION.md, selection.json

Bei gleichem Seed und gleichen Rohdaten entstehen byte-identische Dateien.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goldset import render_dxf, sd_data, sd_plan  # noqa: E402
from goldset.inspect_raw import get_scan  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

SEED = 20201020
# (Name, min Wandzeilen, max Wandzeilen, Soll-Zahl)
STRATA = [("klein", 20, 60, 16), ("mittel", 61, 110, 16), ("gross", 111, 170, 12)]
ROOM_BINS = [(1, 5), (6, 12), (13, 10**6)]
BUDGET_BYTES = 5_800_000
MIN_PLANS = 40
TARGET_ELIGIBLE_FACTOR = 2.5
BATCH = 100
# Mindestens so viele Kandidaten je Klasse werden geprueft, damit die Ausschlusszaehlungen etwas aussagen
MIN_EVALUATED_PER_STRATUM = 300
MAX_EVALUATED_PER_STRATUM = 1500

SOURCE = {
    "datensatz": "Swiss Dwellings: A large dataset of apartment models including aggregated geolocation-based "
                 "simulation results covering viewshed, natural light, traffic noise, centrality and geometric analysis",
    "version": "3.0.0",
    "doi": "10.5281/zenodo.7788422",
    "urheber": "Archilyse AG (Matthias Standfest, Michael Franzen, Yvonne Schröder, Luis Gonzalez Medina, "
               "Yarilo Villanueva Hernandez, Jan Hendrik Buck, Yen-Ling Tan, Milena Niedzwiecka, Rachele Colmegna)",
    "lizenz": "CC BY 4.0",
    "lizenz_url": "https://creativecommons.org/licenses/by/4.0/",
    "aenderungen": "Auswahl einzelner Geschosse, Drehung auf die Hauptrichtung der Waende, Verschiebung, "
                   "Zusammenfassen sich ueberlappender Wandrechtecke, Zusammenfassen offener Nachbarraeume, "
                   "Umwandlung in DXF mit selbst gewaehlten Layern, Zusatz von Stoerelementen",
}

SYNTHETIC_NOTE = (
    "Die Zeichenkonventionen sind synthetisch gewaehlt. Das Goldset beweist die Pipeline auf "
    "Swiss-Dwellings-Geometrie mit selbst gewaehlten Layern, nicht auf echten Buero-Exporten."
)


def room_bin(n: int) -> int:
    for i, (lo, hi) in enumerate(ROOM_BINS):
        if lo <= n <= hi:
            return i
    return 0


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def plan_key(plan_id: str) -> str:
    return f"sd_p{int(plan_id):05d}"


def _cm(v: Any) -> Any:
    """Rundet Koordinaten in verschachtelten Listen auf Zentimeter (GT-Geometrie fuer den Abgleich)."""
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, list):
        return [_cm(x) for x in v]
    return v


def build_gt_json(plan: Dict[str, Any], variante: Dict[str, Any], key: str, dxf_name: str,
                  dxf_sha: str, info: Dict[str, Any], floors_repeat: int) -> Dict[str, Any]:
    kv = render_dxf.KONVENTIONEN[variante["konvention"]]
    gt = plan["gt"]
    return {
        "schema": "planproof-goldset-gt/1",
        "plan_key": key,
        "datei": dxf_name,
        "dxf_sha256": dxf_sha,
        "quelle": {
            "datensatz": SOURCE["datensatz"], "version": SOURCE["version"], "doi": SOURCE["doi"],
            "lizenz": SOURCE["lizenz"], "plan_id": plan["plan_id"], "floor_id": plan["floor_id"],
            "geschosse_mit_diesem_plan_id": floors_repeat,
        },
        "konvention": {
            "id": variante["konvention"],
            "name": kv["name"],
            "variante": variante["id"],
            "wand_ausfuehrung": variante["wand"],
            "unterbrechung_an_oeffnungen": variante["unterbrechung"],
            "schraffur": variante["schraffur"],
            "layer": {k: kv[k] for k in ("wand_aussen", "wand_innen", "tuer", "fenster", "raum", "raumpolygon",
                                          "bemassung", "text", "schraffur", "ausstattung", "stuetze",
                                          "gelaender", "treppe")},
            "blockname_tuer": kv["block_tuer"],
            "blockname_fenster": kv["block_fenster"],
            "oeffnungen_auf_wandlayer": kv["oeffnung_auf_wandlayer"],
            "synthetisch": True,
            "hinweis": SYNTHETIC_NOTE,
        },
        "einheiten": {
            "dxf": "mm", "insunits": 4, "gt": "m",
            "koordinaten": "GT-Koordinaten = DXF-Koordinaten / 1000, gleicher Ursprung",
        },
        "gt": {
            "wandlaenge_achse_m": gt["wandlaenge_achse_m"],
            "wandlaenge_flaechenkante_m": gt["wandlaenge_flaechenkante_m"],
            "wandlaenge_summe_rechtecke_m": gt["wandlaenge_summe_rechtecke_m"],
            "n_waende": gt["n_waende"],
            "tueren": gt["tueren"],
            "tueren_davon_haustuer": gt["tueren_davon_haustuer"],
            "fenster": gt["fenster"],
            "raeume": gt["raeume"],
            "raumflaeche_summe_m2": gt["raumflaeche_summe_m2"],
            "nichtraeume_eingeschlossen": gt["nichtraeume_eingeschlossen"],
            "bbox_m": _cm(plan["bbox"]),
            "geometrie_genauigkeit": "Koordinaten der Listen auf 1 cm gerundet; Flaechen und Laengen exakt",
            "waende": [{"p0": _cm(w["p0"]), "p1": _cm(w["p1"]), "T": w["T"], "aussen": bool(w.get("aussen"))}
                       for w in plan["waende"]],
            "oeffnungen": [{"id": o["id"], "art": o["art"], "sd_typ": o["sd_typ"], "mitte": _cm(o["mitte"]),
                            "richtung": o["richtung"], "breite": o["breite"], "dicke": o["dicke"],
                            "wand": o["wand"]} for o in plan["oeffnungen"]],
            "raeume_liste": [{"id": r["id"], "label": r["label"], "sd_typen": r["sd_typen"],
                              "flaeche_m2": r["flaeche_m2"], "teile": r["teile"], "punkt": _cm(r["punkt"]),
                              "schwerpunkt": _cm(r["schwerpunkt"]), "polygon": _cm(r["polygon"]),
                              **({"loecher": _cm(r["loecher"])} if r["loecher"] else {})}
                             for r in plan["raeume"]],
            "nichtraeume": [{"sd_typ": n["sd_typ"], "flaeche_m2": n["flaeche_m2"], "polygon": _cm(n["polygon"])}
                            for n in plan["nichtraeume"]],
        },
        "dxf": info,
    }


def _dump(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False) + "\n").encode("utf-8")


def evaluate_candidates(csv: Path, order: Dict[str, List[str]], floor_of: Dict[str, str],
                        quota: Dict[str, int], log) -> Tuple[Dict[str, sd_plan.Analyse], Dict[str, Counter]]:
    """Prueft die Kandidaten je Klasse in fester Reihenfolge, bis genug zulaessige gefunden sind."""
    pos = {s: 0 for s in order}
    results: Dict[str, sd_plan.Analyse] = {}
    reasons: Dict[str, Counter] = {s: Counter() for s in order}
    eligible = {s: 0 for s in order}
    while True:
        batch: Dict[str, str] = {}
        batch_stratum: Dict[str, str] = {}
        for s, ids in order.items():
            enough = eligible[s] >= TARGET_ELIGIBLE_FACTOR * quota[s] and pos[s] >= MIN_EVALUATED_PER_STRATUM
            if enough or pos[s] >= min(len(ids), MAX_EVALUATED_PER_STRATUM):
                continue
            take = ids[pos[s]:pos[s] + BATCH]
            pos[s] += len(take)
            for pid in take:
                batch[pid] = floor_of[pid]
                batch_stratum[pid] = s
        if not batch:
            break
        log(f"  Durchlauf ueber die CSV fuer {len(batch)} Plaene ...")
        rows = sd_data.load_floor_rows(csv, batch)
        for pid in sorted(batch, key=int):
            a = sd_plan.analysiere_geschoss(rows[pid], pid, batch[pid])
            results[pid] = a
            s = batch_stratum[pid]
            if a.ok:
                eligible[s] += 1
                reasons[s]["zulaessig"] += 1
            else:
                for g in a.gruende:
                    reasons[s][g] += 1
                reasons[s]["primaer:" + a.gruende[0]] += 1
                reasons[s]["ausgeschlossen_gesamt"] += 1
        log(f"  zulaessig bisher: {eligible}")
    return results, reasons


def _pct(vals: List[float], qs=(5, 50, 95)) -> Dict[str, float]:
    import numpy as np

    if not vals:
        return {}
    arr = np.asarray(vals, dtype=float)
    return {f"p{q}": round(float(np.percentile(arr, q)), 3) for q in qs}


def aggregate_diag(results: Dict[str, sd_plan.Analyse]) -> Dict[str, Any]:
    """Zaehlungen ueber alle geprueften Plaene: belegt die Aussagen in GROUND_TRUTH.md."""
    ecken: Dict[str, Counter] = defaultdict(Counter)
    dup: Counter = Counter()
    roh: Counter = Counter()
    dicken: List[float] = []
    in_wand: List[float] = []
    breiten: Dict[str, List[float]] = {"tuer": [], "fenster": []}
    spanne: List[float] = []
    ortho: List[float] = []
    dreh: List[float] = []
    nonrect_walls = n_walls_rect = 0
    plans_with_merge = merged = rooms = 0
    typ_areas: Dict[str, List[float]] = defaultdict(list)
    plan_rooms: List[int] = []
    diag_plans = [a.diag for a in results.values() if "eckpunkte" in a.diag and "wand_dicken" in a.diag]
    for d in diag_plans:
        for g, c in d["eckpunkte"].items():
            ecken[g].update({int(k): v for k, v in c.items()})
        dup.update(d.get("exakte_duplikate", {}))
        roh.update(d["zeilen_roh"])
        dicken += d["wand_dicken"]
        in_wand += d["oeffnung_in_wand_anteil"]
        for art, w, _t in d["oeffnung_breite_tuer_fenster"]:
            breiten[art].append(w)
        if d.get("laengen_spanne_pct") is not None:
            spanne.append(d["laengen_spanne_pct"])
        ortho.append(d["anteil_rechtwinklig_2grad"])
        dreh.append(abs(d["drehung_grad"]))
        nonrect_walls += d["wand_nicht_rechteckig"]
        n_walls_rect += d["wand_rechtecke"]
        merged += d["raeume_aus_offenen_nachbarn"]
        rooms += d["raeume"]
        plans_with_merge += 1 if d["raeume_aus_offenen_nachbarn"] else 0
        plan_rooms.append(d["raeume"])
        for s, a in d["raum_typ_flaeche"]:
            typ_areas[s].append(a)
    n = len(diag_plans)
    top_types = sorted(typ_areas, key=lambda s: -len(typ_areas[s]))[:10]
    return {
        "plaene_ausgewertet": n,
        "eckpunkte_je_polygon": {g: dict(sorted(c.items())) for g, c in sorted(ecken.items())},
        "zeilen_roh_summe": dict(sorted(roh.items(), key=lambda kv: -kv[1])[:12]),
        "exakte_duplikate_summe": dict(dup),
        "wanddicke_m_je_wand": _pct(dicken, (5, 25, 50, 75, 95)),
        "oeffnung_flaechenanteil_in_wand": {
            **_pct(in_wand, (1, 5, 50)),
            "anteil_oeffnungen_ueber_90_pct": round(sum(1 for x in in_wand if x >= 0.9) / max(1, len(in_wand)), 4),
            "anzahl": len(in_wand),
        },
        "oeffnungsbreite_m": {k: _pct(v, (5, 50, 95)) for k, v in breiten.items()},
        "wandlaenge_achse_gegen_flaechenkante_pct": {**_pct(spanne, (50, 90, 99)), "max": round(max(spanne), 3) if spanne else None},
        "anteil_wandlaenge_rechtwinklig_2grad_je_plan": _pct(ortho, (5, 50, 95)),
        "drehung_grad_betrag": _pct(dreh, (5, 50, 95)),
        "plaene_mit_drehung_ueber_1_grad": sum(1 for x in dreh if x > 1.0),
        "waende_nicht_rechteckig": {"anzahl": nonrect_walls, "von": n_walls_rect},
        "raeume_je_plan": _pct([float(x) for x in plan_rooms], (5, 50, 95)),
        "raeume_aus_offenen_nachbarn": {"raeume_gesamt": rooms, "davon_zusammengefasst": merged,
                                        "plaene_mit_zusammenfassung": plans_with_merge},
        "raumflaeche_m2_je_sd_typ": {s: {"n": len(typ_areas[s]), **_pct(typ_areas[s], (5, 50, 95))} for s in top_types},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--index-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--budget-bytes", type=int, default=BUDGET_BYTES)
    ap.add_argument("--min-plans", type=int, default=MIN_PLANS)
    args = ap.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    t0 = time.time()
    log("1/6 Index")
    scan = get_scan(args.csv, args.index_cache)
    by_plan = sd_data.plan_floor_table(scan["floors"])
    rep = sd_data.repetition_stats(scan["floors"])

    log("2/6 Schichten")
    floor_of: Dict[str, str] = {}
    n_walls: Dict[str, int] = {}
    seen_hash: Dict[int, str] = {}
    pre: Counter = Counter()
    strata_ids: Dict[str, List[str]] = {s[0]: [] for s in STRATA}
    for pid in sorted(by_plan, key=int):
        fid, fl = by_plan[pid][0]
        floor_of[pid] = fid
        nw = fl["n"].get("separator/WALL", 0)
        n_walls[pid] = nw
        pre["plaene_gesamt"] += 1
        if fl["hash"] in seen_hash:
            pre["gleiche_geometrie_wie_anderer_plan_id"] += 1
            continue
        seen_hash[fl["hash"]] = pid
        if not any(k.startswith("area/") for k in fl["n"]):
            pre["keine_flaechen"] += 1
            continue
        if nw < STRATA[0][1]:
            pre["zu_wenige_wandzeilen_unter_%d" % STRATA[0][1]] += 1
            continue
        if nw > STRATA[-1][2]:
            pre["zu_viele_wandzeilen_ueber_%d" % STRATA[-1][2]] += 1
            continue
        for name, lo, hi, _q in STRATA:
            if lo <= nw <= hi:
                strata_ids[name].append(pid)
                break
    rng = random.Random(args.seed)
    for name in strata_ids:
        strata_ids[name].sort(key=int)
        rng.shuffle(strata_ids[name])
    pre["kandidaten_je_klasse"] = {k: len(v) for k, v in strata_ids.items()}  # type: ignore[assignment]
    log(f"  Kandidaten: {pre['kandidaten_je_klasse']}")

    quota = {s[0]: s[3] for s in STRATA}
    log("3/6 Pruefung der Kandidaten")
    results, reasons = evaluate_candidates(args.csv, strata_ids, floor_of, quota, log)

    log("4/6 Auswahl")
    selected: List[str] = []
    seen_sig: Dict[str, str] = {}
    dup_sig = 0
    per_stratum_eligible: Dict[str, List[str]] = {}
    for name, ids in strata_ids.items():
        elig = []
        for pid in ids:
            a = results.get(pid)
            if a is None or not a.ok:
                continue
            sig = a.plan["signatur"]
            if sig in seen_sig:
                dup_sig += 1
                continue
            seen_sig[sig] = pid
            elig.append(pid)
        per_stratum_eligible[name] = elig
    for name, elig in per_stratum_eligible.items():
        bins: Dict[int, List[str]] = defaultdict(list)
        for pid in elig:
            bins[room_bin(results[pid].plan["gt"]["raeume"])].append(pid)
        picked: List[str] = []
        order_bins = sorted(bins)
        while len(picked) < quota[name] and any(bins[b] for b in order_bins):
            for b in order_bins:
                if bins[b] and len(picked) < quota[name]:
                    picked.append(bins[b].pop(0))
        selected += picked
    selected.sort(key=int)
    log(f"  gewaehlt: {len(selected)} Plaene")

    log("5/6 Rendern")
    plan_files: Dict[str, List[Dict[str, Any]]] = {}
    for idx, pid in enumerate(selected):
        plan = results[pid].plan
        key = plan_key(pid)
        n_floors_same = len(by_plan[pid])
        variants = []
        for v in render_dxf.varianten_fuer_index(idx):
            data, info = render_dxf.render_dxf(plan, v, key, seed=args.seed)
            name = f"{key}__{v['id']}.dxf"
            sha = sha256_bytes(data)
            gt_json = _dump(build_gt_json(plan, v, key, name, sha, info, n_floors_same))
            variants.append({"variant": v, "name": name, "dxf": data, "sha": sha, "json": gt_json, "info": info})
        plan_files[pid] = variants

    def total_bytes() -> int:
        return sum(len(f["dxf"]) + len(f["json"]) for vs in plan_files.values() for f in vs)

    budget_log: List[str] = []
    size0 = total_bytes()
    budget_log.append(f"Start: {len(plan_files)} Plaene, {sum(len(v) for v in plan_files.values())} Dateien, {size0} Bytes (DXF + JSON)")
    while total_bytes() > args.budget_bytes and len(plan_files) > args.min_plans:
        biggest = max(plan_files, key=lambda p: sum(len(f["dxf"]) + len(f["json"]) for f in plan_files[p]))
        sz = sum(len(f["dxf"]) + len(f["json"]) for f in plan_files[biggest])
        budget_log.append(f"Plan {biggest} entfernt ({sz} Bytes, Budget {args.budget_bytes})")
        del plan_files[biggest]
    while total_bytes() > args.budget_bytes:
        # Mindestzahl erreicht: die groessten Plaene nur noch in einer Konvention
        cand = [p for p, vs in plan_files.items() if len(vs) > 1]
        if not cand:
            break
        biggest = max(cand, key=lambda p: sum(len(f["dxf"]) + len(f["json"]) for f in plan_files[p]))
        dropped = plan_files[biggest].pop()
        budget_log.append(f"Plan {biggest}: zweite Konvention ({dropped['variant']['id']}) entfernt")
    selected = [p for p in selected if p in plan_files]
    final_bytes = total_bytes()
    budget_log.append(f"Ende: {len(plan_files)} Plaene, {sum(len(v) for v in plan_files.values())} Dateien, {final_bytes} Bytes")
    log("  " + "\n  ".join(budget_log))

    log("6/6 Schreiben")
    plans_dir = args.out / "plans"
    if plans_dir.exists():
        for f in plans_dir.iterdir():
            if f.suffix in (".dxf", ".json"):
                f.unlink()
    plans_dir.mkdir(parents=True, exist_ok=True)
    manifest_plans = []
    for pid in selected:
        plan = results[pid].plan
        stratum = next(s[0] for s in STRATA if s[1] <= n_walls[pid] <= s[2])
        files = []
        for f in plan_files[pid]:
            (plans_dir / f["name"]).write_bytes(f["dxf"])
            (plans_dir / (f["name"][:-4] + ".json")).write_bytes(f["json"])
            files.append({
                "dxf": f"plans/{f['name']}", "gt_json": f"plans/{f['name'][:-4]}.json",
                "konvention": f["variant"]["konvention"], "variante": f["variant"]["id"],
                "wand_ausfuehrung": f["variant"]["wand"],
                "unterbrechung_an_oeffnungen": f["variant"]["unterbrechung"],
                "schraffur": f["variant"]["schraffur"],
                "dxf_bytes": len(f["dxf"]), "json_bytes": len(f["json"]), "dxf_sha256": f["sha"],
            })
        gt = plan["gt"]
        manifest_plans.append({
            "plan_key": plan_key(pid), "plan_id": pid, "floor_id": plan["floor_id"],
            "klasse": stratum, "wandzeilen_roh": n_walls[pid],
            "gt": {k: gt[k] for k in ("wandlaenge_achse_m", "wandlaenge_flaechenkante_m", "n_waende",
                                       "tueren", "fenster", "raeume", "raumflaeche_summe_m2")},
            "dateien": files,
        })
    sel_stats = {
        "seed": args.seed,
        "rohdaten_wiederholung": rep,
        "vorfilter": {k: v for k, v in pre.items()},
        "geprueft_je_klasse": {s: sum(1 for p in strata_ids[s] if p in results) for s in strata_ids},
        "ausschlussgruende_je_klasse": {s: dict(c) for s, c in reasons.items()},
        "geprueft_gesamt": len(results),
        "ausschlussgruende_gesamt": dict(sum((c for c in reasons.values()), Counter())),
        "diagnose_geprueft": aggregate_diag(results),
        "zulaessig_aber_gleiche_signatur": dup_sig,
        "zulaessig_je_klasse": {s: len(v) for s, v in per_stratum_eligible.items()},
        "quoten": quota,
        "budget": budget_log,
    }
    (args.out / "selection.json").write_text(json.dumps(sel_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    n_files = sum(len(p["dateien"]) for p in manifest_plans)
    manifest = {
        "schema": "planproof-goldset-manifest/1",
        "titel": "planproof_cad Goldset (DXF aus Swiss Dwellings v3.0.0)",
        "quelle": SOURCE,
        "hinweis_synthetisch": SYNTHETIC_NOTE,
        "lizenzhinweis": "Die abgeleiteten Dateien stehen unter CC BY 4.0 (Namensnennung siehe ATTRIBUTION.md).",
        "seed": args.seed,
        "plaene": len(manifest_plans),
        "dateien_dxf": n_files,
        "gesamtgroesse_bytes": sum(f["dxf_bytes"] + f["json_bytes"] for p in manifest_plans for f in p["dateien"]),
        "budget_bytes": args.budget_bytes,
        "konventionen": {k: {"name": v["name"], "layer_wand": [v["wand_aussen"], v["wand_innen"]],
                              "layer_tuer": v["tuer"], "layer_fenster": v["fenster"], "layer_raum": v["raum"]}
                         for k, v in render_dxf.KONVENTIONEN.items()},
        "varianten": render_dxf.VARIANTEN,
        "plan_liste": manifest_plans,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_attribution(args.out / "ATTRIBUTION.md")
    log(f"fertig in {time.time() - t0:.0f} s: {manifest['plaene']} Plaene, {n_files} DXF, {manifest['gesamtgroesse_bytes']} Bytes")
    return 0


def write_attribution(path: Path) -> None:
    text = f"""# Namensnennung

Die DXF-Dateien und JSON-Dateien in `plans/` sind abgeleitete Werke aus dem Datensatz

**{SOURCE['datensatz']}**

- Urheber: {SOURCE['urheber']}
- Version {SOURCE['version']}, Zenodo, DOI [{SOURCE['doi']}](https://doi.org/{SOURCE['doi']})
  (Sammel-DOI aller Versionen: 10.5281/zenodo.7070951)
- Lizenz: [{SOURCE['lizenz']}]({SOURCE['lizenz_url']})

## Aenderungen

{SOURCE['aenderungen']}.

Im Einzelnen: aus `geometries.csv` wurden je plan_id ein Geschoss gewaehlt, die Polygone (Meter) auf die
Hauptrichtung der Waende gedreht und verschoben und mit selbst gewaehlten Layernamen als DXF gezeichnet
(Einheit Millimeter). Die Konventionen sind synthetisch; sie bilden keine echten Buero-Exporte ab. Die
Ground Truth in den JSON-Dateien ist aus denselben Polygonen abgeleitet (siehe `../GROUND_TRUTH.md`).

Die Urheber des Datensatzes unterstuetzen diese abgeleiteten Dateien nicht und haben sie nicht geprueft.
Die abgeleiteten Dateien stehen ebenfalls unter CC BY 4.0. Die Rohdaten (ZIP, CSV) liegen nicht im Repository.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
