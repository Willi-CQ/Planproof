"""Quantity takeoff from SceneModel — CSV/XLSX export.

Generates structured quantity summaries for walls, doors, windows, rooms,
and slabs from the SceneModel built by the plan-to-BIM pipeline.
"""
from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_quantity_takeoff(scene) -> Dict[str, Any]:
    """Compute quantity summary from a SceneModel.

    Returns a dict with per-category breakdowns and totals.
    """
    walls: List[Dict] = []
    doors: List[Dict] = []
    windows: List[Dict] = []
    columns: List[Dict] = []
    stairs_list: List[Dict] = []
    rooms: List[Dict] = []

    for comp in scene.components.values():
        attrs = comp.attrs
        dims = comp.dims  # (length, height, thickness)

        if comp.kind == "WALL":
            length = dims[0] if dims else 0
            height = dims[1] if dims else 2.80
            thickness = float(attrs.get("thickness_m", dims[2] if dims else 0.12))
            walls.append({
                "id": comp.id,
                "wall_type": attrs.get("wall_type", "unknown"),
                "is_loadbearing": attrs.get("is_loadbearing", None),
                "length_m": round(length, 2),
                "height_m": round(height, 2),
                "thickness_m": round(thickness, 3),
                "area_m2": round(length * height, 2),
                "volume_m3": round(length * height * thickness, 3),
                "material": attrs.get("material_hint", ""),
            })

        elif comp.kind == "DOOR":
            width = float(attrs.get("width_m", dims[0] if dims else 0.90))
            height = float(attrs.get("height_m", dims[1] if dims else 2.10))
            doors.append({
                "id": comp.id,
                "width_m": round(width, 2),
                "height_m": round(height, 2),
                "area_m2": round(width * height, 2),
                "swing_direction": attrs.get("swing_direction", ""),
                "frame_material": attrs.get("frame_material", ""),
            })

        elif comp.kind == "WINDOW":
            width = float(attrs.get("width_m", dims[0] if dims else 1.20))
            height = float(attrs.get("height_m", dims[1] if dims else 1.20))
            windows.append({
                "id": comp.id,
                "width_m": round(width, 2),
                "height_m": round(height, 2),
                "area_m2": round(width * height, 2),
                "sill_height_m": round(float(attrs.get("sill_height_m", 0.90)), 2),
            })

        elif comp.kind == "COLUMN":
            columns.append({
                "id": comp.id,
                "width_m": round(dims[0], 2) if dims else 0.30,
                "depth_m": round(dims[2], 2) if dims else 0.30,
                "height_m": round(dims[1], 2) if dims else 2.80,
                "material": attrs.get("material_hint", "concrete"),
            })

        elif comp.kind == "STAIRS":
            stairs_list.append({
                "id": comp.id,
                "width_m": round(dims[0], 2) if dims else 1.00,
                "height_m": round(dims[1], 2) if dims else 2.80,
            })

        elif comp.kind == "ROOM":
            rooms.append({
                "id": comp.id,
                "label": attrs.get("label", comp.name),
                "room_type": attrs.get("room_type", ""),
                "area_m2": round(float(attrs["area_m2"]), 1) if attrs.get("area_m2") else None,
            })

    # Totals
    total_wall_length = sum(w["length_m"] for w in walls)
    total_wall_area = sum(w["area_m2"] for w in walls)
    total_wall_volume = sum(w["volume_m3"] for w in walls)
    total_door_area = sum(d["area_m2"] for d in doors)
    total_window_area = sum(w["area_m2"] for w in windows)
    total_floor_area = sum(r["area_m2"] for r in rooms if r["area_m2"])

    # Exterior vs interior walls
    exterior_walls = [w for w in walls if w["wall_type"] == "exterior"]
    interior_walls = [w for w in walls if w["wall_type"] != "exterior"]

    return {
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "columns": columns,
        "stairs": stairs_list,
        "rooms": rooms,
        "summary": {
            "walls_count": len(walls),
            "walls_exterior_count": len(exterior_walls),
            "walls_interior_count": len(interior_walls),
            "walls_total_length_m": round(total_wall_length, 1),
            "walls_total_area_m2": round(total_wall_area, 1),
            "walls_total_volume_m3": round(total_wall_volume, 2),
            "doors_count": len(doors),
            "doors_total_area_m2": round(total_door_area, 1),
            "windows_count": len(windows),
            "windows_total_area_m2": round(total_window_area, 1),
            "columns_count": len(columns),
            "stairs_count": len(stairs_list),
            "rooms_count": len(rooms),
            "total_floor_area_m2": round(total_floor_area, 1),
        },
    }


def export_csv(takeoff: Dict[str, Any], out_path: str) -> None:
    """Export quantity takeoff as CSV."""
    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")

    # Header
    writer.writerow(["Kategorie", "ID", "Typ", "Laenge_m", "Hoehe_m", "Dicke_m",
                      "Flaeche_m2", "Volumen_m3", "Material", "Eigenschaft"])

    for w in takeoff["walls"]:
        writer.writerow([
            "Wand", w["id"], w["wall_type"], w["length_m"], w["height_m"],
            w["thickness_m"], w["area_m2"], w["volume_m3"], w["material"],
            f"tragend={w['is_loadbearing']}" if w["is_loadbearing"] is not None else "",
        ])

    for d in takeoff["doors"]:
        writer.writerow([
            "Tuer", d["id"], d.get("swing_direction", ""), d["width_m"], d["height_m"],
            "", d["area_m2"], "", d.get("frame_material", ""),
            f"Schlagrichtung: {d.get('swing_direction', '')}",
        ])

    for w in takeoff["windows"]:
        writer.writerow([
            "Fenster", w["id"], "", w["width_m"], w["height_m"],
            "", w["area_m2"], "", "",
            f"Bruestung: {w['sill_height_m']}m",
        ])

    for c in takeoff["columns"]:
        writer.writerow([
            "Stuetze", c["id"], c["material"], c["width_m"], c["height_m"],
            c["depth_m"], "", "", c["material"], "",
        ])

    for r in takeoff["rooms"]:
        writer.writerow([
            "Raum", r["id"], r.get("room_type", ""), "", "", "",
            r["area_m2"] or "", "", "", r["label"],
        ])

    # Summary
    s = takeoff["summary"]
    writer.writerow([])
    writer.writerow(["ZUSAMMENFASSUNG"])
    writer.writerow(["Waende gesamt", s["walls_count"], "Laenge", f"{s['walls_total_length_m']}m",
                      "Flaeche", f"{s['walls_total_area_m2']}m2", "Volumen", f"{s['walls_total_volume_m3']}m3"])
    writer.writerow(["Tueren gesamt", s["doors_count"], "Flaeche", f"{s['doors_total_area_m2']}m2"])
    writer.writerow(["Fenster gesamt", s["windows_count"], "Flaeche", f"{s['windows_total_area_m2']}m2"])
    writer.writerow(["Raeume gesamt", s["rooms_count"], "Grundflaeche", f"{s['total_floor_area_m2']}m2"])

    Path(out_path).write_text(output.getvalue(), encoding="utf-8-sig")


def export_xlsx(takeoff: Dict[str, Any], out_path: str) -> None:
    """Export quantity takeoff as XLSX with formatted sheets."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        logger.warning("openpyxl not installed, skipping XLSX export")
        return

    wb = openpyxl.Workbook()
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    def _write_sheet(ws, headers, rows):
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = header_font
            cell.fill = header_fill
        for row_idx, row in enumerate(rows, 2):
            for col_idx, val in enumerate(row, 1):
                ws.cell(row=row_idx, column=col_idx, value=val)

    # Walls sheet
    ws_walls = wb.active
    ws_walls.title = "Waende"
    _write_sheet(ws_walls,
        ["ID", "Typ", "Tragend", "Laenge (m)", "Hoehe (m)", "Dicke (m)",
         "Flaeche (m2)", "Volumen (m3)", "Material"],
        [[w["id"], w["wall_type"], w.get("is_loadbearing"), w["length_m"],
          w["height_m"], w["thickness_m"], w["area_m2"], w["volume_m3"], w["material"]]
         for w in takeoff["walls"]])

    # Doors sheet
    ws_doors = wb.create_sheet("Tueren")
    _write_sheet(ws_doors,
        ["ID", "Breite (m)", "Hoehe (m)", "Flaeche (m2)", "Schlagrichtung", "Rahmen"],
        [[d["id"], d["width_m"], d["height_m"], d["area_m2"],
          d.get("swing_direction", ""), d.get("frame_material", "")]
         for d in takeoff["doors"]])

    # Windows sheet
    ws_win = wb.create_sheet("Fenster")
    _write_sheet(ws_win,
        ["ID", "Breite (m)", "Hoehe (m)", "Flaeche (m2)", "Bruestung (m)"],
        [[w["id"], w["width_m"], w["height_m"], w["area_m2"], w["sill_height_m"]]
         for w in takeoff["windows"]])

    # Rooms sheet
    ws_rooms = wb.create_sheet("Raeume")
    _write_sheet(ws_rooms,
        ["ID", "Bezeichnung", "Raumtyp", "Flaeche (m2)"],
        [[r["id"], r["label"], r.get("room_type", ""), r["area_m2"] or ""]
         for r in takeoff["rooms"]])

    # Summary sheet
    ws_sum = wb.create_sheet("Zusammenfassung")
    s = takeoff["summary"]
    summary_rows = [
        ["Waende", s["walls_count"], f"{s['walls_total_length_m']} m", f"{s['walls_total_area_m2']} m2"],
        ["  davon Aussen", s["walls_exterior_count"]],
        ["  davon Innen", s["walls_interior_count"]],
        ["Tueren", s["doors_count"], "", f"{s['doors_total_area_m2']} m2"],
        ["Fenster", s["windows_count"], "", f"{s['windows_total_area_m2']} m2"],
        ["Stuetzen", s["columns_count"]],
        ["Treppen", s["stairs_count"]],
        ["Raeume", s["rooms_count"], "", f"{s['total_floor_area_m2']} m2"],
    ]
    _write_sheet(ws_sum,
        ["Kategorie", "Anzahl", "Gesamt-Laenge", "Gesamt-Flaeche"],
        summary_rows)

    wb.save(out_path)


class QuantityTakeoffStep:
    """Pipeline step: compute and export quantity takeoff from SceneModel."""
    name = "quantity_takeoff"
    soft_fail = True

    def run(self, ctx, state):
        if not state.scene:
            ctx.log("No SceneModel for quantity takeoff, skipping")
            return state

        takeoff = compute_quantity_takeoff(state.scene)
        state.metrics["quantity_summary"] = takeoff["summary"]

        output_dir = Path(state.output_dir)

        # CSV export
        csv_path = output_dir / "quantity_takeoff.csv"
        export_csv(takeoff, str(csv_path))
        ctx.log(f"Quantity takeoff CSV: {csv_path}")

        # XLSX export (optional, depends on openpyxl)
        xlsx_path = output_dir / "quantity_takeoff.xlsx"
        export_xlsx(takeoff, str(xlsx_path))
        if xlsx_path.exists():
            ctx.log(f"Quantity takeoff XLSX: {xlsx_path}")

        # Store takeoff data for CreatePackageStep
        state.config["_quantity_takeoff"] = takeoff

        s = takeoff["summary"]
        ctx.log(
            f"Quantity: {s['walls_count']} walls ({s['walls_total_length_m']}m), "
            f"{s['doors_count']} doors, {s['windows_count']} windows, "
            f"{s['rooms_count']} rooms ({s['total_floor_area_m2']}m2)"
        )
        return state
