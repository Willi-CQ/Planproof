# Integration guide: planproof-cad

This guide is for a team that wants to call the package from its own system: as a Python library, from
the command line, over HTTP (REST) or from an MCP client. It also describes the output files, the
quantity model with every assumption it makes, and what the package does not cover.

All human-readable strings in the results (assumptions, rationales, hints) are German, because the
cost groups and the plans are German. Keys, codes and file names are ASCII.

## 1. What the package does

Input is a DXF drawing (a DWG only if an ODA File Converter or LibreDWG bridge is installed, see
`ODA_AND_DWG.md`). The package reads the vector geometry, builds a wall graph, attaches door and window
blocks to wall edges, detects rooms, and writes an IFC4 model, a GLB view, a `scene.json` and quantity
files. On top of the scene it computes quantities per component type and proposes cost group numbers.
There is no machine learning, no image recognition and no cloud call.

How well the geometry is recovered depends on the drawing. The measured state on a synthetic goldset is
in `goldset/GROUND_TRUTH.md`. On real office exports (Allplan, ArchiCAD, Vectorworks) it has not been
measured. Treat every quantity as a draft that a person checks.

Three layers, one source:

| Layer | Module | Purpose |
|---|---|---|
| Services | `planproof_cad.services` (public facade `planproof_cad.api`) | The four functions. CLI, REST and MCP only call these. |
| Mengenkern and rules | `planproof_cad.kg.zuordnung` | Quantities and cost group proposals on a plain scene dict. Standard library only. |
| Pipeline | `planproof_cad.cad` | DXF to scene, IFC, GLB. Imported only when a DXF is converted. |

`takeoff` and `kg` on an existing `scene.json` do not import ezdxf, ifcopenshell, shapely or numpy.

## 2. Installation

Python 3.11 or newer.

```
pip install planproof_cad-0.1.0-py3-none-any.whl            # library, CLI, MCP
pip install "planproof_cad-0.1.0-py3-none-any.whl[rest]"    # adds fastapi, uvicorn, python-multipart
```

Required dependencies: numpy, shapely, ezdxf, pydantic, ifcopenshell. Extras: `rest`, `xlsx`
(openpyxl), `graph` (networkx), `dev` (pytest, ruff, networkx, httpx for the REST test client).

From the source tree:

```
cd packages/planproof_cad
uv sync --locked --extra rest --extra dev
.venv/Scripts/python.exe -m pytest        # Windows; .venv/bin/python on Linux and macOS
```

## 3. Python API

```python
from planproof_cad import api

res = api.run_dxf("plan.dxf", "out", {"wall_height_m": 2.8, "dxf_unit": "mm"}, geschoss="EG")
mengen = api.takeoff("out/scene.json", geschoss="EG", wall_height_m=None)   # dict, see section 6.2
kg = api.assign_kg(mengen)                                                   # dict, see section 7
layers = api.list_layers("plan.dxf")                                         # dict, see section 8
```

| Function | Input | Returns |
|---|---|---|
| `run_dxf(path, out_dir, config=None, *, geschoss=None, mengen=True)` | `.dxf` or `.dwg`, output directory, pipeline config | Dict with `ok`, `error`, `metrics`, `warnings`, `artifacts`, `mengen`, `kg_vorschlaege`. Writes the files of section 5. |
| `takeoff(quelle, geschoss=None, wall_height_m=None, *, out_dir=None, work_dir=None)` | scene dict, `SceneModel`, path to `scene.json`, or path to a `.dxf` (then converted into `work_dir` or `out_dir`) | `mengen` dict. With `out_dir` it also writes `mengen.json` and `mengen_de.csv`. |
| `assign_kg(mengen, *, geschoss=None, wall_height_m=None, out_dir=None, work_dir=None)` | a `mengen` dict, a list of them (several floors, with sums), a path to `mengen.json`, or anything `takeoff` accepts | Cost group proposals. |
| `list_layers(path)` | `.dxf` (or `.dwg` with bridge) | Layers with classification and entity counts. |

`api.combine_kg(list_of_proposal_documents)` merges proposal documents of several floors.

Errors the caller can fix raise `api.ServiceError` with a stable `code` (`input_not_found`,
`unsupported_input`, `scene_unreadable`, `not_a_scene`, `not_mengen`, `invalid_wall_height`,
`path_outside_root`, ...). The pipeline itself reports a failed conversion as `ok: false` in the result.

`planproof_cad.kg.zuordnung` can also be used on its own: `mengen_aus_scene(scene_dict, geschoss=...)`
and `vorschlaege_aus_scene(scene_dict, geschoss=...)`. The file has no import from this package, so it
can be copied byte for byte into a project that has a `scene_v1` dict but not this package. A test runs a
copy of the file in an interpreter without the package and compares the result.

## 4. Command line

```
planproof-cad demo ./out                                   # synthetic demo plan, full run
planproof-cad convert plan.dxf ./out --unit mm --height 2.80 --geschoss EG
planproof-cad takeoff ./out/scene.json --geschoss EG [-o ./out] [--height 3.0]
planproof-cad kg ./out/scene.json --geschoss EG [-o ./out]
planproof-cad layers plan.dxf
planproof-cad serve-rest [--host 127.0.0.1] [--port 8080] [--workdir DIR] [--max-upload-mb 50]
planproof-cad serve-mcp --out-dir ./out [--read-root DIR ...]
```

Every command takes `--json` where it produces a result: stdout then carries exactly one JSON document
(ASCII-escaped), progress and log lines go to stderr. Exit codes: 0 success, 1 conversion failed,
2 bad input or usage.

`takeoff` and `kg` accept several inputs, one `--geschoss` per input; `kg` then adds sums over all floors.
`takeoff` and `kg` also accept a `.dxf`; the conversion result then goes to `-o`.

Real output of `planproof-cad kg out/scene.json --geschoss EG` on the demo plan:

```
Alle Einträge sind Vorschläge zur Freigabe. Ein Mensch prüft und gibt frei; dieses Modul legt nichts an.
  Außenwände EG 94,44 m2 netto, KG 330, Vorschlag zur Freigabe  [Konfidenz hoch]
  Außenfenster EG 3 Stk, KG 334, Vorschlag zur Freigabe  [Konfidenz hoch]
  Außentüren EG 1 Stk, KG 334, Vorschlag zur Freigabe  [Konfidenz hoch]
  Innenwände EG 35,22 m2 netto, KG 340, Vorschlag zur Freigabe  [Konfidenz mittel]
  Innentüren EG 2 Stk, KG 344, Vorschlag zur Freigabe  [Konfidenz mittel]
  Bezugsfläche Räume: 73,13 m2, keine Kostengruppe
Nicht abgedeckt: Stützen, Treppen, Decken, Dächer
```

## 5. Output files

`planproof-cad convert` and `api.run_dxf` write into the output directory:

| File | Content |
|---|---|
| `plan.ifc` | IFC4 model: walls, openings, doors, windows, spaces. |
| `model.glb` | GLB view of the same scene. |
| `scene.json` | The component model (`scene_v1`), the input of everything below. |
| `quantity_takeoff.csv` | One English row per component (`id,kind,wall_type,length_m,...`). Unchanged since 0.1.0. |
| `meta.json`, `result.json` | Pipeline metrics, warnings, template assignment status. |
| `mengen.json` | Quantities per component type, with assumptions (section 6). |
| `mengen_de.csv` | The same quantities as German CSV that the Planproof desktop application reads (section 6.4). |
| `kg_vorschlaege.json` | Cost group proposals (section 7). |

`mengen.json`, `mengen_de.csv` and `kg_vorschlaege.json` get a floor suffix (`mengen_EG.json`) when
`planproof-cad takeoff` or `kg` is called with several floors.

## 6. Quantities

### 6.1 Model

The scene stores walls as **edges of a wall graph**. A physical wall can be split into several edges at
junctions, so a row is not a wall. The quantities add up edges and never claim that a row is a wall.

Per wall edge:

- length = `attrs.length_m` (edge length on the wall axis)
- height = the `height_m` of the scene, or the parameter `wall_height_m` if given, or 2.80 m if the scene
  has none
- gross area = length times height
- opening deduction = sum over the linked openings (below)
- net area = gross area minus deduction, never below zero

Per opening (door, window): it hangs on a wall edge through `attrs.parent_edge_id`. Its area is
width times effective height. The effective height runs from the sill (`sill_height_m`, 0 for doors) to the
head (`head_height_m`), clipped to the wall height. The area is subtracted from the parent edge. An opening
without a parent edge (missing or unknown id) is **not** subtracted, is counted in the group
"location unknown" and reported as hint `OEFFNUNG_OHNE_WAND`. The location of an opening (exterior or
interior) is the location of its parent wall.

Rooms: count and area. The area is `attrs.area_net_m2` if present, otherwise `attrs.area_m2`. If any room has
only `area_m2`, `flaeche_quelle` is `gemischt` or `area_m2` and a hint says that the sum is not a net room
area. Room areas are reference areas. They get no cost group; they belong to DIN 277.

The floor is a free text parameter (`geschoss`, for example `EG`). A DXF has no floors, the parameter is
passed through to every output.

### 6.2 `mengen.json`

`schema` is `planproof_cad.mengen`, `schema_version` is `mengen_v1`. Top-level keys:

| Key | Content |
|---|---|
| `geschoss` | The floor parameter or `null`. |
| `annahmen` | List of `{id, art, wert, einheit, text}`. Every assumption (section 6.3). |
| `waende.gruppen` | One group per location (`aussen`, `innen`, `unbekannt`) and bearing state (`tragend` true, false or null): `anzahl_kanten`, `laenge_m`, `flaeche_brutto_m2`, `oeffnungsabzug_m2`, `flaeche_netto_m2`, `hoehen_m`, `belege`, `source_ids`. |
| `waende.je_lage`, `waende.summe` | Sums per location and in total. |
| `oeffnungen.gruppen` | One group per kind (`tuer`, `fenster`) and location: `anzahl`, `breite_summe_m`, `hoehen_m`, `flaeche_m2`, `abgezogen_m2`, `source_ids`. |
| `oeffnungen.je_art`, `oeffnungen.ohne_wand` | Sums per kind; number of openings without a parent wall. |
| `raeume` | `anzahl`, `flaeche_m2`, `flaeche_quelle`, `kostengruppe: null`, `bezug`, `hinweis`. |
| `hinweise` | List of `{code, text, ids}`. |
| `nicht_abgedeckt` | Components the pipeline does not process (columns, stairs, slabs, roofs). |
| `nicht_verarbeitet` | Component kinds found in the scene that this module does not quantify. |
| `elemente` | One row per wall edge, opening and room (basis of the CSV). |

Bearing (`tragend`) is reported only if the wall component says so (`attrs.bearing`, also read from
`attrs.meta.bearing` or `component.meta.bearing`: boolean or one of `tragend`, `nichttragend`,
`load_bearing`, `non_bearing`). Otherwise the bearing state is `null` and the group is not split.

### 6.3 Assumptions (never reported as measurements)

| Id | Statement |
|---|---|
| `wandhoehe` | The wall height is a configuration assumption (default 2.80 m), or a parameter. A DXF contains no storey height. |
| `oeffnungshoehen` | Door height, window sill and window head come from the scene (defaults of the pipeline configuration while the block carries no height). They are not measured. The width comes from the block or is a default width. |
| `netto_methode` | Wall area is edge length on the wall axis times height. Overlaps at corners and junctions, reveals and overmeasure rules are not considered. Net is gross minus linked openings. |
| `wandkanten` | Rows are graph edges. Sums add edges. |
| `lage_aussen_innen` | Exterior or interior is taken from `wall_type`. The pipeline sets "exterior" only from a layer name; "interior" is also the default. Without a marker, "interior" is therefore not evidence. |
| `geschoss` | The floor is a parameter or open. |

The fields `wall_type_source` (`layer`, `geometry`, `default`), `bearing_source` and `layer` on a wall
component are read when present and raise or lower the confidence (section 7.3).

### 6.4 `mengen_de.csv`

German CSV in the dialect the desktop application parses (`quantityCsv.ts`): UTF-8 with BOM, separator
`;`, decimal point, header

```
Kategorie;ID;Typ;Laenge_m;Hoehe_m;Dicke_m;Flaeche_m2;Volumen_m3;Material;Eigenschaft
```

One row per wall edge (`Wand`, type `exterior` or `interior`, empty if unknown), door (`Tuer`), window
(`Fenster`) and room (`Raum`). For walls `Flaeche_m2` is the **net** area and `Volumen_m3` is net area times
thickness. For doors and windows `Flaeche_m2` is the opening area, for rooms the reference area.
`Eigenschaft` is free text without a semicolon: floor, `tragend=true|false|unbekannt`, gross and deduction,
sill, parent wall. After the first blank line come `ZUSAMMENFASSUNG`, one `ANNAHME` line per assumption and
`HINWEIS` lines; the desktop parser stops at the blank line. The categories stay ASCII (`Tuer`), as in the
desktop pipeline; the parser accepts both spellings.

A test runs the real `parseQuantityCsv` from `clients/desktop-tauri` with Node against the file when Node and
the desktop directory are present, and always runs a Python re-implementation of the same algorithm.

## 7. Cost group proposals

### 7.1 Rules

The rule table is `src/planproof_cad/kg/kg_rules.json` (package data, also embedded as a constant in
`kg/zuordnung.py`; a test keeps both equal). It holds numbers and neutral component names, no norm text.

| Component | Location | Bearing | KG | `element_class` |
|---|---|---|---|---|
| wall | exterior | unknown | 330 | `Aussenwand` |
| wall | exterior | bearing | 331 | `Aussenwand tragend` |
| wall | exterior | non-bearing | 332 | `Aussenwand nichttragend` |
| wall | interior | unknown | 340 | `Innenwand` |
| wall | interior | bearing | 341 | `Innenwand tragend` |
| wall | interior | non-bearing | 342 | `Innenwand nichttragend` |
| door | exterior | | 334 | `Aussentuer` |
| window | exterior | | 334 | `Aussenfenster` |
| door | interior | | 344 | `Innentuer` |
| window | interior | | 344 | `Innenfenster` |

Rooms get no cost group. Columns, stairs, slabs and roofs are not processed by the pipeline; they are listed in
`not_covered` and never produce a proposal. Walls and openings whose location is unknown get no number and
appear under `unassigned` with the reason.

### 7.2 `kg_vorschlaege.json`

`schema` is `planproof_cad.kg_vorschlaege`. `proposals` holds one entry per rule that has quantities:

```json
{
  "id": "kgv_EG_wand_innen",
  "kg": "340",
  "element_class": "Innenwand",
  "geschoss": "EG",
  "basis": "Nettoflaeche",
  "quantity": 35.22,
  "unit": "m2",
  "confidence": "mittel",
  "rationale": "Innenwand, Tragverhalten nicht bekannt: ... Außen oder innen ist nicht durch einen Layernamen belegt.",
  "assumptions": ["Wandhöhe 2,8 m, ...", "..."],
  "source_ids": ["e_2_...", "..."],
  "summary": "Innenwände EG 35,22 m2 netto, KG 340, Vorschlag zur Freigabe",
  "status": "vorschlag"
}
```

`basis` is `Nettoflaeche` (unit `m2`) for walls and `Stueck` (unit `Stk`) for doors and windows.
`status` is always `vorschlag`: a person approves, the package never creates anything in a cost system.
With several floors the document also has `summen` (sum per number and component class, weakest confidence).
`reference_quantities` carries the room area with `kg: null`.

### 7.3 Confidence

`hoch` only with layer evidence, otherwise `mittel` or `niedrig`. A proposal takes the weakest evidence of
its source components.

| Evidence | Level |
|---|---|
| Location from a layer name (`wall_type_source: layer`, or "exterior" without marker, which the pipeline sets only from a layer) | hoch |
| Opening kind from a block name (`source: dwg_block`) | hoch |
| Bearing present together with a `layer` entry, or `bearing_source: layer` | hoch |
| Location from geometry; "interior" without marker; bearing without layer entry | mittel |
| Location is a default (`wall_type_source: default`) | niedrig |

## 8. Layers

`list_layers` shows how a DXF is read: for every layer the class (`wall_exterior`, `wall_interior`, `wall`,
`door`, `window`, `room`, `annotation`, `unknown`, ...), the classification signal, the number of entities
(lines, block inserts, texts) and `read_as` (`wand`, `oeffnung_block`, `nicht_gelesen`). The summary names
the wall layers, the number of wall lines, the number of door and window blocks, and
`fallback_all_lines_as_walls` (true if the file has no wall layer at all, in which case the pipeline treats every
line as a wall candidate). Hints explain what to rename.

## 9. REST

```
planproof-cad serve-rest --workdir ./work --port 8080          # binds 127.0.0.1
```

| Method and path | Input | Output |
|---|---|---|
| `GET /healthz` | | `{"status": "ok", "version": "..."}` |
| `POST /v1/convert` | multipart: `file` or `path`; optional `unit`, `wall_height_m`, `geschoss` | `ok`, `run_id`, `files` (name and URL), `metrics`, `warnings`, `mengen` (without `elemente`), `kg_vorschlaege`. HTTP 422 with the same body if the conversion failed. |
| `GET /v1/runs/{run_id}/files/{name}` | | A result file of a conversion. |
| `POST /v1/takeoff` | JSON: `scene` or `path`; optional `geschoss`, `wall_height_m` | `mengen` |
| `POST /v1/kg-proposals` | JSON: one of `mengen`, `scene`, `path`; optional `geschoss`, `wall_height_m` | cost group proposals |
| `POST /v1/layers` | multipart: `file` or `path` | layer listing |

```
curl -s -F "file=@plan.dxf" -F geschoss=EG http://127.0.0.1:8080/v1/convert
curl -s -X POST http://127.0.0.1:8080/v1/kg-proposals -H "Content-Type: application/json" \
     -d '{"path": "runs/<run_id>/scene.json", "geschoss": "EG"}'
curl -s -X POST http://127.0.0.1:8080/v1/takeoff -H "Content-Type: application/json" \
     -d '{"scene": {...}, "geschoss": "EG", "wall_height_m": 3.0}'
curl -s -O http://127.0.0.1:8080/v1/runs/<run_id>/files/mengen_de.csv
curl -s -F "file=@plan.dxf" http://127.0.0.1:8080/v1/layers
```

Errors have the form `{"error": {"code": "...", "message": "..."}}` with these statuses: 400 invalid path or
arguments, 403 `path_outside_root`, 404 `input_not_found`, 411 `length_required`, 413 `payload_too_large`,
415 `unsupported_input`, 422 unreadable or wrong content.

Operating rules:

- No authentication, no database, no cloud calls. The default bind is `127.0.0.1`. Put an authenticating
  reverse proxy in front before exposing it.
- `path` parameters are resolved inside the work directory only. `..` segments, absolute paths elsewhere,
  neighbour directories with the same name prefix and symlinks or junctions that lead out are rejected with
  403. Uploads are stored in `workdir/uploads`, results in `workdir/runs`.
- Uploads are limited by `--max-upload-mb` (default 50). The limit is checked on `Content-Length` before the body
  is parsed and again while the file is stored. A request without `Content-Length` gets 411.
- Conversions run one at a time (a lock inside the app).
- Responses do not contain absolute server paths.

## 10. MCP (stdio)

The server speaks JSON-RPC 2.0 over stdio, one message per line, protocol version 2025-06-18, without an SDK.
It supports `initialize`, `notifications/initialized`, `ping`, `tools/list` and `tools/call`.

Configuration for a stdio client (the file name and location depend on the client):

```json
{
  "mcpServers": {
    "planproof-cad": {
      "command": "planproof-cad",
      "args": ["serve-mcp", "--out-dir", "C:/work/planproof-cad-out", "--read-root", "C:/plans"]
    }
  }
}
```

If `planproof-cad` is not on the PATH of the client, use the interpreter of the environment:
`"command": "C:/venvs/planproof-cad/Scripts/python.exe", "args": ["-m", "planproof_cad.mcp_stdio", "--out-dir", "..."]`.

Tools (each with a JSON schema, `additionalProperties: false`):

| Tool | Arguments | Effect |
|---|---|---|
| `dwg_convert` | `path`, `geschoss`, `unit`, `wall_height_m` | Converts a DXF into `out-dir/<name>/`, returns metrics, quantities, proposals. |
| `quantity_takeoff` | `path` (scene.json or DXF), `geschoss`, `wall_height_m`, `include_elements` | Returns quantities; writes `mengen*.json` and `mengen_de*.csv`. |
| `assign_kg_proposal` | `path` or `mengen`, `geschoss`, `wall_height_m` | Returns proposals; writes `kg_vorschlaege*.json`. |
| `list_layers` | `path` | Returns the layer listing. Writes nothing. |

A tool result carries the JSON both as text and as `structuredContent`. A failing tool returns `isError: true`
with `{error: {code, message}}`, so the model can read it and correct the call. Unknown tool names and
malformed messages are JSON-RPC errors (`-32602`, `-32601`, `-32700`, `-32600`).

File access: the server writes only below `--out-dir` and chooses the sub folder itself (the file name stem,
sanitized), so a caller cannot name a target path. It reads inputs only from the current directory at start,
`--out-dir` and every `--read-root`. Relative paths are searched in that order.

## 11. What is not covered

- **Components:** columns, beams, stairs, railings, slabs, roofs and furniture. The pipeline does not build
  them; there are no quantities and no proposals for them. They appear under `nicht_abgedeckt`.
- **Cost groups:** only the numbers in section 7.1. No other numbers, no norm texts, no demarcation rules.
  Rooms and reference areas get no number.
- **Heights:** wall height and opening heights are assumptions (section 6.3). Without a storey height in the
  drawing the net areas are only as good as that assumption.
- **Net quantities:** areas follow the wall axis. Corner overlaps, junctions, reveals and overmeasure rules are not
  modelled. Openings that the pipeline could not attach to a wall edge are missing from the scene; the number is in
  `meta.json` (`metrics.openings.unmatched`) and appears as hint `PIPELINE_OEFFNUNG_NICHT_ANGEHAENGT` when
  `meta.json` lies next to the `scene.json`.
- **Floors:** one drawing is one floor. Several floors are several runs with one `geschoss` parameter each.
- **DWG:** not read without an external bridge (`ODA_AND_DWG.md`).
- **Real office exports:** not measured. The goldset uses synthetic layer names.
- **Security of the REST service:** no authentication and no TLS; see section 9.
