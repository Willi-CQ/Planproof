# planproof-cad

Eine CAD-Zeichnung (DXF) wird in ein BIM-Modell umgerechnet: Wände, Türen und Fenster, Räume,
IFC4, GLB und eine Mengenliste. Die Geometrie wird aus Vektoren gelesen, ohne maschinelles Lernen,
ohne Bilderkennung und ohne Cloud-Zugriff.

Stand: Skelett 0.1.0. Der Code ist aus der DXF-Pipeline von Planproof herausgelöst und rechnet
dasselbe. Getestet nur auf einem synthetischen DXF (Demo-Grundriss aus `planproof_cad.cad.synth`).
Das Goldset unter `goldset/` (40 Pläne aus Swiss Dwellings, synthetische Layer) zeigt, wo die Pipeline
steht (siehe `goldset/GROUND_TRUTH.md`). Auf echten Büro-Exporten aus Allplan, ArchiCAD oder Vectorworks ist
sie nicht gemessen.

## Installation

Python 3.11 oder neuer.

```
pip install planproof_cad-0.1.0-py3-none-any.whl
```

Pflicht-Abhängigkeiten: numpy, shapely, ezdxf, pydantic, ifcopenshell. Extras: `rest`
(fastapi, uvicorn, python-multipart), `xlsx` (openpyxl), `graph` (networkx), `dev` (pytest, ruff, httpx für den REST-Test).

Aus dem Quellbaum, mit eigenem venv:

```
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

## Demo

```
planproof-cad demo ./out
```

Erzeugt einen Demo-Grundriss, rechnet ihn durch und schreibt nach `./out`: `plan.ifc`, `model.glb`,
`scene.json`, `quantity_takeoff.csv`, `meta.json`, `result.json`.

Eine eigene Zeichnung:

```
planproof-cad convert plan.dxf ./out --unit mm --height 2.80
```

DWG wird nur über eine ODA-Brücke gelesen, die dieses Paket nicht mitbringt und die eine eigene
Lizenz braucht.

## Mengen, Kostengruppen-Vorschlag, Schnittstellen

Aus der `scene.json` entstehen Mengen je Bauteilart (Wände außen und innen mit Brutto- und Nettofläche,
Öffnungen werden von der Elternwand abgezogen, Räume als Bezugsfläche) und ein Vorschlag für
Kostengruppen. Wandhöhe und Öffnungshöhen sind Annahmen und stehen als solche im Ergebnis. Jeder
Kostengruppen-Eintrag ist ein Vorschlag zur Freigabe durch einen Menschen.

```
planproof-cad convert plan.dxf ./out --geschoss EG     # wie bisher, plus mengen.json, mengen_de.csv, kg_vorschlaege.json
planproof-cad takeoff ./out/scene.json --geschoss EG   # Mengen (--json für maschinenlesbar, -o für Dateien)
planproof-cad kg ./out/scene.json --geschoss EG        # Kostengruppen-Vorschläge zur Freigabe
planproof-cad layers plan.dxf                          # Layer, Klassifikation, Entity-Zahlen
planproof-cad serve-rest                               # REST auf 127.0.0.1:8080 (Extra "rest")
planproof-cad serve-mcp --out-dir ./out                # MCP-Server über stdio
```

Aufbau der Dateien, REST-Beispiele mit curl, MCP-Konfiguration, das Mengenmodell mit allen Annahmen und
was nicht abgedeckt ist: `docs/INTEGRATION.md` (Englisch).

## Katalog-Zuweisung

Jedes Bauteil wird einem Aufbau aus den mitgelieferten Katalog-Seeds zugeordnet
(`planproof_cad/catalog/data`, geladen über `importlib.resources`). Ob das geklappt hat, steht in
`meta.json` und im Ergebnisobjekt unter `template_assignment`:

```
{"ok": true, "error": null, "templates_loaded": 119, "seed_files_skipped": 0}
```

## Tests

```
pytest
```

Der Test `test_importverbot` führt den Demo-Lauf mit gesperrten Importen aus (torch, ultralytics,
cv2, scipy, PIL und weitere). `test_wheel_inhalt` baut das Wheel und installiert es in ein frisches
venv. Der Drift-Test gegen die Quell-Pipeline liegt nicht im Paket, sondern im Planproof-Repo unter
`scripts/ops/plan_paket/`, weil er `planproof_geometry` importieren muss.
