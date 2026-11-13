# Goldset: Ground Truth

Stand 20.10.2026. Dieses Dokument beschreibt, wie aus Swiss Dwellings v3.0.0 ein DXF-Goldset mit
automatisch abgeleiteter Wahrheit entsteht, welche Pläne aus welchem Grund fehlen und was an der
Wahrheit nicht eindeutig ist. Alle Zahlen stammen aus `data/raw_stats.json`, `data/selection.json`,
`data/manifest.json` oder aus dem Basislauf in Abschnitt 9; sie sind mit den Skripten in diesem
Ordner nachrechenbar (Abschnitt 10).

## 0. Was das Goldset beweist und was nicht

Die Zeichenkonventionen sind synthetisch gewählt. Die Layernamen (AIA, DACH-Büropraxis,
Sachsen-SIB-Kostengruppen, ArchiCAD-artige Stiftlayer) stammen aus öffentlichen Listen und aus
der Praxis, die Zeichnungen selbst aus Polygonen eines Wohnungsdatensatzes. Das Goldset beweist die
Pipeline auf Swiss-Dwellings-Geometrie mit selbst gewählten Layern. Es beweist sie nicht auf echten
Büro-Exporten aus Allplan, ArchiCAD oder Vectorworks. Dafür fehlen echte Exporte; sie sind eine eigene
Aufgabe. Der Hinweis steht deshalb im Manifest, in jeder Schwester-JSON und im Bericht von
`run_goldset.py`.

Die Pläne stammen aus dem Schweizer Datensatz der Archilyse AG (digitalisiert, manuell geprüft laut
Zenodo-Beschreibung). Die Nutzung ist überwiegend Wohnen: 81,4 % der Zeilen sind `RESIDENTIAL`, 18,1 %
`PUBLIC` (zum Beispiel Treppenhäuser), 0,5 % `COMMERCIAL`. Büro-, Gewerbe- und Industriebau sowie
Bestandspläne mit Zeichenfehlern kommen kaum vor.

## 1. Quelle und Lizenz

- Datensatz: Swiss Dwellings v3.0.0, Zenodo, DOI 10.5281/zenodo.7788422, Urheber Archilyse AG
  (Standfest et al.), Lizenz CC BY 4.0.
- Datei `swiss-dwellings-v3.0.0.zip`, 931.868.205 Bytes, md5 `3b7915ecd5bf8e492a3e78c598b74123`.
  Die md5 aus der Zenodo-API stimmt mit dem Download vom 20.10.2026 überein (geprüft).
- Die ZIP und alle Inhalte wurden nur als Daten gelesen (`zipfile`, `csv`), nichts wurde ausgeführt.
  Die Rohdaten liegen außerhalb des Repos (`C:\tmp\swiss-dwellings`).
- Die abgeleiteten Dateien unter `data/` stehen unter CC BY 4.0. Namensnennung und Änderungshinweis
  stehen in `data/ATTRIBUTION.md`.

## 2. Rohdaten (gezählt)

Gelesen wurde `geometries.csv` (1.093.732.415 Bytes, 3.255.905 Zeilen, Streaming ohne Komplettladen).
Spalten: `apartment_id, site_id, building_id, plan_id, floor_id, unit_id, area_id, unit_usage,
entity_type, entity_subtype, geometry, elevation, height`.

| Größe | Wert |
|---|---|
| Standorte (`site_id`) | 1.466 |
| Gebäude | 3.184 |
| Pläne (`plan_id`) | 8.941 |
| Geschosse (`floor_id`) | 13.905 |
| Geometrietyp der WKT | in allen 3.255.905 Zeilen `POLYGON`; kein `MULTIPOLYGON`, kein `LINESTRING` |
| Einheit | Meter (laut Zenodo-Beschreibung; Wanddicken im Median 0,16 m und Türbreiten im Median 0,86 m bestätigen das) |
| Koordinaten | lokales Koordinatensystem je Standort, georeferenziert; Pläne liegen gedreht, meist nicht achsparallel |

Zeilen je `entity_type/entity_subtype` (die größten; die Schreibweise der Subtypen ist durchgehend
Großbuchstaben mit Unterstrich):

| Typ / Subtyp | Zeilen |
|---|---|
| separator / WALL | 1.519.546 |
| opening / DOOR | 346.079 |
| opening / WINDOW | 260.081 |
| separator / RAILING | 158.398 |
| opening / ENTRANCE_DOOR | 108.776 |
| area / SHAFT | 100.389 |
| area / ROOM | 83.715 |
| area / BATHROOM | 70.317 |
| area / CORRIDOR | 63.514 |
| area / KITCHEN | 44.477 |
| area / BALCONY | 43.720 |
| area / LIVING_DINING | 24.435 |
| area / BEDROOM | 23.180 |
| separator / COLUMN | 22.869 |
| feature / (KITCHEN, SINK, TOILET, STAIRS, BATHTUB u. a.) | 317.973 |

Die vollständige Liste der Subtypen steht in `data/raw_stats.json`. Türen heißen `DOOR` und `ENTRANCE_DOOR`
(Haustür), Fenster `WINDOW`.

**Wände.** Eine Wand ist ein Polygon, in 99,3 % der Fälle ein Rechteck mit vier Eckpunkten (78.437 von
rund 79.000 Wandpolygonen in den 900 geprüften Plänen; Rest 3, 5, 6 und mehr Eckpunkte). Die Wanddicke liegt
im Median bei 0,16 m (5. bis 95. Perzentil: 0,06 bis 0,44 m). Die Wände liegen als einzelne Rechtecke
vor, die sich an Ecken und T-Stößen überlappen. In den
900 geprüften Plänen sind 9.304 der rund 79.000 Wandzeilen (11,8 %) exakte Duplikate eines anderen
Wandpolygons desselben Geschosses (Wände zwischen zwei Einheiten stehen doppelt). Zusätzlich überlappen
sich Rechtecke teilweise (Abschnitt 4.1).

**Öffnungen.** Türen und Fenster sind Rechtecke (immer vier Eckpunkte), die in der Wand liegen: bei
99,96 % von 34.311 Öffnungen liegen mehr als 90 % der Öffnungsfläche in der Wandfläche (der Wert ist für
die meisten Öffnungen 95,2 %). Die Wand ist an der Öffnung nicht unterbrochen. Haustüren stehen häufig
doppelt (2.534 von 5.794 Zeilen in den 900 Plänen sind exakte Duplikate, weil die Tür an zwei Einheiten
grenzt). Mehrflügelige Balkontüren erscheinen als mehrere nebeneinanderliegende `DOOR`-Polygone.

**Räume.** `area`-Polygone sind die lichte Fläche eines Raums (Rand an der Wandfläche; in drei von Hand
geprüften Plänen beträgt der Abstand zur Wandfläche im Median 1 mm). Sie haben vier Eckpunkte
(11.864 von 27.471 Flächen in den 900 geprüften Plänen) oder mehr (85,6 % haben höchstens zwölf).
Flächen im Median (Flächen ab 1 m², ohne Außenbereiche): Zimmer 14,4 m², Bad 3,8 m², Flur 8,3 m²,
Küche 8,1 m², Wohnen/Essen 29,4 m². Offene Nachbarräume (Küche und Wohnen ohne Wand) sind als getrennte
Flächen angelegt, die sich in den von Hand geprüften Plänen im Abstand von etwa 1 cm gegenüberstehen.
Außenbereiche (Balkon, Loggia, Terrasse) sind eigene Flächen.

**Wiederholung.** 2.146 Pläne gehören zu mehreren Geschossen (`plan_id` gleich, `floor_id` verschieden); in
allen 2.146 sind die Geometrie-Zeilen der Geschosse identisch (Prüfsumme über Typ, Subtyp und WKT). Das
sind 4.964 Geschosse, die ein anderes Geschoss desselben `plan_id` wiederholen. Weitere 12 `plan_id`
tragen dieselbe Geometrie wie ein anderer `plan_id`.

## 3. Aufbereitung je Plan

1. Je `plan_id` das Geschoss mit der kleinsten `floor_id`. Wiederholungen entfallen, 12 identische
   Pläne unter anderem `plan_id` ebenfalls.
2. Exakte Duplikate (Rundung 5 mm) entfallen; Türen und Fenster, die sich zu mindestens 50 % überdecken
   (gleiche Art), zählen einmal.
3. Drehung um die Hauptrichtung der Wände (längengewichteter Mittelwert der vierfachen Richtung), damit
   die Zeichnung achsparallel liegt wie die meisten CAD-Pläne. 835 der 900 geprüften Pläne waren um mehr
   als 1 Grad gedreht (Median 21 Grad). Längen und Flächen ändern sich dadurch nicht.
4. Verschiebung, sodass die kleinste Koordinate bei (10 m, 10 m) liegt.
5. Ausgabe als DXF in Millimeter (`$INSUNITS` = 4). Die Ground Truth steht in Metern im Koordinatensystem
   der DXF (DXF-Wert geteilt durch 1000, gleicher Ursprung).

## 4. Definitionen der Ground Truth

Alle Werte stehen je DXF-Datei in der Schwester-JSON unter `gt`.

### 4.1 Wandlänge (Achslänge)

`wandlaenge_achse_m` ist die Summe der Achslängen aller Wände, Achse zu Achse.

Verfahren aus den Polygonen:

1. Jedes Wandrechteck wird durch sein kleinstes umschließendes Drehrechteck ersetzt: Längsachse, Länge,
   Dicke.
2. Rechtecke derselben Wand werden zusammengefasst: Richtung innerhalb 1,5 Grad, Achsversatz höchstens
   35 % der größeren Dicke, Lücke zwischen den Teilstücken höchstens 2 cm. Doppelt gezeichnete und in Teilstücke
   zerlegte Wände zählen dadurch einmal. Die zusammengefasste Länge heißt `wandlaenge_flaechenkante_m`
   (Länge von Wandende zu Wandende der Rechtecke, ohne Knotenkorrektur).
3. Wandenden, die an einer anderen Wand enden (Winkel mindestens 15 Grad, Abstand höchstens halbe Dicke
   der Gegenwand plus 5 cm), werden auf den Schnittpunkt der beiden Achsen gezogen oder gekürzt. Das ergibt
   die Achslänge: ein T-Stoß oder eine Ecke wird bis zur Achse der anderen Wand gerechnet.
4. Öffnungen unterbrechen die Wand nicht (die Wand läuft laut Rohdaten durch).

Robustheit (belegt):

- Achslänge und Flächenkantenlänge unterscheiden sich in den 900 geprüften Plänen im Median um 1,2 %, im
  90. Perzentil um 2,7 %, im 99. um 4,9 %, höchstens um 7,1 %. Im Goldset liegt das Verhältnis
  Flächenkante zu Achse zwischen 0,961 und 1,034. Eine Pipeline, die Wandenden an der Wandfläche statt an
  der Achse abschneidet, liegt also systematisch um bis zu etwa 4 % darunter.
- Die Summe der Rechtecklängen ohne Zusammenfassen liegt im Goldset im Median 7,5 % über der Achslänge
  (0,975 bis 1,313), weil Überlappungen doppelt zählen. Diese Größe taugt nicht als Wahrheit.

Nicht eindeutig und deshalb Ausschlussgründe:

- **Teilweise überlappende parallele Wände** (`wand_parallel_mehrdeutig`): Rechtecke, die weder dieselbe
  Wand sind noch bündig nebeneinanderliegen. Ob die Überlappung doppelt zu zählen ist, entscheidet kein
  Algorithmus. Ein Plan fällt heraus, wenn die so unsichere Länge mehr als 2 % der Achslänge ausmacht.
- **Nicht rechteckige Wände** (`wand_nicht_rechteckig`): Polygone, deren Fläche weniger als 93 % ihres
  umschließenden Drehrechtecks beträgt (Wandnischen, schräge Enden). 243 von 69.711 Wandrechtecken
  (0,35 %). Ein Plan fällt heraus, wenn diese Wände mehr als 2 % der Wandfläche stellen.

### 4.2 Türen

`tueren` ist die Zahl der Polygone `DOOR` und `ENTRANCE_DOOR` nach dem Entfernen von Duplikaten.
`tueren_davon_haustuer` zählt die `ENTRANCE_DOOR` darunter. Mehrflügelige Türen zählen je Polygon einmal,
wie im Datensatz. Breite und Dicke einer Öffnung messen entlang und quer zur Wirtswand. Wirtswand ist die
Wand mit der größten Überdeckung. Liegt weniger als 60 % einer Öffnung in einer Wand, fällt der Plan heraus
(`oeffnung_ohne_wand`). Überlappen sich zwei Öffnungen um mehr als 10 % der kleineren Fläche (nicht
dieselbe Öffnung), fällt der Plan heraus (`oeffnung_ueberlappt`).

### 4.3 Fenster

`fenster` ist die Zahl der Polygone `WINDOW` nach dem Entfernen von Duplikaten, gleiche Regeln wie bei Türen.
Der Datensatz enthält schmale Fenster (5. Perzentil der Breite 0,59 m) und breite (95. Perzentil 2,8 m).

### 4.4 Räume und Raumfläche

Ein Raum im Sinn dieser Ground Truth ist ein wandbegrenzter Raum:

- Jede `area`-Fläche mit mindestens 1,0 m² zählt, außer Außenbereichen (`BALCONY`, `LOGGIA`, `TERRACE`,
  `GARDEN`, `PATIO`, `OUTDOOR_VOID`, `LIGHTWELL`, `CARPARK`, `RAMP`) und eingeschlossenen Nicht-Räumen
  (`SHAFT`, `ELEVATOR`, `VOID`). Die Nicht-Räume stehen getrennt unter `nichtraeume`, weil eine Pipeline,
  die Flächen aus dem Wandnetz bildet, auch sie als Fläche findet.
- Flächen, die sich ohne Wand berühren (Abstand höchstens 2 cm über mindestens 20 cm Kantenlänge), sind
  ein Raum. In den 900 geprüften Plänen sind das 1.571 von 16.418 Räumen (9,6 %) in 604 Plänen. Die
  Raumfläche ist die Summe der Teilflächen, die Kontur das Polygon nach Schließen der 1-cm-Lücke, der
  Name der des größten Teils.
- `flaeche_m2` ist die lichte Fläche (Polygonfläche der `area`-Entitäten, Summe der Teile). Eine
  Pipeline, die Räume aus den Wandachsen bildet, liefert zwangsläufig größere Flächen. Diese Abweichung
  ist ein Befund über das Verfahren der Pipeline.
- Ein Raum muss wandbegrenzt sein: mindestens 90 % seiner Kontur liegen höchstens 5 cm von Wand, Öffnung
  oder Stütze entfernt. Sonst fällt der Plan heraus (`raum_nicht_wandbegrenzt`): Der Raum ist zur Seite
  offen, und kein Verfahren auf Wänden kann ihn finden.
- Überlappen sich Flächen um mehr als 5 % der kleineren, fällt der Plan heraus (`flaechen_ueberlappen`).
  Besteht ein Raum nicht aus einem einzigen Polygon, ebenso (`raum_mehrteilig`).

`raeume` ist die Zahl dieser Räume, `raumflaeche_summe_m2` die Summe der lichten Flächen.

### 4.5 Was die Ground Truth nicht enthält

Wandhöhen, Wandtypen (außen/innen sind nur Layerhilfe beim Zeichnen), Materialien, Deckenflächen,
Treppen und Möbel. Mengen wie Wandfläche oder Rauminhalt sind abgeleitet und nicht eigens geprüft.

## 5. Ausschlüsse (gezählt)

Vorfilter über alle 8.941 Pläne (`data/selection.json`, `vorfilter`):

| Grund | Pläne |
|---|---|
| gleiche Geometrie wie ein anderer `plan_id` | 12 |
| weniger als 20 Wandzeilen im gewählten Geschoss | 190 |
| mehr als 170 Wandzeilen (Dateigröße, Budget 6 MB) | 1.150 |
| verbleibende Kandidaten | 7.589 (klein 2.444, mittel 3.595, groß 1.550) |

Klassen nach Wandzeilen: klein 20 bis 60, mittel 61 bis 110, groß 111 bis 170.

Prüfung der Wahrheit an 900 Kandidaten (je Klasse 300, in der festen Zufallsreihenfolge des Seeds; nicht
an allen 7.589, weil das Laden der CSV je Durchlauf etwa eine Minute braucht und die Prüfung je Plan
etwa 0,1 bis 0,2 Sekunden):

| Ergebnis | Pläne |
|---|---|
| zulässig | 656 (72,9 %) |
| ausgeschlossen | 244 (27,1 %) |

Ausschlussgründe (ein Plan kann mehrere haben; "erster" ist der zuerst festgestellte Grund):

| Grund | alle | erster |
|---|---|---|
| `raum_nicht_wandbegrenzt` | 134 | 118 |
| `wand_parallel_mehrdeutig` | 100 | 95 |
| `wand_nicht_rechteckig` | 29 | 29 |
| `oeffnung_ueberlappt` | 3 | 2 |
| `oeffnung_ohne_wand`, `flaechen_ueberlappen`, `raum_mehrteilig`, `keine_raeume`, `zu_wenige_waende` | 0 | 0 |

Je Klasse: klein 65 von 300 ausgeschlossen, mittel 94 von 300, groß 85 von 300. Pläne mit offenem Raum
(`raum_nicht_wandbegrenzt`): klein 33, mittel 61, groß 40.

Aussagekraft: Die Anteile gelten für die 900 geprüften Pläne der drei Größenklassen. Für Pläne mit mehr
als 170 oder weniger als 20 Wandzeilen wurde die Prüfung nicht gefahren. Aus den 656 zulässigen Plänen wurde
ausgewählt und danach für das Größenbudget gekürzt (Abschnitt 6).

## 6. Auswahl und Umfang

- Fester Seed 20201020. Die Kandidaten je Klasse werden nach `plan_id` sortiert und dann mit dem Seed
  gemischt (reproduzierbar).
- Aus den zulässigen Plänen je Klasse ausgewogen nach Raumzahl (1 bis 5, 6 bis 12, ab 13 Räume), Quote
  klein 16, mittel 16, groß 12 als Ziel; das Größenbudget von 6 MB führte zu klein 16, mittel 16, groß 8 (40 Pläne). Zulässige Pläne mit gleicher Signatur (Wandlängen, Dicken, Türen,
  Fenster, Räume) zählen einmal; die Prüfung fand 2 solche Fälle.
- Jeder Plan wird in zwei Varianten verschiedener Konvention gezeichnet. Das Budget (DXF plus JSON höchstens
  5,8 MB) kürzt die Auswahl von den größten Plänen aus, bis 40 Pläne erreicht sind, danach bekommen die
  größten Pläne nur noch eine Variante.
- Ergebnis: **40 Pläne, 74 DXF-Dateien mit je einer Schwester-JSON**, 5.679.163 Bytes für `plans/`,
  5.745.875 Bytes für den ganzen Ordner `data/`. 34 Pläne haben zwei Varianten, 6 eine.
  Klassen: klein 16, mittel 16, groß 8.
- Summen der Wahrheit über die 40 Pläne (einmal gezählt): 6.332 m Wandachse, 625 Türen, 459 Fenster,
  494 Räume.

Mehr Pläne brauchen ein größeres Budget (`--budget-bytes`); ein Plan kostet im Mittel etwa 140 KB (alle
Dateien eines Plans, DXF und JSON).

## 7. Zeichenkonventionen

Vier Konventionen, je zwei Varianten (Wandausführung und Unterbrechung):

| Konvention | Wände | Türen | Fenster | Räume | Blockname |
|---|---|---|---|---|---|
| `aia` | `A-WALL` | `A-DOOR` | `A-GLAZ` | `A-AREA` (Label) | `DR_0900`, `WN_1200` (Name ohne Aussage, nur der Layer sagt es) |
| `dach` | `A_Wand_a` (außen), `A_Wand_i` (innen) | `A_Tuer` | `A_Fenster` | `A_Raum` (Label) | `Tuer_0900`, `Fenster_1200` |
| `sib` | `300_WAND` | `300_TUEROEFFNUNGEN` | `300_FENSTEROEFFNUNGEN` | `0_RAUMPOLYGON` (Polygon und Label) | `TUER_0900`, `FENSTER_1200` |
| `archicad` | `STIFT_18` (außen), `STIFT_23` (innen) | Blöcke auf dem Wandlayer des Wirts | Blöcke auf dem Wandlayer des Wirts | `STIFT_7` (Label) | `AC_Tuer_0900`, `AC_Fenster_1200` |

Varianten (`data/manifest.json`, `varianten`): `aia_doppel`, `aia_kontur_unterbr`, `dach_doppel_unterbr`,
`dach_kontur`, `sib_doppel`, `sib_kontur_unterbr`, `archicad_doppel_unterbr`, `archicad_kontur`.

- **Wandausführung `doppel`**: je Wandrechteck zwei Linien über die volle Länge (`LINE`).
  **`kontur`**: geschlossene Polylinien der Wandfläche (Außen- und Innenwände je als Vereinigung; die
  Räume erscheinen als Innenkonturen).
- **Unterbrechung**: mit `unterbr` sind die Wandlinien an Türen und Fenstern aufgetrennt, bei `doppel` mit
  Laibungslinien quer zur Wand. Ohne `unterbr` laufen die Linien durch die Öffnung.
- **Öffnungen**: `INSERT`-Blöcke. Tür: Blatt (Linie) und Schwingbogen (Viertelkreis), Anschlagseite und
  Schwingrichtung zufällig (feste Zufallsquelle je Plan und Variante; die Rohdaten enthalten beides nicht). Fenster:
  Rechteck mit Mittellinie. Breiten auf 25 mm gerundet, ein Block je Breite. Der Einfügepunkt ist das Ende
  der Öffnung.
- **Räume**: `TEXT`-Label (ein Label je Raum, mittig auf einem Punkt im Raum, deutsche Bezeichnung aus dem
  Datensatz-Typ, z. B. "Küche", "Wohnen/Essen", "Flur"). Bei `sib` zusätzlich das Raumpolygon als
  geschlossene Polylinie.
- **Störelemente**: Bemaßungslinien (Maßlinie, Hilfslinien, Striche, Maßzahl als Text) rundum auf dem
  Annotationslayer, Schraffur der Wände (`A-WALL-PATT`, `A_Wand_Schraffur`, `300_WAND_SCHRAFFUR`,
  `STIFT_5`; als `HATCH`, bei `aia_kontur_unterbr` aufgelöst zu 45-Grad-Linien in den Außenwänden),
  drei Beschriftungstexte, Ausstattung und Treppen aus den `feature`-Polygonen, Geländer, Stützen.
  Layernamen der Schraffur enthalten absichtlich das Wort Wand.
- DXF R2000, ASCII, Zeilenende LF, feste Metadaten (keine Zeitstempel, keine GUIDs).
  Koordinaten auf ganze Millimeter gerundet.

## 8. Was nicht eindeutig war, und Grenzen

- **Wandlänge**: Achse zu Achse und Flächenkante unterscheiden sich um bis zu 4 % im Goldset (Abschnitt
  4.1). Die Wahrheit ist Achse zu Achse. Wer das nicht will, findet `wandlaenge_flaechenkante_m` daneben.
- **Teilweise überlappende Wände** und **nicht rechteckige Wände** sind nicht entscheidbar; solche Pläne
  fehlen (100 und 29 von 900 geprüften).
- **Offene Räume**: 134 von 900 Plänen haben mindestens einen zur Seite offenen Raum. Pläne mit offenen
  Räumen fehlen. Dadurch sind Raumsuch-Verfahren im Goldset leichter dran als im Bestand.
- **Zusammengefasste offene Nachbarräume**: Was der Datensatz in Küche und Wohnen trennt, gilt hier als
  ein Raum, weil keine Wand dazwischen steht. Ein Verfahren, das Küche und Wohnen trennen soll, kann
  diese Wahrheit nicht erreichen; ein Verfahren auf Wänden erreicht sie.
- **Mehrflügelige Türen** zählen je Polygon. Eine Pipeline, die benachbarte Blöcke zu einer Tür
  zusammenfasst, bekommt weniger Treffer.
- **Türen mit zufälliger Anschlagseite**: der Einfügepunkt der Blöcke liegt am Öffnungsende, die
  Anschlagseite ist zufällig gewählt (die Rohdaten kennen sie nicht).
- **Konventionen**: synthetisch (Abschnitt 0). Tür- und Fensterblöcke tragen keine Attribute. Die Stiftnummern
  in `archicad` sind frei gewählt.
- **Koordinatengenauigkeit**: DXF auf 1 mm, GT-Listen (Polygone, Mittelpunkte) auf 1 cm; Längen und
  Flächen exakt.
- **Größe**: Pläne über 170 Wandzeilen fehlen wegen des Budgets (1.150 von 8.941 Plänen); große Pläne
  sind also unterrepräsentiert.
- **Bestehende Pipeline**: Der Lader von `planproof_geometry` liest `LINE`, `LWPOLYLINE`, `POLYLINE`, `INSERT`,
  `TEXT` und `MTEXT` und ignoriert `HATCH`. Die Schraffur als `HATCH` erreicht ihn also nicht. Die aufgelöste
  Schraffur in `aia_kontur_unterbr` liegt auf `A-WALL-PATT`, und der Klassifikator ordnet jeden Layer mit dem
  Teilwort `a-wall` den Wänden zu (laut Code). Die Wirkung auf die Messung wurde nicht getrennt untersucht.

## 9. Messung (`metrics.py`, `run_goldset.py`)

`run_goldset.py` führt jede DXF durch eine Pipeline-Funktion (`--pipeline planproof_geometry`,
`planproof_cad` oder `modul:funktion`), liest die `scene.json` und schreibt `report.json` (je Datei) und
`report.md` (Tabelle). Es werden keine Cloud-Aufrufe gemacht.

Metriken je Datei:

- **Wandlänge**: Summe `length_m` der `WALL`-Komponenten gegen `wandlaenge_achse_m`, Abweichung in Prozent.
- **Türen und Fenster**: 1:1-Zuordnung nach Art und Abstand der Mittelpunkte (Toleranz halbe GT-Breite plus
  0,5 m), Recall = Treffer / GT, Precision = Treffer / Ergebnis, Abweichung der Zahl.
- **Räume**: ein GT-Raum ist getroffen, wenn sein Schwerpunkt (sonst ein Innenpunkt) im Ergebnis-Polygon
  liegt. Je Raum relative Flächenabweichung (Ergebnis minus GT durch GT), Median und 90. Perzentil des
  Betrags, Median mit Vorzeichen. Ergebnis-Räume, die einen eingeschlossenen Nicht-Raum (Schacht,
  Aufzug) treffen, zählen nicht gegen die Precision.
- **Ausrichtung**: Die Pipeline schiebt den Ursprung. `metrics.py` probiert drei Verschiebungen (Ursprung
  der DXF laut Ladeverfahren, keine Verschiebung, Bounding-Box der Wandachsen) und nimmt die mit den
  meisten Treffern. Der gewählte Modus steht im Bericht. Das ist eine leichte Begünstigung des Ergebnisses.

Basislauf am 20.10.2026 mit `planproof_geometry` (Pipeline `dwg_to_bim`, Stand Working Tree, lokal,
74 Dateien, 50 Sekunden, keine Fehler):

| Konvention | Dateien | Fehler | Wandlänge Abw. Median / P90 (%) | Tür Recall / Precision | Fenster Recall / Precision | Raum Recall / Precision | Raumzahl Abw. Median | Raumfläche Abw. Median / P90 (%) (n Räume) |
|---|---|---|---|---|---|---|---|---|
| aia | 20 | 0 | 34,0 / 57,8 | 89,0 % / 91,9 % | 82,7 % / 84,9 % | 40,1 % / 100,0 % | 5,5 | 12,6 / 43,8 (103) |
| archicad | 17 | 0 | 20,6 / 46,9 | 81,0 % / 85,0 % | 77,3 % / 80,5 % | 3,3 % / 100,0 % | 10,0 | 6,1 / 12,0 (6) |
| dach | 19 | 0 | 37,6 / 54,1 | 77,7 % / 85,5 % | 57,1 % / 73,0 % | 0,9 % / 100,0 % | 11,0 | 179,2 / 252,0 (2) |
| sib | 18 | 0 | 26,5 / 58,5 | 85,7 % / 91,2 % | 76,6 % / 84,8 % | 31,0 % / 100,0 % | 6,0 | 16,1 / 90,5 (71) |
| **gesamt** | 74 | 0 | 30,9 / 55,6 | 83,7 % / 88,8 % | 74,3 % / 81,6 % | 20,7 % / 100,0 % | 8,0 | 14,3 / 67,0 (182) |

Lesehilfe:

- Die Wandlänge ist im Median 30,9 % zu kurz (mit Vorzeichen -30,9 %). Je Variante (Median des Betrags):
  `sib_doppel` 2,3 %, `aia_doppel` 4,9 %, `archicad_kontur` 15,4 %, `archicad_doppel_unterbr` 21,0 %,
  `dach_doppel_unterbr` 33,3 %, `dach_kontur` 43,7 %, `aia_kontur_unterbr` 49,6 %, `sib_kontur_unterbr`
  51,3 %. Nur die beiden Varianten mit durchgehenden Doppellinien auf erkannten Layern liegen unter 5 %.
  Mit Konturen oder an Öffnungen unterbrochenen Wänden liegt der Median zwischen 15 % und 51 %.
- Die Raumerkennung findet 20,7 % der Räume (182 von 878). Die Precision von 100 % sagt wenig, weil
  wenige Räume gefunden werden. Die Flächenabweichung gilt nur für die gefundenen Räume (Spalte n),
  bei `dach` sind es 2, bei `archicad` 6; die Zahlen dort sind kein Maß.
- Die gefundenen Räume sind im Median 14,3 % zu groß (Vorzeichen positiv). Das erwartet man, weil die
  Pipeline Räume aus Wandachsen bildet, die Wahrheit aber die lichte Fläche ist.
- Türen und Fenster werden zu 84 % und 74 % gefunden (Zuordnung nach Position). Die Ursache der
  Verluste wurde nicht untersucht.
- `planproof_cad` (Kopie des Codes im Paket, Stand 20.10.2026) liefert dieselben Zahlen.
- Alle Werte sind Zählungen und Prozente aus 74 Dateien von 40 Plänen. Streuung zwischen Plänen ist
  groß; ein Konfidenzintervall wurde nicht berechnet.

## 10. Nachbauen und prüfen

```
# Rohdaten laden und md5 prüfen (außerhalb des Repos)
#   https://zenodo.org/records/7788422  ->  swiss-dwellings-v3.0.0.zip
#   md5 3b7915ecd5bf8e492a3e78c598b74123 ;  geometries.csv entpacken

python inspect_raw.py  --csv <pfad>/geometries.csv --index-cache <pfad>/floor_index.json   # Rohdaten-Zählung
python build_goldset.py --csv <pfad>/geometries.csv --index-cache <pfad>/floor_index.json  # Goldset bauen
python run_goldset.py  --pipeline planproof_geometry --out <ordner>                           # messen

pip install -r requirements.txt
python -m pytest tests -q                              # Tests ohne Rohdaten
SD_GEOMETRIES_CSV=<pfad>/geometries.csv python -m pytest tests/test_reproducibility.py   # zweimal bauen
```

Zweimal bauen ergibt byte-identische Dateien (SHA-256 aller 74 DXF verglichen). Das gilt für gleiche
Rohdaten, gleichen Seed und gleiche Versionen von ezdxf (1.4.5) und shapely (2.2.0).
