# Goldset-Ergebnisse der Pipeline planproof_cad

Messung der DXF-Pipeline auf dem Goldset (40 Pläne aus Swiss Dwellings, 74 DXF, vier synthetische
Zeichenkonventionen). Gemessen mit `goldset/run_goldset.py`, Pipeline `planproof_cad`, lokal, ohne Cloud
und ohne Modellaufruf. Alle Zahlen stammen aus `report.json` der jeweiligen Läufe.

**Was die Zahlen beweisen und was nicht.** Das Goldset beweist die Pipeline auf Swiss-Dwellings-Geometrie mit
selbst gewählten Layern. Es beweist sie nicht auf echten Büro-Exporten aus Allplan, ArchiCAD oder
Vectorworks (Abschnitt 7).

## 1. Aufbau der Messung

- **Trennung.** `--split dev` (30 Pläne, 54 Dateien) und `--split heldout` (10 Pläne, 20 Dateien;
  Regel `int(sha256(plan_id).hexdigest(), 16) % 4 == 0`). Entwickelt wurde nur gegen dev. Jeder Lauf mit
  heldout oder all steht mit Zeitpunkt und Kommentar in `heldout_runs.log` (vier Läufe: Basis, drei
  Meilensteine nach den Wänden, nach den Räumen und am Ende). Der Basislauf lief mit dem unveränderten
  Code und ist ein vierter Lauf neben den drei Meilensteinen; er floss in keine Entscheidung ein.
- **Metriken** wie in `GROUND_TRUTH.md` Abschnitt 9. Die Ausrichtung (Verschiebung zur Ground Truth) wählt
  `metrics.py` aus drei Kandidaten; das begünstigt das Ergebnis leicht.
- **Streuung.** 30 beziehungsweise 10 Pläne sind wenig. Ein Konfidenzintervall wurde nicht berechnet. Die
  Held-out-Zahlen hängen an 20 Dateien und einzelne Pläne verschieben sie sichtbar.

## 2. Vorher und Nachher

### 2.1 dev (54 Dateien), Basis gegen Endstand

| Konvention | Wandlänge Abw. Median (%) | Tür Recall (%) | Fenster Recall (%) | Raum Recall (%) | Raumfläche Abw. Median (%) |
|---|---|---|---|---|---|
| aia | 33,1 -> 7,6 | 88,7 -> 96,1 | 81,5 -> 95,1 | 45,4 -> 67,6 | 12,2 -> 0,3 |
| archicad | 20,3 -> 3,6 | 81,8 -> 97,5 | 76,6 -> 98,1 | 4,8 -> 61,5 | 6,0 -> 0,6 |
| dach | 36,5 -> 4,0 | 77,3 -> 96,1 | 53,1 -> 98,4 | 1,3 -> 62,5 | 179,2 -> 0,4 |
| sib | 37,0 -> 8,5 | 81,4 -> 92,4 | 68,8 -> 93,1 | 29,2 -> 58,5 | 16,8 -> 0,6 |
| **gesamt** | 32,2 -> 7,0 | 82,9 -> 95,3 | 70,8 -> 95,9 | 23,8 -> 62,9 | 14,2 -> 0,4 |

### 2.2 held-out (20 Dateien), Basis gegen Endstand

| Konvention | Wandlänge Abw. Median (%) | Tür Recall (%) | Fenster Recall (%) | Raum Recall (%) | Raumfläche Abw. Median (%) |
|---|---|---|---|---|---|
| aia | 57,6 -> 6,8 | 90,3 -> 98,4 | 87,2 -> 93,6 | 18,0 -> 74,0 | 19,6 -> 0,3 |
| archicad | 21,3 -> 13,6 | 80,0 -> 97,3 | 78,4 -> 90,5 | 1,3 -> 61,8 | 12,9 -> 0,3 |
| dach | 39,8 -> 3,0 | 78,6 -> 100,0 | 68,1 -> 97,9 | 0,0 -> 68,3 | - -> 0,5 |
| sib | 0,3 -> 0,8 | 98,6 -> 97,1 | 98,3 -> 98,3 | 36,2 -> 89,7 | 14,8 -> 0,3 |
| **gesamt** | 23,9 -> 5,9 | 85,6 -> 98,2 | 83,2 -> 94,7 | 12,7 -> 72,5 | 15,6 -> 0,3 |

### 2.3 Endstand dev, volle Tabelle

| Konvention | Dateien | Wandlänge Abw. Median / P90 (%) | Tür R / P (%) | Fenster R / P (%) | Raum R / P (%) | Raumfläche Abw. Median / P90 (%) (n) |
|---|---|---|---|---|---|---|
| aia | 17 | 7,6 / 14,9 | 96,1 / 98,4 | 95,1 / 97,2 | 67,6 / 98,6 | 0,3 / 2,4 (140) |
| archicad | 10 | 3,6 / 22,6 | 97,5 / 98,3 | 98,1 / 98,1 | 61,5 / 92,8 | 0,6 / 7,2 (64) |
| dach | 14 | 4,0 / 13,2 | 96,1 / 96,7 | 98,4 / 99,2 | 62,5 / 100,0 | 0,4 / 22,1 (95) |
| sib | 13 | 8,5 / 17,8 | 92,4 / 95,6 | 93,1 / 96,1 | 58,5 / 100,0 | 0,6 / 8,0 (100) |
| **gesamt** | 54 | 7,0 / 17,8 | 95,3 / 97,2 | 95,9 / 97,5 | 62,9 / 98,3 | 0,4 / 7,0 (399) |

### 2.4 Endstand held-out, volle Tabelle

| Konvention | Dateien | Wandlänge Abw. Median / P90 (%) | Tür R / P (%) | Fenster R / P (%) | Raum R / P (%) | Raumfläche Abw. Median / P90 (%) (n) |
|---|---|---|---|---|---|---|
| aia | 3 | 6,8 / 8,7 | 98,4 / 100,0 | 93,6 / 97,8 | 74,0 / 100,0 | 0,3 / 1,6 (37) |
| archicad | 7 | 13,6 / 36,9 | 97,3 / 97,3 | 90,5 / 95,7 | 61,8 / 90,4 | 0,3 / 4,0 (47) |
| dach | 5 | 3,0 / 14,1 | 100,0 / 100,0 | 97,9 / 100,0 | 68,3 / 100,0 | 0,5 / 2,7 (41) |
| sib | 5 | 0,8 / 17,9 | 97,1 / 98,6 | 98,3 / 98,3 | 89,7 / 100,0 | 0,3 / 2,0 (52) |
| **gesamt** | 20 | 5,9 / 31,9 | 98,2 / 98,8 | 94,7 / 97,7 | 72,5 / 97,2 | 0,3 / 2,7 (177) |

### 2.5 Basis dev und held-out, volle Tabelle (Vergleich)

Basis dev:

| Konvention | Dateien | Wandlänge Abw. Median / P90 (%) | Tür R / P (%) | Fenster R / P (%) | Raum R / P (%) | Raumfläche Abw. Median / P90 (%) (n) |
|---|---|---|---|---|---|---|
| aia | 17 | 33,1 / 53,5 | 88,7 / 91,9 | 81,5 / 84,3 | 45,4 / 100,0 | 12,2 / 31,5 (94) |
| archicad | 10 | 20,3 / 34,2 | 81,8 / 85,3 | 76,6 / 79,6 | 4,8 / 100,0 | 6,0 / 9,2 (5) |
| dach | 14 | 36,5 / 53,8 | 77,3 / 85,4 | 53,1 / 67,3 | 1,3 / 100,0 | 179,2 / 252,0 (2) |
| sib | 13 | 37,0 / 56,5 | 81,4 / 88,6 | 68,8 / 79,1 | 29,2 / 100,0 | 16,8 / 91,4 (50) |
| **gesamt** | 54 | 32,2 / 54,1 | 82,9 / 88,5 | 70,8 / 78,7 | 23,8 / 100,0 | 14,2 / 67,2 (151) |

Basis held-out:

| Konvention | Dateien | Wandlänge Abw. Median / P90 (%) | Tür R / P (%) | Fenster R / P (%) | Raum R / P (%) | Raumfläche Abw. Median / P90 (%) (n) |
|---|---|---|---|---|---|---|
| aia | 3 | 57,6 / 58,8 | 90,3 / 91,8 | 87,2 / 87,2 | 18,0 / 100,0 | 19,6 / 79,2 (9) |
| archicad | 7 | 21,3 / 47,8 | 80,0 / 84,6 | 78,4 / 81,7 | 1,3 / 100,0 | 12,9 / 12,9 (1) |
| dach | 5 | 39,8 / 53,3 | 78,6 / 85,7 | 68,1 / 88,9 | 0,0 / 0,0 | - / - (0) |
| sib | 5 | 0,3 / 46,0 | 98,6 / 98,6 | 98,3 / 98,3 | 36,2 / 100,0 | 14,8 / 25,6 (21) |
| **gesamt** | 20 | 23,9 / 57,8 | 85,6 / 89,4 | 83,2 / 88,7 | 12,7 / 100,0 | 15,6 / 61,5 (31) |

### 2.6 held-out an den Meilensteinen

| Schritt | Wandlänge Abw. Median / P90 (%) | Wandlänge Median mit Vorzeichen (%) | Tür Recall / Precision (%) | Fenster Recall / Precision (%) | Raum Recall / Precision (%) | Raumfläche Abw. Median / P90 (%) (n Räume) |
|---|---|---|---|---|---|---|
| Basis | 23,9 / 57,8 | -23,9 | 85,6 / 89,4 | 83,2 / 88,7 | 12,7 / 100,0 | 15,6 / 61,5 (31) |
| Meilenstein 1, nach den Wänden | 6,0 / 33,0 | 1,5 | 98,2 / 98,8 | 94,7 / 97,7 | 69,7 / 94,4 | 0,4 / 9,1 (170) |
| Meilenstein 2, nach den Räumen | 5,9 / 31,9 | 0,4 | 98,2 / 98,8 | 94,7 / 97,7 | 72,5 / 97,2 | 0,3 / 2,7 (177) |
| Meilenstein 3, Ende | 5,9 / 31,9 | 0,4 | 98,2 / 98,8 | 94,7 / 97,7 | 72,5 / 97,2 | 0,3 / 2,7 (177) |

## 3. Verlauf auf dev, Schritt für Schritt (gesamt)

| Schritt | Wandlänge Abw. Median / P90 (%) | Wandlänge Median mit Vorzeichen (%) | Tür Recall / Precision (%) | Fenster Recall / Precision (%) | Raum Recall / Precision (%) | Raumfläche Abw. Median / P90 (%) (n Räume) |
|---|---|---|---|---|---|---|
| 00 Basis (Code unverändert) | 32,2 / 54,1 | -32,2 | 82,9 / 88,5 | 70,8 / 78,7 | 23,8 / 100,0 | 14,2 / 67,2 (151) |
| 01 Layer-Klassifikator neu (Schraffur, Text, Abzug, oberhalb, Stiftlayer, bearing) | 32,7 / 55,2 | -32,7 | 82,9 / 88,5 | 69,4 / 77,8 | 23,8 / 100,0 | 14,2 / 67,2 (151) |
| 02 Wandrekonstruktion neu (Sammelschritt, siehe Text), Räume lichte Fläche und Label, Öffnungsbox, Layerwahl per Geometrie | 12,4 / 39,2 | -2,1 | 85,8 / 90,9 | 78,8 / 83,7 | 45,7 / 96,7 | 0,4 / 3,7 (290) |
| 03 Nächster Partner je Seite (Abschattung gestapelter Linien) | 13,8 / 32,7 | -11,8 | 85,2 / 90,5 | 77,7 / 82,7 | 42,3 / 96,4 | 0,3 / 2,2 (268) |
| 04 Gegenseitiger Knoten-Fangradius, zwei Verlängerungsdurchläufe | 13,8 / 33,3 | -11,8 | 85,3 / 90,7 | 77,4 / 82,3 | 43,5 / 96,5 | 0,3 / 2,4 (276) |
| 05 Linienmengen über Öffnungsblöcke verbunden, Achsen-Cluster global | 8,8 / 32,0 | -6,5 | 91,1 / 94,8 | 87,2 / 95,5 | 53,8 / 98,3 | 0,3 / 2,7 (341) |
| 06 Maximale Wanddicke 0,80 m (vorher 0,60 m) | 7,3 / 23,1 | -3,8 | 92,2 / 95,0 | 92,1 / 96,6 | 55,2 / 98,3 | 0,3 / 4,5 (350) |
| 07 Maximale Wanddicke 1,00 m | 7,3 / 19,6 | -2,0 | 92,3 / 95,2 | 94,3 / 96,6 | 56,3 / 98,4 | 0,4 / 12,2 (357) |
| 08 Cluster-Normalabstand statt Summe der Beträge | 7,3 / 17,0 | 0,1 | 95,2 / 97,1 | 96,0 / 97,5 | 63,9 / 98,5 | 0,5 / 14,3 (405) |
| 09 Abschattung über die Gegenseite für alle Kandidaten (verworfen) | 8,2 / 19,3 | -7,7 | 94,8 / 96,8 | 95,5 / 97,4 | 58,5 / 98,4 | 0,4 / 7,7 (371) |
| 10 Abschattung über die Gegenseite nur für dicke Kandidaten (Flur) | 7,0 / 17,8 | -2,3 | 95,3 / 97,2 | 95,9 / 97,5 | 62,9 / 98,3 | 0,4 / 7,0 (399) |
| 11 Endstand (gleicher Code wie 10, nach Ruff und Tests) | 7,0 / 17,8 | -2,3 | 95,3 / 97,2 | 95,9 / 97,5 | 62,9 / 98,3 | 0,4 / 7,0 (399) |

Versuche neben dem Hauptweg, jeweils gegen Schritt 10 gemessen und nicht übernommen:

| Schritt | Wandlänge Abw. Median / P90 (%) | Wandlänge Median mit Vorzeichen (%) | Tür Recall / Precision (%) | Fenster Recall / Precision (%) | Raum Recall / Precision (%) | Raumfläche Abw. Median / P90 (%) (n Räume) |
|---|---|---|---|---|---|---|
| V1 Achsenverlängerung mit Reserve 0,20 m statt 0,10 m | 7,2 / 16,5 | 0,1 | 94,9 / 96,9 | 96,0 / 97,5 | 63,9 / 98,5 | 0,4 / 17,6 (405) |
| V2 Knoten-Fangradius 1,0 mal Dicke plus 0,05 m statt 0,6 mal Dicke | 7,5 / 17,4 | -0,5 | 94,9 / 96,9 | 96,0 / 97,5 | 65,0 / 98,1 | 0,7 / 19,4 (412) |

Zu Schritt 02: Die Defekte wurden test-first bearbeitet (siehe `tests/test_pipeline_defects.py`, jeder
Test war vor dem Fix rot). Die Fixes für Wandpaarung, Brücken, Verlängerung, Blockauflösung, Einheitsprüfung,
Öffnungsbox, lichte Fläche, Labelregeln und Layerwahl entstanden zusammen und wurden zum ersten Mal als
Ganzes gemessen. Ihren einzelnen Beitrag zeigen deshalb die Ablationen in Abschnitt 5, nicht dieser Verlauf.

Zu Schritt 01 und 03: Schritt 01 allein verschlechterte den Fenster Recall von 70,8 auf 69,4 %. Die jetzt
ausgeschlossenen Schraffurlinien hatten in der Basis zufällig als Wirtswände für Öffnungen gedient. Er blieb,
weil der Klassifikator die geforderte Testmatrix erfüllen muss und die Wirkung klein ist. Schritt 03 senkte den
Raum Recall von 45,7 auf 42,3 % und brachte die Wandlänge P90 von 39,2 auf 32,7 % herunter (Phantomwände aus
gestapelten Linien). Er blieb, weil Wandmengen sonst nicht stimmen; die Raumverluste holten die späteren
Schritte 05 und 08 auf. Die Ablation in Abschnitt 5 zeigt, dass der Raum Recall ohne diese Abschattung auch im
Endstand höher läge (67,3 gegen 62,9 %), bei viel schlechteren Wandmengen.

Zu den verworfenen Schritten:

- **Schritt 09** schattete auch Kandidaten auf der Gegenseite einer Kante ab. Raum Recall fiel von 63,9 auf
  58,5 %, die Wandlänge bekam eine Abweichung von minus 7,7 % mit Vorzeichen. Schritt 10 beschränkt die
  Gegenseiten-Abschattung auf Kandidaten ab 0,4 m Dicke, deren Partner der anderen Seite höchstens halb so
  dick ist (Flur zwischen zwei dünnen Wänden). Gegen Schritt 08 kostet das etwa 1 Prozentpunkt Raum Recall und
  bringt die Flächen-P90 von 14,3 auf 7,0 %.
- **V1** (Reserve 0,20 m) änderte den Raum Recall nicht (63,9 %). Die offenen Enden liegen nicht an der
  Reserve.
- **V2** (größerer Fangradius) hob den Raum Recall um 1,1 Prozentpunkte auf 65,0 %, verschlechterte aber die
  Flächen-P90 von 14,3 auf 19,4 % und die Konvention `sib` bei der Fläche (Median 0,7 auf 1,2 %).

## 4. Endstand dev je Variante

| Variante | Dateien | Wandlänge Abw. Median (%) | Wandlänge Median mit Vorzeichen (%) | Tür Recall (%) | Fenster Recall (%) | Raum Recall (%) (Treffer/GT) |
|---|---|---|---|---|---|---|
| aia_doppel | 9 | 5,4 | 1,6 | 99,3 | 100,0 | 94,9 (111/117) |
| aia_kontur_unterbr | 8 | 9,4 | -9,4 | 91,5 | 89,7 | 32,2 (29/90) |
| archicad_doppel_unterbr | 7 | 3,8 | 1,3 | 96,3 | 100,0 | 60,0 (42/70) |
| archicad_kontur | 3 | 0,7 | -0,4 | 100,0 | 94,7 | 64,7 (22/34) |
| dach_doppel_unterbr | 7 | 2,7 | 0,2 | 93,7 | 100,0 | 62,0 (49/79) |
| dach_kontur | 7 | 10,0 | -3,5 | 98,8 | 95,7 | 63,0 (46/73) |
| sib_doppel | 5 | 8,2 | 6,4 | 100,0 | 100,0 | 81,0 (68/84) |
| sib_kontur_unterbr | 8 | 11,5 | -11,5 | 85,2 | 87,4 | 36,8 (32/87) |

Endstand held-out je Variante (wenige Dateien je Variante, nur zur Orientierung):

| Variante | Dateien | Wandlänge Abw. Median (%) | Wandlänge Median mit Vorzeichen (%) | Tür Recall (%) | Fenster Recall (%) | Raum Recall (%) (Treffer/GT) |
|---|---|---|---|---|---|---|
| aia_doppel | 1 | 0,5 | 0,5 | 100,0 | 100,0 | 100,0 (34/34) |
| aia_kontur_unterbr | 2 | 8,0 | -8,0 | 95,7 | 88,0 | 18,8 (3/16) |
| archicad_doppel_unterbr | 3 | 10,5 | -2,1 | 97,8 | 92,9 | 48,0 (12/25) |
| archicad_kontur | 4 | 24,7 | 20,4 | 96,9 | 89,1 | 68,6 (35/51) |
| dach_doppel_unterbr | 2 | 5,1 | -3,3 | 100,0 | 100,0 | 20,0 (2/10) |
| dach_kontur | 3 | 3,0 | 0,5 | 100,0 | 97,3 | 78,0 (39/50) |
| sib_doppel | 4 | 0,5 | 0,5 | 98,5 | 100,0 | 98,1 (52/53) |
| sib_kontur_unterbr | 1 | 28,4 | -28,4 | 75,0 | 80,0 | 0,0 (0/5) |

Basis dev je Variante:

| Variante | Dateien | Wandlänge Abw. Median (%) | Wandlänge Median mit Vorzeichen (%) | Tür Recall (%) | Fenster Recall (%) | Raum Recall (%) (Treffer/GT) |
|---|---|---|---|---|---|---|
| aia_doppel | 9 | 5,5 | -0,5 | 100,0 | 97,9 | 80,3 (94/117) |
| aia_kontur_unterbr | 8 | 42,8 | -42,8 | 72,6 | 63,2 | 0,0 (0/90) |
| archicad_doppel_unterbr | 7 | 20,6 | -20,6 | 79,0 | 73,9 | 0,0 (0/70) |
| archicad_kontur | 3 | 11,7 | -11,7 | 87,5 | 81,6 | 14,7 (5/34) |
| dach_doppel_unterbr | 7 | 29,6 | -29,6 | 78,9 | 51,9 | 0,0 (0/79) |
| dach_kontur | 7 | 43,5 | -43,5 | 75,6 | 55,3 | 2,7 (2/73) |
| sib_doppel | 5 | 4,1 | 0,8 | 100,0 | 94,5 | 59,5 (50/84) |
| sib_kontur_unterbr | 8 | 49,0 | -49,0 | 63,9 | 47,1 | 0,0 (0/87) |

## 5. Ablationen auf dev (je ein Bestandteil abgeschaltet, Rest wie Endstand)

| Schritt | Wandlänge Abw. Median / P90 (%) | Wandlänge Median mit Vorzeichen (%) | Tür Recall / Precision (%) | Fenster Recall / Precision (%) | Raum Recall / Precision (%) | Raumfläche Abw. Median / P90 (%) (n Räume) |
|---|---|---|---|---|---|---|
| Endstand | 7,0 / 17,8 | -2,3 | 95,3 / 97,2 | 95,9 / 97,5 | 62,9 / 98,3 | 0,4 / 7,0 (399) |
| ohne Achsenverlängerung auf Schnittpunkte | 7,8 / 18,2 | -3,7 | 95,4 / 97,5 | 96,4 / 98,4 | 29,8 / 97,9 | 1,6 / 6,8 (189) |
| ohne Lückenschluss (Brücken) | 15,5 / 28,1 | -13,4 | 81,1 / 87,1 | 72,7 / 78,8 | 37,7 / 100,0 | 0,3 / 1,7 (239) |
| ohne Nächster-Partner-Abschattung | 10,5 / 43,1 | 9,3 | 96,0 / 97,5 | 96,9 / 98,2 | 67,3 / 98,4 | 0,7 / 14,0 (427) |
| ohne Wandlayer-Wahl aus der Geometrie | 7,9 / 21,9 | -0,1 | 95,3 / 97,2 | 95,5 / 97,2 | 61,7 / 96,5 | 0,4 / 14,2 (391) |
| Raumfläche als Achsfläche (wie vor der lichten Fläche) | 7,0 / 17,8 | -2,3 | 95,3 / 97,2 | 95,9 / 97,5 | 62,9 / 98,5 | 14,7 / 32,4 (399) |
| maximale Wanddicke 0,60 m | 7,6 / 26,6 | -4,6 | 93,9 / 96,7 | 88,6 / 96,4 | 56,5 / 98,1 | 0,4 / 4,8 (358) |

Lesehilfe: Die Schalter heißen `wall_extend_junctions`, `wall_bridge_gaps`, `wall_nearest_partner`,
`wall_layer_vote`, `room_area_mode` und `wall_max_thickness_m` (Pipeline-Konfiguration; Läufe mit
`run_goldset.py --config-json`).

- Ohne Brücken sinken Tür Recall auf 81,1 %, Fenster Recall auf 72,7 % und Raum Recall auf 37,7 %. Die
  Wandlänge liegt im Median 13,4 % zu kurz. An Öffnungen unterbrochene Wände sind für Türen, Fenster und Wandlänge der größte Einzelposten.
- Ohne Verlängerung auf Achsenschnittpunkte halbiert sich der Raum Recall (29,8 %); die Wandlänge ändert
  sich dabei wenig (die Ecken fehlen nur als kurze Stücke).
- Ohne Nächster-Partner-Abschattung entstehen Phantomwände aus gestapelten Linien: Wandlänge im Median
  10,5 % und P90 43,1 % daneben, mit einer Abweichung von plus 9,3 % mit Vorzeichen. Der Raum Recall liegt
  dabei höher (67,3 %); mehr Linien bedeuten mehr geschlossene Flächen, auch falsche. Die Flächen-P90 liegt
  bei 14,0 %. Der Schalter bleibt im Endstand an, weil die Wandmengen sonst nicht stimmen.
- Ohne Layerwahl aus der Geometrie fallen alle Linien unbekannter Layer als Wandkandidaten an. Das betrifft
  nur `archicad`: Wandlänge Median der Konvention 3,6 auf 18,8 %, Raum Recall 61,5 auf 53,8 %, Raum Precision 92,8 auf 82,3 %.
- Mit Achsfläche statt lichter Fläche liegt die Raumfläche im Median 14,7 % (P90 32,4 %) über der Ground
  Truth. Mit lichter Fläche sind es 0,4 % (P90 7,0 %).
- Mit maximaler Wanddicke 0,60 m fehlen Fenster (88,6 % statt 95,9 %) und Wandlänge (P90 26,6 % statt
  17,8 %): einzelne Pläne haben Wände über 0,6 m.

Blockauflösung (Defekt 5), Einheitsprüfung (6), Labelregeln (9) und der Modus je Linienmenge (10) lassen sich
auf dem Goldset nicht ablatieren: Es enthält weder Wände in Blöcken noch falsch deklarierte Einheiten noch
gemischte Zeichenstile, und seine Texte sind eindeutig. Dafür stehen die Tests mit synthetischen DXF.

## 6. Wo die Pipeline weiter schwach ist (Zahlen, dev, Endstand)

- **Wandlänge.** Median 7,0 %, P90 17,8 % daneben (held-out: 5,9 % und 31,9 %). Am schwächsten sind die
  Konturvarianten mit Öffnungslücken (`sib_kontur_unterbr`, `aia_kontur_unterbr`) mit negativer Abweichung:
  Wandstücke ohne Öffnungsblock werden nicht verbunden. In `sd_p00856` fehlt in einer Wand eine Lücke von
  1,38 m ohne Öffnungsblock in der Zeichnung; die Ground Truth hat dort ebenfalls keine Wand, das Fehlen
  ist also kein Fehler der Pipeline.
- **Raumerkennung.** Raum Recall 62,9 % auf dev (Precision 98,3 %), 72,5 % auf held-out. Von den
  nicht getroffenen Räumen hat auf dev (Auswertung mit Stand Schritt 08) die Mehrzahl kein Face (169 Soll-Räume); bei den Varianten mit Öffnungslücken sind die
  Ränder oft zu 75 bis 95 % durch Achsen gedeckt, aber nicht geschlossen. Eine zweite Gruppe sind zusammengefasste
  Räume, wenn eine dünne Trennwand fehlt.
- **Held-out.** Der Endstand liegt dort insgesamt nicht schlechter als auf dev. Auffällig ist `archicad` mit Wandlänge
  Median 13,6 % und P90 36,9 %; `archicad_kontur` liegt mit vier Dateien im Median 24,7 % (mit Vorzeichen plus 20,4 %)
  daneben. `sib_kontur_unterbr` hat auf held-out nur eine Datei (Wandlänge minus 28,4 %, kein Raum getroffen).
- **Raumfläche.** Median 0,4 %, P90 7,0 %. Das gilt nur für getroffene Räume (n = 399 von 634 Soll-Räumen auf dev).
- **archicad.** Stiftlayer wie STIFT_23 sind nicht über den Namen erkennbar. Die Wandlayer ergeben sich aus der
  Doppellinien-Geometrie. Dabei kommt gelegentlich ein Layer mit Geländern und Treppen hinzu (STIFT_9), und in
  `archicad_doppel_unterbr` liegt die Wandlänge im Mittel 6,2 % zu hoch, in einem Plan (`sd_p01421`)
  um 40,6 %. Die Pipeline weist die Wahl als Warnung aus; `wall_layers` setzt sie fest.
- **Fenster- und Türrecall.** Tür 95,3 %, Fenster 95,9 %. Fehltreffer konzentrieren sich in den
  Varianten mit Öffnungslücken (`aia_kontur_unterbr`: Tür 97 von 106, Fenster 78 von 87). Die Ursache wurde nicht
  einzeln untersucht.
- **Fremdachsen.** Zusätzliche Achsen außerhalb der Ground-Truth-Wände (Abstand über 0,2 m) liegen bei 0,5 bis
  2,2 % der Achslänge in den meisten Varianten, bei `archicad_doppel_unterbr` bei 9,0 % (nicht neu gemessen
  für den Endstand, Messung mit Schritt 08).
- **Laufzeit.** Nicht systematisch gemessen. Der Lauf über 54 Dateien dauert rund 100 Sekunden einschließlich
  IFC- und GLB-Export. Ein synthetischer Plan mit 4.800 Wandlinien braucht für die Rekonstruktion 1,0 Sekunden.
  Echte Zeichnungen mit mehr als 5.000 Wandlinien wurden nicht gemessen.

## 7. Was über echte Büro-Exporte weiterhin nicht belegt ist

- Die Layernamen der vier Konventionen sind aus öffentlichen Listen entnommen, die Zeichnungen daraus
  erzeugt. Wie sich echte Exporte aus Allplan, ArchiCAD und Vectorworks verhalten, ist nicht gemessen:
  Layerschemata, Blockaufbau (Türblöcke mit Attributen, Parameterobjekte), Schraffur, Bemaßung in
  Blöcken, Mehrfach-Layerwände (Putz, Dämmung, Tragschicht als eigene Linien) und Xrefs sind nicht
  abgedeckt.
- Wände mit mehreren Linien je Wandschicht (Putz, Dämmung, Tragschicht) sind nicht getestet. Erwartet sind mehrere Achsen mit zu kleiner Dicke.
- Gebogene Wände (ARC, Bögen in Polylinien) und Wände aus Flächen (SOLID, 3DFACE) werden nicht gelesen.
- DWG: Es gibt die Schnittstelle (`dwg_adapter`) und Tests mit einem Fake-Konverter. Ein echter ODA- oder
  LibreDWG-Aufruf wurde nicht ausgeführt, also ist auch nicht gemessen, wie die Pipeline mit echten
  DWG-Konvertierungen läuft.
- Die Zeichnungen enthalten kaum Büro- oder Gewerbebau, keine Bestandspläne mit Zeichenfehlern und keine
  Pläne mit offenen Räumen (`GROUND_TRUTH.md`, Abschnitte 0 und 8). Raumsuche ist im Goldset leichter als im
  Bestand.
- Wandhöhen, Türhöhen und Brüstungshöhen stehen nicht in den Zeichnungen und sind Konfigurationswerte
  (`assumptions` in `meta.json` und `scene.json`).
