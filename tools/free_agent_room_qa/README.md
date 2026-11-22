# Grounded room and door questions

This CPU reader answers one floor-plan question over one original PDF page or image. The output identifies the source by SHA-256, page number, displayed width and height, and coordinate system. Every evidence box uses top-left origin and `[left, top, right, bottom]`.

From the repository root, use Python with the packages listed in `models/sheet-doors-v1/requirements.txt`, Shapely 2.x, and the Tesseract executable with English language data for pixel text. Keep the bundled door reader and its weight file in their repository locations.

```sh
python -m tools.free_agent_room_qa floorplan.pdf "Which door directly connects CORRIDOR and BED2?" --page 1
python -m tools.free_agent_room_qa drawing.png "What floor area is printed for room R01, in m²?"
```

The German and English question grammar covers printed door identifiers, named-room adjacency and counts, project titles, printed room areas, documented fire-resistance ratings and recorded as-built clear widths. A ground-floor qualifier requires the corresponding visible page title.

For a door property, the reader looks for a labelled note naming one door or a table with an identifiable door column, property header and matching row. The answer cites the note or all three table cells. A request for a certified rating requires explicit certification wording. Widths require an as-built or field-measured clear-width field, and an explicit unit in the note, cell or column header. The result expresses widths in millimetres by default. English and German requests for centimetres or metres can select a different output unit. Conflicting rows, design-only dimensions and unsupported units return an unresolved result. This reads what the drawing records about a door.

An optional [bilingual research classifier](../../models/room-intent-v1/README.md) provides standalone operation suggestions for natural questions. The drawing CLI keeps its explicit grammar for automatic answers.

## Read once and ask several questions

```python
from tools.free_agent_room_qa import answer, answer_program, build_scene, compile_question

scene = build_scene("floorplan.pdf", page_number=1)
result = answer(scene, "Where is the door marked D6 on this sheet?")

program = compile_question("How many labeled doors directly connect to CORRIDOR?")
result = answer_program(scene, program)
```

Both the scene and the program serialize as ordinary JSON. A supported typed program contains `op` and `args`. For example, `{"op":"adjacent_doors","args":{"rooms":["CORRIDOR","BED2"]}}` selects labelled openings shared by both rooms. `arm="facts"` disables relationship answers and reads the same cached page facts. `arm="relations"` adds room-to-opening queries. A new scene is required after replacing a drawing or changing pages.

## Evidence and coverage

Native PDF extraction checks glyph visibility and keeps word locations. The vector reader looks for a physical opening supported by the wall on both sides, using a frame outline or compatible hinge, leaf and swing geometry. It binds a printed identifier only when it can distinguish the opening from competing candidates. Room polygons come from closed PDF paths or separated white components. For open rooms the reader can trace unobstructed rays from both sides of a verified frame to named interiors. The answer records the marks, geometry and available polygons or probe points.

On raster inputs the reader augments whole-page Tesseract with enlarged, orientation-aware text crops. It preserves positions in original image coordinates and merges repeated sightings of the same tag. An identified tag can support a location or a lower-bound count. Room-to-door answers additionally require physical evidence on both sides of the opening. Raw CNN proposals stay in `scene["suggestions"]`.

Use `status` when displaying or evaluating a result. `answered` is the reader's claim of a complete answer within the requested scope. `partial` carries positively observed items with a coverage reason. `unresolved` carries a missing-evidence or unsupported-question reason. The distinct mark count recognizes D, DR and DOOR conventions with numeric identifiers. An all-vector visible text layer is used as a coverage proxy. Unknown identifier conventions and outlined glyphs can defeat that proxy. OCR coverage and OCR spelling on scans can also fail, especially for small or rotated text. Physical room-contact inference remains sensitive to open boundaries and furniture strokes.
