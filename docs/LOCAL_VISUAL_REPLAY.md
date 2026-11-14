# Offline visual replay

`tools/build_visual_replay.mjs` creates a single HTML file from a saved public run
record and its unchanged PNG. Open the resulting file directly in a browser; it
needs neither the backend nor an internet connection. No app history is written.

The viewer imports the actual PlanCanvas2D, RunActivity, AnswerBriefing and
EvidenceInspector components. It adds clearly labelled replay controls: pause,
previous/next event, event slider, example selection, tool-stage jumps and playback
pace. Guided playback presents one recorded event every two seconds; the original
event times and measured durations remain unchanged. Original-timing modes are
also available. This is a historical UI review, not a new model-accuracy test.

Build from the repository root after installing frontend dependencies:

```text
node tools/build_visual_replay.mjs .artifacts/replay/manifest.json .artifacts/replay/Planproof-All-Tools.html
```

The private manifest has this structure (paths resolve from the current directory):

```json
{
  "image": ".artifacts/replay/original.png",
  "sourceRevision": "revision used to build the UI",
  "examples": [
    {
      "record": ".artifacts/replay/saved-run.json",
      "title": "Recorded inspection",
      "note": "Historical execution; original limitations remain visible."
    }
  ]
}
```

All examples must refer to the exact same original PNG. The builder checks its
SHA256 and dimensions against each record. It copies the original bytes without
resizing and projects public event, answer and evidence fields; raw provider
payloads, local crop paths and internal reasoning are excluded. Questions and
findings render as text. Embedded JSON escapes HTML delimiters.

The generated file embeds its image, evidence, scripts and styles. Its Content
Security Policy blocks network connections and external resources. Output must
stay under `.artifacts/`; private drawings, manifests and replays are not source
assets and must not be published with the repository. The builder also writes a
small local build receipt. A fresh build captures the current working-tree UI.

At each playback position only the recorded prefix is shown. Final answers wait
until the end; candidate exploration becomes available after the recorded
algorithm result. Rewinding clears selected evidence and layers. Source hashes
still guard all overlays. Earlier incomplete or inaccurate model findings are
preserved, not silently corrected to make the replay look successful.

Visual review exposed ambiguous labels in older multi-image Nemotron calls. Shared
UI labels now include a view number when a call has several crops, and previews
show the recorded crop ID so references such as `crop-2` can be matched directly.

Validation: two portable synthetic browser/builder tests cover offline execution,
playback, crop identity, null timing, candidate inspection, source mismatch and
private-field exclusion. Four shared activity regressions pass. The local real
review separately stepped through 33- and 63-event records with their native
10200x6600 PNG, inspected desktop/390px/320px screenshots, exercised layers and
refresh, and recorded zero browser errors or external requests. Real records and
screenshots remain local. TypeScript and both replay/product builds pass.
