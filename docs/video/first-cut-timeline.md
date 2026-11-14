# PlanProof product video: first-cut timeline

Status: **editor-ready timeline, no playable cut**. No video editor is available in this environment and no recordings or screenshots were supplied, so every product shot below is a placeholder the team must record.

- Format: 16:9, 1920×1080, 30 fps, English voice-over, English subtitles.
- Final runtime: **1:50.0** (cap 2:00).
- Subtitles: [planproof-first-cut.srt](planproof-first-cut.srt), 18 cues, timed to this timeline.
- Voice: calm, confident, about 2.3 words per second. Total 140 words.

## Source material

| Clip name | Content | State |
| --- | --- | --- |
| `LOGO_stoa.png` | `plan-checker/stoa-icon-512.png` from the repository | Available |
| `SHOT_01_two_counts` | Real drawing, tight crop on a door symbol | **To record** |
| `FOOT_01_manual_count` | Hands marking door symbols on a printed plan | **To record**, or generated footage |
| `SHOT_02_discrepancy` | Test screenshot showing 3 geometry candidates, 10 ML candidates, 0 matched pairs | **Missing. See the rule under 0:22** |
| `REC_01_upload` | Stoa screen: import a real PDF or PNG until the page renders | **To record** |
| `REC_02_question` | Typing "How many doors are shown on this plan?" and submitting | **To record** |
| `REC_03_evidence` | Analysis finishing, evidence markers appearing on the plan | **To record** |
| `REC_04_nemotron_crop` | The crop sent for review and the returned advisory | **To record**, only if it ran |
| `REC_05_sona_answer` | Sona's written response with citations | **To record** |
| `REC_06_citation_click` | Click on one citation, workspace moves to the region | **To record** |
| `REC_07_needs_review` | A result where Sona says what still needs checking | **To record**, only if it occurs |
| `MUSIC_bed` | Low, steady instrumental | To choose |

Record all `REC_` clips in one continuous session on the same drawing so the question, result and citation click stay in order. Capture at 1920×1080, browser zoom 110–125%, cursor visible.

## Timeline

| Time | Picture | On-screen text | Voice-over |
| --- | --- | --- | --- |
| 0:00–0:07 | `SHOT_01_two_counts`. Slow push-in on one door symbol. At 0:02 two count chips appear left and right of the symbol, each labelled "Estimate A" and "Estimate B". | One drawing. Two counts. Which one do you price? | "One drawing. Two counts. Which one do you price?" (0:01.0–0:05.5, then a 1.5 s beat) |
| 0:07–0:22 | `FOOT_01_manual_count`. Close-up of hands ticking door symbols one by one; cut at 0:13 to a wider view of the full sheet; at 0:17 a second pass over the same area. | None | "Before a builder can price a job, someone has to turn drawings into quantities. When two readings disagree, the work starts all over again. Which symbols were counted, and where are they on the plan?" (0:07.5–0:21.5) |
| 0:22–0:35 | `SHOT_02_discrepancy`. Hold the screenshot; reveal one label at a time at 0:23.5, 0:27.0 and 0:29.0. Footnote from 0:23.5 to 0:35. | "3 candidates from geometry rules" · "10 candidates from the ML detector" · "0 matched pairs" · footnote: "Detector proposals from one test. Verified door total unknown." | "On this test drawing, one detector proposed three candidates. The other proposed ten. None of their proposals matched. The team needed to see where the disagreement came from." (0:22.5–0:34.7) |
| 0:35–0:43 | `REC_01_upload`, then `REC_02_question`. Cut the render wait. Punch in on the question field while it is typed. | Lower third: "Ask Sona" | "Upload a drawing and ask Sona a question." (0:35.5–0:38.5) |
| 0:43–0:53 | `REC_03_evidence`. Cut idle time; keep the moment the markers appear on the plan. | Callouts: "Geometry rules" · "Machine learning" | "PlanProof can use geometry rules and machine learning to find evidence." (0:44.0–0:49.0) |
| 0:53–1:01 | `REC_04_nemotron_crop`. Punch in on the crop and the advisory. | Tag: "Optional: crop review with NVIDIA Nemotron" | "When needed, NVIDIA Nemotron can inspect a specific crop." (0:53.5–0:57.5) |
| 1:01–1:08 | `REC_05_sona_answer`. Full answer visible, citations readable. | None | "Gemini writes Sona's final response." (1:01.5–1:04.5) |
| 1:08–1:24 | `REC_06_citation_click`. Cursor to the citation, click at 1:13.5, 1 s beat with no voice, then split view (answer left, plan right) and punch in on the highlighted region. Hold 1:16–1:24. | Thin connector line from the citation to the region | "Click a citation and the workspace takes you to the exact area on the drawing." (1:08.5–1:13.5) |
| 1:24–1:39 | `REC_07_needs_review`. The result panel in its review state, then the regions Sona lists. | None | "If the evidence is inconclusive, Sona tells you what still needs checking." (1:25.0–1:30.0) |
| 1:39–1:50 | Return to the cited region from `REC_06` (1:39–1:44), dissolve to the end card (1:44–1:50). | End card: "PlanProof" / "Built for the Stoa Build challenge at Junction 2026" | "Find the quantity. Follow the evidence. Try PlanProof." (1:40.0–1:47.0) |

## Conditional rules from the brief

**0:22–0:35, the figures.** The 3 / 10 / 0 figures are not recorded anywhere in the repository, so they are unverified here. Use them only with the team's own screenshot of that test. Without it, cut this version:

- Picture: the same drawing as the opening, two marker layers in different colours that do not overlap.
- On-screen text: "Two detectors. Different proposals." No figures.
- Voice-over: "On one test drawing, our two detectors disagreed. Their proposals did not line up. The team needed to see where the disagreement came from."
- Subtitles: replace cues 6 to 9 with the two cues listed at the end of this document.

**0:53–1:01, NVIDIA Nemotron.** Keep the "Optional" tag unless the recording or its trace shows the review ran on this exact drawing. If it did not run, use the crop region from the plan as picture and keep the same voice-over line, which is phrased as a capability.

**1:24–1:39, uncertainty.** Show this state only from a real recording. If none exists, drop the shot, extend the hold on the highlighted region to 1:30, and move the voice-over line over that hold. The runtime becomes 1:41.

**0:35–1:39, the question.** If the recording uses another question with solid citations, keep that exact question and its real result, and leave the voice-over unchanged.

## Shots the team must record

1. A real drawing the team may show publicly, with a clear door symbol (`SHOT_01`).
2. Manual counting on a printed plan, 15 s usable (`FOOT_01`), or approve generated footage for this section only.
3. The discrepancy screenshot with its three figures visible (`SHOT_02`).
4. One continuous Stoa session: import, question, evidence, Sona's answer, citation click (`REC_01` to `REC_06`).
5. A result in its review state (`REC_07`), if one can be produced honestly.

The Stoa interface in the repository is in French ("Analyser le plan", "Vérifier les portes sur plan"). For an English video, either record with English labels or add English callouts over the French buttons that the viewer needs to read.

## Editing notes

- Generated footage only in 0:07–0:22. Every product interaction and result is a real capture.
- Use the drawing's own linework and the app's marker colours for callouts and connector lines.
- Music stays under the voice (about −24 dB while speaking, −16 dB in gaps). Leave silence in the voice track after the opening question (0:05.5–0:07.0) and after the citation click (1:13.5–1:14.5).
- Blur before export: the sign-in screen and any password field, API keys, account names and avatars, the browser address bar if it shows a private URL, project identifiers in the drawing's title block, and file paths in upload dialogs.

## Alternative subtitles for 0:22–0:35 without figures

```
6
00:00:22,500 --> 00:00:27,500
On one test drawing, our two detectors
disagreed. Their proposals did not line up.

7
00:00:28,500 --> 00:00:33,500
The team needed to see where
the disagreement came from.
```

Renumber the following cues from 8 if this version is used.
