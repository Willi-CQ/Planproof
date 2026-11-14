"""A portable, source-bound report of saved work. No model or tool execution."""

from datetime import datetime, timezone
from html import escape
from io import BytesIO
from math import ceil, floor, isfinite

import pymupdf
from PIL import Image

from run_tracking import box_valid

FRAME_KEYS = ("raster_sha256", "page_index", "width", "height")
STAGES = {
    "run": "Task",
    "gemini": "Drawing review and next decision",
    "get_algorithms": "Room and structural clues",
    "crop_region": "Closer source view",
    "crop_room": "Room detail",
    "inspect_nemotron": "Nemotron visual check",
    "inspect_regions": "Concurrent detail inspections",
    "review_answer": "Answer cross-check",
    "reflect": "Coverage and uncertainty check",
    "calculate_area": "Calculation from cited dimensions",
    "measure_polygon": "Boundary measurement",
    "answer": "Final briefing",
    "provider_attempt": "Service connection",
}
CSS = """
body { font-family: sans-serif; font-size: 9pt; line-height: 1.4; color: #303733; }
h1 { font-size: 25pt; font-weight: normal; margin: 5pt 0 9pt; color: #253d31; }
h2 { font-size: 15pt; font-weight: normal; margin: 17pt 0 8pt; page-break-after: avoid; }
h3 { font-size: 10pt; margin: 8pt 0 4pt; page-break-after: avoid; }
p { margin: 4pt 0 7pt; } a { color: #3f6950; text-decoration: none; }
.brand { font-size: 10pt; color: #3f6950; letter-spacing: 2pt; }
.meta { font-size: 8pt; color: #66716b; }
.question { font-size: 13pt; } .value { font-size: 23pt; color: #253d31; }
.note { padding: 7pt; background-color: #f0f3ef; font-size: 8pt; }
.source { width: 50%; vertical-align: top; padding: 0 8pt 6pt 0; }
.sources { width: 100%; border-collapse: collapse; }
.sources tr { page-break-inside: avoid; }
.step { border-top: 1px solid #e4e9e4; padding-top: 3pt; page-break-inside: avoid; font-size: 8pt; }
.step p { margin: 3pt 0 5pt; }
.reference { font-size: 8pt; }
"""


def text(value):
    return escape(str(value if value is not None else ""))


def same_frame(left, right):
    return bool(left and right and left.get("raster_sha256")) and all(
        left.get(key) == right.get(key) for key in FRAME_KEYS
    )


def concise(value, limit=240):
    value = " ".join(str(value or "").split())
    return value if len(value) <= limit else value[: limit - 1].rsplit(" ", 1)[0] + "…"


def activity_steps(events, terminal=True):
    """Merge call lifecycles, preserving parallel worker identity and failures."""
    steps = []
    for event in events:
        if event.get("stage") in {"run", "provider_attempt", "gemini", "answer"}:
            continue
        pending = next(
            (
                step
                for step in reversed(steps)
                if step.get("stage") == event.get("stage")
                and step.get("inspection_id") == event.get("inspection_id")
                and step.get("state") == "running"
            ),
            None,
        )
        if pending is not None and event.get("state") != "running":
            images = event.get("images") or pending.get("images", [])
            pending.update(event)
            pending["images"] = images
        else:
            steps.append(dict(event))
    if terminal:
        for step in steps:
            if step.get("state") == "running":
                step["state"] = "interrupted"
    return steps


def activity_html(run, source):
    events = run.get("events", [])
    steps = activity_steps(events)
    attempts = [e for e in events if e.get("stage") == "provider_attempt"]
    failures = [e for e in attempts if e.get("category")]
    reviews = sum(
        e.get("stage") == "gemini" and e.get("state") == "running" for e in events
    )
    times = [
        e["elapsed"]
        for e in events
        if isinstance(e.get("elapsed"), (int, float)) and isfinite(e["elapsed"])
    ]
    html = ["<h2>Activity</h2>"]
    if events:
        summary = [f"{len(steps)} tool steps", f"{reviews} agent reviews"]
        if times:
            summary.append(f"{max(times):.1f}s recorded elapsed")
        html.append(f'<p class="meta">{text(" · ".join(summary))}</p>')
        html.append(
            '<p class="meta">Compact activity summary. Start/return events are combined; long prompts and findings are shortened. Full detail remains in the saved task.</p>'
        )
    else:
        html.append("<p>No execution events were recorded.</p>")
    if attempts:
        categories = ", ".join(sorted({str(e["category"]) for e in failures}))
        html.append(
            f'<p class="meta">Service connection: {len(attempts)} attempts; {len(failures)} reported issues{": " + text(categories) if categories else "."}</p>'
        )
    for index, step in enumerate(steps, 1):
        advisory = step.get("advisory") or {}
        status = [step.get("state", "unknown")]
        status.extend(
            str(advisory[k]) for k in ("status", "verdict") if advisory.get(k)
        )
        elapsed = step.get("elapsed")
        if isinstance(elapsed, (int, float)) and isfinite(elapsed):
            status.append(f"+{elapsed:.1f}s")
        html.append(
            f'<div class="step"><p><b>{index}. {text(STAGES.get(step.get("stage"), "Recorded action"))}</b> · {text(" · ".join(status))}</p>'
        )
        question = step.get("question", "")
        if "COUNT_UNIT=" in question:
            question = (
                "Locate visible door leaves in the selected region."
                if "COUNT_UNIT=door_leaf" in question
                else "Locate visible door openings in the selected region."
            )
        elif "Focused question (data): " in question:
            question = question.split("Focused question (data): ", 1)[1]
        parts = [concise(question)] if question else []
        if step.get("classes"):
            parts.append("Requested clues: " + concise(", ".join(step["classes"])))
        for key in ("summary", "next_action"):
            if step.get(key):
                parts.append(concise(step[key]))
        for key in ("checks", "unresolved"):
            if step.get(key):
                parts.append(
                    key.capitalize() + ": " + concise("; ".join(step[key]), 350)
                )
        if advisory.get("answer") is not None:
            parts.append("Reported answer: " + concise(advisory["answer"]))
        findings = advisory.get("findings") or []
        if findings:
            parts.append("Findings: " + concise("; ".join(findings), 350))
        if parts:
            html.append(f"<p>{text(' · '.join(dict.fromkeys(parts)))}</p>")
        bounds = [
            ", ".join(str(round(v, 1)) for v in r["canonical_bbox"])
            for r in step.get("images", [])
            if r.get("page_index") == source["page_index"]
            and box_valid(r.get("canonical_bbox"), source)
        ]
        if bounds:
            html.append(
                f'<p class="meta">Source bounds: {text(" / ".join(dict.fromkeys(bounds)))}</p>'
            )
        html.append("</div>")
    return html


def task_report(run, png, question, filename):
    """Caller serializes PyMuPDF access. Only allowlisted public fields enter PDF."""
    source = run["source"]
    result = run.get("result") or {}
    answer = result.get("answer") or {}
    answered = (
        result.get("status") == "ok"
        and answer.get("status") == "answered"
        and answer.get("answer") is not None
    )
    citations = []
    # A mismatched result can never lend evidence to this source document.
    if same_frame(result.get("source"), source):
        for index, citation in enumerate(result.get("canonical_citations") or []):
            if citation.get("page_index") == source["page_index"] and box_valid(
                citation.get("canonical_bbox"), source
            ):
                citations.append(
                    {**citation, "number": citation.get("source_index", index + 1)}
                )
    numbers = {c["number"] for c in citations}
    details = {d["citation"]: d for d in answer.get("citation_details") or []}
    archive = pymupdf.Archive()

    def picture(image, name, max_width, max_height):
        image = image.convert("RGB")
        image.thumbnail((max_width * 2, max_height * 2))
        data = BytesIO()
        image.save(data, format="JPEG", quality=82, optimize=True)
        archive.add((data.getvalue(), name))
        scale = min(max_width / image.width, max_height / image.height)
        return f'<img src="{name}" width="{round(image.width * scale)}" height="{round(image.height * scale)}">'

    def refs(values):
        return " ".join(
            f'<a class="reference" href="#source-{int(n)}">[{int(n)}]</a>'
            for n in values
            if n in numbers
        )

    created = datetime.fromtimestamp(run["created"], timezone.utc).strftime(
        "%d %b %Y · %H:%M UTC"
    )
    html = [
        '<p class="brand">PLANPROOF</p><h1>Task report</h1>',
        f'<p class="meta">{text(filename)} · Page {source["page_index"] + 1}<br>{text(created)} · {text(run["status"])}</p>',
        f'<p class="question">{text(run.get("question") or question or "Question not recorded.")}</p>',
    ]
    if answered:
        value = f"{answer['answer']} {answer.get('unit') or ''}".strip()
        approximate = (result.get("measurement") or {}).get("approximate") is True
        html.append(f'<p class="value">{"≈ " if approximate else ""}{text(value)}</p>')
        if approximate:
            html.append(
                '<p class="meta">Estimate from the drawing; not an exact stated dimension.</p>'
            )
    else:
        html.append('<p class="value">Evidence incomplete</p>')
        html.append(
            f"<p>{text(run.get('message') or 'This task did not establish a verified answer.')}</p>"
        )
    if answer.get("briefing"):
        for paragraph in answer["briefing"]:
            html.append(
                f"<p>{text(paragraph['text'])} {refs(paragraph.get('citations', []))}</p>"
            )
    elif answer.get("explanation"):
        html.append(
            f"<p>{text(answer['explanation'])} {refs([c['number'] for c in citations])}</p>"
        )
    if answer.get("coverage"):
        html.append(f"<h3>Scope</h3><p>{text(answer['coverage'])}</p>")
    if answer.get("uncertainties"):
        html.append(
            "<h3>More information</h3><ul>"
            + "".join(f"<li>{text(u)}</li>" for u in answer["uncertainties"])
            + "</ul>"
        )
    html.append(
        '<p class="note">Recorded answer and source evidence. Algorithm and specialist observations are supporting clues, not independently verified facts.</p>'
    )
    with Image.open(BytesIO(png)) as image:
        html.append("<div><h2>Drawing &amp; cited evidence</h2>")
        html.append(picture(image, "overview.jpg", 470, 180))
        html.append(
            f'<p class="meta">Page {source["page_index"] + 1} · {source["width"]} × {source["height"]} source pixels. Preview scaled for this report; original unchanged.</p>'
        )
        if not citations:
            html.append(
                "<p>No source-bound visual citations were recorded for this answer.</p>"
            )
        html.append('<table class="sources">')
        for index, citation in enumerate(citations):
            if index % 2 == 0:
                html.append("<tr>")
            number = citation["number"]
            detail = details.get(number, {})
            box = citation["canonical_bbox"]
            crop_box = (floor(box[0]), floor(box[1]), ceil(box[2]), ceil(box[3]))
            html.append(
                f'<td class="source" id="source-{int(number)}"><h3>[{int(number)}] {text(detail.get("label", "Source detail"))}</h3>'
            )
            html.append(
                picture(image.crop(crop_box), f"source-{int(number)}.jpg", 215, 105)
            )
            html.append(
                f'<p class="meta">{text(detail.get("kind", "Evidence"))} · Bounds: {text(", ".join(str(round(v, 2)) for v in box))}</p></td>'
            )
            if index % 2 == 1 or index == len(citations) - 1:
                html.append("</tr>")
        html.append(
            '</table><p class="meta">Bounds are source pixels: left, top, right, bottom; origin at top left.</p></div>'
        )
    html.extend(activity_html(run, source))
    if run.get("telemetry_warning"):
        html.append(f'<p class="note">{text(run["telemetry_warning"])}</p>')
    html.append(
        f'<h3>Source identity</h3><p class="meta">Task: {text(run["request_id"])}<br>Raster SHA-256:<br>{text(source["raster_sha256"])}</p>'
    )
    story = pymupdf.Story(html="".join(html), user_css=CSS, archive=archive)
    page = pymupdf.paper_rect("a4")
    content = page + (48, 40, -48, -48)
    with story.write_with_links(lambda *_: (page, content, None)) as document:
        document.set_metadata({"title": "Planproof task report", "author": "Planproof"})
        for index, sheet in enumerate(document):
            sheet.insert_text(
                (48, page.height - 24),
                f"PLANPROOF   /   TASK REPORT                                              {index + 1} / {len(document)}",
                fontsize=8,
                color=(0.4, 0.45, 0.42),
            )
        return document.tobytes(garbage=4, deflate=True)
