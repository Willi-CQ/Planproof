"""A portable, source-bound report of saved work. No model or tool execution."""

from datetime import datetime, timezone
from html import escape
from io import BytesIO
from math import ceil, floor

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
body { font-family: sans-serif; font-size: 10pt; line-height: 1.5; color: #303733; }
h1 { font-size: 30pt; font-weight: normal; margin: 8pt 0 14pt; color: #253d31; }
h2 { font-size: 17pt; font-weight: normal; margin: 22pt 0 10pt; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 5pt; page-break-after: avoid; }
p { margin: 5pt 0 9pt; } a { color: #3f6950; text-decoration: none; }
.brand { font-size: 10pt; color: #3f6950; letter-spacing: 2pt; }
.meta { font-size: 8pt; color: #66716b; }
.question { font-size: 15pt; } .value { font-size: 24pt; color: #253d31; }
.note { padding: 9pt; background-color: #f0f3ef; }
.source { page-break-inside: avoid; margin: 14pt 0; }
.step { border-top: 1px solid #e4e9e4; padding-top: 6pt; }
.reference { font-size: 8pt; } .section { page-break-before: always; }
"""


def text(value):
    return escape(str(value if value is not None else ""))


def same_frame(left, right):
    return bool(left and right and left.get("raster_sha256")) and all(
        left.get(key) == right.get(key) for key in FRAME_KEYS
    )


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
        image.save(data, format="JPEG", quality=88)
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
            "<h3>Verification notes</h3><ul>"
            + "".join(f"<li>{text(u)}</li>" for u in answer["uncertainties"])
            + "</ul>"
        )
    html.append(
        '<p class="note">This report records the returned answer and its evidence. Algorithm candidates and specialist observations are supporting clues, not independent proof. Verify the cited drawing details before relying on the answer.</p>'
    )
    with Image.open(BytesIO(png)) as image:
        html.append('<div class="section"><h2>Drawing &amp; cited evidence</h2>')
        html.append(picture(image, "overview.jpg", 470, 280))
        html.append(
            f'<p class="meta">Page {source["page_index"] + 1} · {source["width"]} × {source["height"]} source pixels. Preview scaled for this report; original unchanged.</p>'
        )
        if not citations:
            html.append(
                "<p>No source-bound visual citations were recorded for this answer.</p>"
            )
        for citation in citations:
            number = citation["number"]
            detail = details.get(number, {})
            box = citation["canonical_bbox"]
            crop_box = (floor(box[0]), floor(box[1]), ceil(box[2]), ceil(box[3]))
            html.append(
                f'<div class="source" id="source-{int(number)}"><h3>[{int(number)}] {text(detail.get("label", "Source detail"))}</h3>'
            )
            html.append(
                picture(image.crop(crop_box), f"source-{int(number)}.jpg", 450, 205)
            )
            html.append(
                f'<p class="meta">{text(detail.get("kind", "Evidence"))} · Page {source["page_index"] + 1} · Bounds: {text(", ".join(str(round(v, 2)) for v in box))}<br>Source pixels: left, top, right, bottom; origin at top left.</p></div>'
            )
        html.append("</div>")
    html.append(
        '<div class="section"><h2>Recorded work</h2><p class="meta">Execution events in order. These are recorded actions and returned observations, not private model reasoning.</p>'
    )
    for event in run.get("events", []):
        elapsed = float(event.get("elapsed", 0))
        html.append(
            f'<div class="step"><h3>{text(STAGES.get(event["stage"], "Recorded action"))}</h3><p class="meta">{text(event["state"])} · +{elapsed:.1f}s</p>'
        )
        for key, label in (
            ("summary", ""),
            ("question", "Inspection question"),
            ("next_action", "Next action"),
        ):
            if event.get(key):
                html.append(
                    f"<p>{f'<b>{label}:</b> ' if label else ''}{text(event[key])}</p>"
                )
        for key, label in (
            ("classes", "Requested clues"),
            ("checks", "Checks"),
            ("unresolved", "Unresolved"),
        ):
            if event.get(key):
                html.append(f"<p><b>{label}:</b> {text('; '.join(event[key]))}</p>")
        advisory = event.get("advisory") or {}
        for key in ("status", "answer", "verdict", "findings"):
            if advisory.get(key) is not None:
                value = advisory[key]
                html.append(
                    f"<p><b>Specialist {key}:</b> {text('; '.join(value) if isinstance(value, list) else value)}</p>"
                )
        for region in event.get("images", []):
            if region.get("page_index") == source["page_index"] and box_valid(
                region.get("canonical_bbox"), source
            ):
                html.append(
                    f'<p class="meta">Inspected source bounds: {text(region["canonical_bbox"])}</p>'
                )
        html.append("</div>")
    if not run.get("events"):
        html.append("<p>No execution events were recorded.</p>")
    if run.get("telemetry_warning"):
        html.append(f'<p class="note">{text(run["telemetry_warning"])}</p>')
    html.append(
        f'<h3>Source identity</h3><p class="meta">Task: {text(run["request_id"])}<br>Raster SHA-256:<br>{text(source["raster_sha256"])}</p></div>'
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
