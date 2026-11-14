"""Compact, optionally crop-scoped evidence for later agent integration."""

from shapely.geometry import box


def agent_context(document, page_index=None, bbox=None):
    pages = [
        p for p in document["pages"] if page_index is None or p["frame"]["page_index"] == page_index
    ]
    indices = {p["frame"]["page_index"] for p in pages}
    region = box(*bbox) if bbox is not None else None
    observations = []
    for o in document["observations"]:
        if o["page_index"] not in indices or not (o.get("class") or o["kind"] == "text_span"):
            continue
        if region is not None and not region.intersects(box(*o["bbox"])):
            continue
        observations.append(
            {
                key: o[key]
                for key in (
                    "id",
                    "page_index",
                    "class",
                    "bbox",
                    "source_family",
                    "mode",
                    "method",
                    "score",
                    "room_type",
                    "text",
                    "dimension",
                    "quantities",
                    "warnings",
                    "dependency_ids",
                )
                if key in o
            }
        )
    ids = {o["id"] for o in observations}
    return {
        "schema_version": "planproof-agent-context-v1",
        "document_id": document["document_id"],
        "scope": {"page_index": page_index, "bbox": bbox},
        "policy": [
            "Candidates guide verification; agreement is not proof.",
            "Hybrid descendants cannot independently confirm their ML parents.",
            "Missing detections are not absence; no physical count or metre scale is assumed.",
            "Page aggregates are not recomputed as crop quantities.",
        ],
        "pages": [
            {"frame": p["frame"], "availability": p["availability"], "warnings": p["warnings"]}
            for p in pages
        ],
        "observations": observations,
        "comparisons": [
            c for c in document["comparisons"] if any(oid in ids for oid in c["source_ids"])
        ],
        "page_aggregates": [a for a in document["aggregates"] if a["page_index"] in indices],
        "calibrations": [c for c in document["calibrations"] if c["page_index"] in indices],
    }
