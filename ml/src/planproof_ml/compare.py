"""Validation-only experiment selection on identical source cohorts."""

from .schema import read_json


def compare(baseline_path, candidate_path):
    baseline, candidate = read_json(baseline_path), read_json(candidate_path)
    for report in (baseline, candidate):
        if report.get("split") != "val":
            raise ValueError("Experiment selection requires validation reports, never test")
        if not report.get("cohort_fingerprints"):
            raise ValueError("Regenerate reports with evaluation cohort fingerprints")
    for field in ("protocol", "image_size", "tiled"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"Different evaluation protocol: {field}")
    sources = sorted(set(baseline["sources"]) & set(candidate["sources"]))
    if not sources:
        raise ValueError("No common validation sources")
    for source in sources:
        if baseline["cohort_fingerprints"][source] != candidate["cohort_fingerprints"][source]:
            raise ValueError(f"Different validation cohort: {source}")
    weights = {"wall_iou": 0.4, "object_map50": 0.3, "room_instance_f1_iou50": 0.3}
    scores = [0.0, 0.0]
    components = {}
    regressions = []
    for metric, weight in weights.items():
        eligible = [
            source
            for source in sources
            if all(r["sources"][source].get(metric) is not None for r in (baseline, candidate))
        ]
        if not eligible:
            raise ValueError(f"No comparable validation support for {metric}")
        values = [
            sum(r["sources"][s][metric] for s in eligible) / len(eligible) for r in (baseline, candidate)
        ]
        for idx in (0, 1):
            scores[idx] += weight * values[idx]
        components[metric] = {"sources": eligible, "baseline": values[0], "candidate": values[1]}
        for source in eligible:
            delta = candidate["sources"][source][metric] - baseline["sources"][source][metric]
            if delta < -0.02:
                regressions.append({"source": source, "metric": metric, "delta": delta})
    return {
        "split": "val",
        "common_sources": sources,
        "baseline_score": scores[0],
        "candidate_score": scores[1],
        "improvement": scores[1] - scores[0],
        "components": components,
        "regressions_over_002": regressions,
        "select_candidate": scores[1] - scores[0] > 0.005 and not regressions,
        "rule": "Improvement >0.005; no common-source component drops by >0.02. Test remains held out.",
    }
