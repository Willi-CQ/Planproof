import copy

from audit_agent_evidence import audit_result


def valid():
    return {"source": {"width": 200, "height": 100, "page_index": 0, "raster_sha256": "pixels"},
            "images": [{"image_id": "crop-1", "source_sha256": "pixels"}],
            "canonical_citations": [{"image_id": "crop-1", "page_index": 0, "canonical_bbox": [1, 2, 190, 90]}],
            "answer": {"status": "answered", "answer": 2}, "delegation": {"required": True}, "nemotron_calls": 1}


def test_evidence_audit_detects_source_identity_and_coordinate_corruption():
    result = valid()
    assert audit_result(result) == []
    result["images"][0]["source_sha256"] = "other"
    result["canonical_citations"][0]["canonical_bbox"] = [1, 2, 201, 90]
    assert audit_result(result) == ["citation_outside_source_or_empty", "image_source_identity_mismatch"]
    result["canonical_citations"][0]["canonical_bbox"] = [1, 2, float("nan"), 90]
    assert "invalid_canonical_bbox" in audit_result(result)


def test_evidence_audit_does_not_accept_missing_delegation_or_partial_total():
    result = valid()
    result["answer"]["count_basis"] = {"scope": "partial"}
    result["nemotron_calls"] = 0
    result["canonical_citations"] = []
    assert audit_result(result) == ["answered_with_partial_count_basis", "answered_without_canonical_citations",
                                    "answered_without_required_specialist_call"]
    abstention = copy.deepcopy(result)
    abstention["answer"] = {"status": "abstain", "answer": None}
    assert audit_result(abstention) == []
