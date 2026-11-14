#!/usr/bin/env python3
"""Smoke test for Deterministic Geometry & Quantity Extraction Engine.

Tests against real dataset samples from ~/Downloads/junction-dataset-preview.
Enforces 20-second timeout per file.

Usage:
    python scripts/smoke_test_deterministic.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add geometry to path
GEOMETRY_DIR = Path(__file__).parent.parent / "geometry"
sys.path.insert(0, str(GEOMETRY_DIR))

from deterministic_engine import extract_deterministic, resolve_dataset_root


# Dataset root resolution
DATASET_ROOT = resolve_dataset_root()
if DATASET_ROOT is None:
    print("ERROR: No dataset root found. Set STOA_DATASET_DIR or ensure dataset is mounted.")
    sys.exit(1)

# Dataset data subdirectory (where actual files are)
DATASET_DATA = DATASET_ROOT / "data" if (DATASET_ROOT / "data").exists() else DATASET_ROOT

# Test cases: (description, file_path, expected_engine, expected_valid)
TEST_CASES = [
    (
        "IFC architectural model",
        DATASET_DATA / "ifc_models" / "bsc-esplanades_1807-ep-ar-v18.ifc",
        "ifc_bim",
        True
    ),
    (
        "Swiss dwelling PNG with DXF sibling",
        DATASET_DATA / "swiss_dwellings" / "site936_bldg1976_floor8706.png",
        "cad_dxf",
        True
    ),
    (
        "Cubicasa5K Finnish floorplan",
        DATASET_DATA / "cubicasa5k" / "colorful-11397.png",
        "raster_cv",
        True
    ),
    (
        "Finna faded scan",
        DATASET_DATA / "finna" / "hka.139877703731200_160449953667200.jpg",
        "raster_cv",
        True
    ),
    (
        "Europeana sheet music (non-floorplan)",
        DATASET_DATA / "europeana" / "401_item_QWKGCOX6OGLOOF7QPOTPOQC5VBPJBIQN.jpg",
        "raster_cv",
        False  # Should detect as non-floorplan
    ),
]


def run_test(description: str, file_path: Path, expected_engine: str, expected_valid: bool) -> Dict[str, Any]:
    """Run a single test case with timeout."""
    result = {
        "description": description,
        "file_path": str(file_path),
        "exists": file_path.exists(),
        "passed": False,
        "engine_used": None,
        "is_valid_floorplan": None,
        "duration_seconds": 0.0,
        "error": None,
        "details": {}
    }
    
    if not file_path.exists():
        result["error"] = "File does not exist"
        return result
    
    start_time = time.time()
    
    try:
        # Run extraction with timeout (20 seconds)
        extraction_result = extract_deterministic(str(file_path))
        
        duration = time.time() - start_time
        result["duration_seconds"] = round(duration, 2)
        result["engine_used"] = extraction_result.get("engine_used")
        result["is_valid_floorplan"] = extraction_result.get("is_valid_floorplan")
        
        # Check if timeout exceeded
        if duration > 20.0:
            result["error"] = f"Timeout: {duration:.2f}s > 20s"
            return result
        
        # Validate results
        engine_match = extraction_result.get("engine_used") == expected_engine
        valid_match = extraction_result.get("is_valid_floorplan") == expected_valid
        ok_match = extraction_result.get("ok") == True
        
        result["passed"] = engine_match and valid_match and ok_match
        
        # Add details
        result["details"] = {
            "walls_amount": extraction_result.get("walls", {}).get("amount"),
            "rooms_amount": extraction_result.get("rooms", {}).get("amount"),
            "doors_amount": extraction_result.get("openings", {}).get("doors", {}).get("amount"),
            "windows_amount": extraction_result.get("openings", {}).get("windows", {}).get("amount"),
            "warnings": extraction_result.get("warnings", [])[:3],  # First 3 warnings
            "scale_confidence": extraction_result.get("scale_confidence")
        }
        
        if not result["passed"]:
            result["error"] = f"Mismatch: engine={extraction_result.get('engine_used')} (expected {expected_engine}), valid={extraction_result.get('is_valid_floorplan')} (expected {expected_valid}), ok={extraction_result.get('ok')}"
        
    except Exception as e:
        duration = time.time() - start_time
        result["duration_seconds"] = round(duration, 2)
        result["error"] = f"Exception: {str(e)}"
    
    return result


def find_actual_files():
    """Find actual test files in the dataset."""
    global TEST_CASES
    
    # Update test cases with actual files if placeholders don't exist
    updated_cases = []
    
    for desc, path, engine, valid in TEST_CASES:
        if not path.exists():
            # Try to find similar files
            if "cubicasa5k" in str(path):
                # Find any PNG in cubicasa5k
                cubicasa_dir = DATASET_DATA / "cubicasa5k"
                if cubicasa_dir.exists():
                    pngs = list(cubicasa_dir.glob("*.png"))[:1]
                    if pngs:
                        path = pngs[0]
            elif "finna" in str(path):
                # Find any JPG in finna
                finna_dir = DATASET_DATA / "finna"
                if finna_dir.exists():
                    jpgs = list(finna_dir.glob("*.jpg"))[:1]
                    if jpgs:
                        path = jpgs[0]
            elif "europeana" in str(path):
                # Find any JPG in europeana
                europeana_dir = DATASET_DATA / "europeana"
                if europeana_dir.exists():
                    jpgs = list(europeana_dir.glob("*.jpg"))[:1]
                    if jpgs:
                        path = jpgs[0]
        
        updated_cases.append((desc, path, engine, valid))
    
    TEST_CASES = updated_cases


def main():
    print("=" * 80)
    print("Planproof Deterministic Engine - Smoke Test")
    print("=" * 80)
    print()
    
    # Check if dataset exists
    if not DATASET_DATA.exists():
        print(f"ERROR: Dataset data not found at {DATASET_DATA}")
        print("Please ensure dataset is mounted and configured correctly.")
        sys.exit(1)
    
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Dataset data: {DATASET_DATA}")
    print()
    
    # Find actual test files
    find_actual_files()
    
    # Run tests
    results = []
    for desc, path, engine, valid in TEST_CASES:
        print(f"Testing: {desc}")
        print(f"  File: {path.name}")
        print(f"  Expected: engine={engine}, valid={valid}")
        
        result = run_test(desc, path, engine, valid)
        results.append(result)
        
        if result["passed"]:
            print(f"  ✓ PASSED ({result['duration_seconds']:.2f}s)")
            print(f"    Engine: {result['engine_used']}")
            print(f"    Valid: {result['is_valid_floorplan']}")
            if result["details"]:
                print(f"    Walls: {result['details'].get('walls_amount', 0)}")
                print(f"    Rooms: {result['details'].get('rooms_amount', 0)}")
                print(f"    Doors: {result['details'].get('doors_amount', 0)}")
                print(f"    Windows: {result['details'].get('windows_amount', 0)}")
        else:
            print(f"  ✗ FAILED")
            if result["error"]:
                print(f"    Error: {result['error']}")
            if result["details"]:
                print(f"    Warnings: {result['details'].get('warnings', [])}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print()
    
    for result in results:
        status = "✓" if result["passed"] else "✗"
        print(f"{status} {result['description']}")
        if result["error"]:
            print(f"  Error: {result['error']}")
    
    print()
    
    # Write detailed results to /tmp (not repo root)
    output_file = Path("/tmp/smoke_test_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": time.time(),
            "dataset_root": str(DATASET_ROOT),
            "dataset_data": str(DATASET_DATA),
            "total_tests": total,
            "passed_tests": passed,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed results written to: {output_file}")
    print()
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
