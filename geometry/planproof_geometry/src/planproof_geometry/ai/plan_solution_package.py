"""
Solution Package: Structured output format for PDF-to-BIM pipeline.

A solution package contains all outputs from the pipeline:
- model.gltf: 3D geometry for the viewer
- extraction.json: Raw extraction data (lines, texts)
- semantics.json: Semantic classifications
- measurements.json: Detected measurements with scale info
- preview.png: Thumbnail for UI
- meta.json: Pipeline metadata, quality metrics
"""
from __future__ import annotations

import json
import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PackageManifest:
    """Manifest describing contents of a solution package."""
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    pipeline_version: str = "0.1.0"
    core_version: str = "0.1.0"
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    
    # Input info
    source_file: str = ""
    source_file_hash: Optional[str] = None
    
    # Output files
    model_file: str = "model.gltf"
    extraction_file: str = "extraction.json"
    semantics_file: str = "semantics.json"
    measurements_file: str = "measurements.json"
    preview_file: str = "preview.png"
    ifc_file: str = "model.ifc"
    
    # Quality metrics
    quality_score: float = 0.0
    quality_label: str = "unknown"  # high, medium, low
    
    # Pipeline stats
    steps_completed: List[str] = field(default_factory=list)
    processing_time_s: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "pipeline_version": self.pipeline_version,
            "core_version": self.core_version,
            "pipeline_config": self.pipeline_config,
            "source": {
                "file": self.source_file,
                "hash": self.source_file_hash,
            },
            "files": {
                "model": self.model_file,
                "extraction": self.extraction_file,
                "semantics": self.semantics_file,
                "measurements": self.measurements_file,
                "preview": self.preview_file,
                "ifc": self.ifc_file,
            },
            "quality": {
                "score": self.quality_score,
                "label": self.quality_label,
            },
            "pipeline": {
                "steps_completed": self.steps_completed,
                "processing_time_s": self.processing_time_s,
                "warnings": self.warnings,
                "errors": self.errors,
            }
        }


@dataclass
class MeasurementData:
    """Extracted measurements with scale information."""
    scale_found: bool = False
    scale_ratio: Optional[float] = None  # e.g., 1:100
    scale_confidence: float = 0.0
    points_to_m: float = 1.0  # Conversion factor
    
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    # Each measurement: {id, value_m, text, bbox, confidence, anchor_line_ids}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "scale": {
                "found": self.scale_found,
                "ratio": self.scale_ratio,
                "confidence": self.scale_confidence,
                "points_to_m": self.points_to_m,
            },
            "measurements": self.measurements,
            "count": len(self.measurements),
        }


class SolutionPackageBuilder:
    """
    Builder for creating solution packages from pipeline outputs.
    
    Usage:
        builder = SolutionPackageBuilder(output_dir="/path/to/output")
        builder.set_source("input.pdf")
        builder.add_model(gltf_path)
        builder.add_extraction(extraction_dict)
        builder.add_semantics(semantics_result)
        builder.add_measurements(measurements)
        builder.add_preview(image_array)
        package_path = builder.build()
    """
    
    def __init__(self, output_dir: str, package_name: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.package_name = package_name or f"solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.package_dir = self.output_dir / self.package_name
        self.package_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest = PackageManifest()
        self._files: Dict[str, Path] = {}
        self._start_time = datetime.now()
    
    def set_source(self, source_path: str, compute_hash: bool = True) -> "SolutionPackageBuilder":
        """Set the source file information."""
        self.manifest.source_file = str(Path(source_path).name)
        
        if compute_hash:
            import hashlib
            try:
                with open(source_path, "rb") as f:
                    self.manifest.source_file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception as e:
                logger.warning(f"Could not compute source hash: {e}")
        
        return self
    
    def add_model(self, gltf_path: str) -> "SolutionPackageBuilder":
        """Add the 3D model file."""
        src = Path(gltf_path)
        if not src.exists():
            self.manifest.errors.append(f"Model file not found: {gltf_path}")
            return self

        dst = self.package_dir / self.manifest.model_file
        shutil.copy2(src, dst)
        self._files["model"] = dst
        self.manifest.steps_completed.append("model_export")

        return self

    def add_ifc(self, ifc_path: str) -> "SolutionPackageBuilder":
        """Add the IFC4 model file (production-ready BIM-Ausgabe)."""
        src = Path(ifc_path)
        if not src.exists():
            self.manifest.errors.append(f"IFC file not found: {ifc_path}")
            return self

        dst = self.package_dir / self.manifest.ifc_file
        shutil.copy2(src, dst)
        self._files["ifc"] = dst
        self.manifest.steps_completed.append("ifc_export")
        return self
    
    def add_extraction(self, extraction_dict: Dict[str, Any]) -> "SolutionPackageBuilder":
        """Add raw extraction data."""
        dst = self.package_dir / self.manifest.extraction_file
        dst.write_text(json.dumps(extraction_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        self._files["extraction"] = dst
        self.manifest.steps_completed.append("extraction")
        
        # Extract quality info
        doc = extraction_dict.get("doc", {})
        quality = doc.get("quality", "unknown")
        self.manifest.quality_label = quality
        
        return self
    
    def add_semantics(self, semantics_dict: Dict[str, Any]) -> "SolutionPackageBuilder":
        """Add semantic classification results."""
        dst = self.package_dir / self.manifest.semantics_file
        dst.write_text(json.dumps(semantics_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        self._files["semantics"] = dst
        self.manifest.steps_completed.append("semantic_classification")
        
        # Calculate quality score based on detection confidence
        elements = semantics_dict.get("elements", [])
        if elements:
            avg_conf = sum(e.get("confidence", 0) for e in elements) / len(elements)
            self.manifest.quality_score = max(self.manifest.quality_score, avg_conf)
        
        return self
    
    def add_measurements(self, measurements: MeasurementData) -> "SolutionPackageBuilder":
        """Add measurement data."""
        dst = self.package_dir / self.manifest.measurements_file
        dst.write_text(json.dumps(measurements.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        self._files["measurements"] = dst
        self.manifest.steps_completed.append("measurements")
        
        return self
    
    def add_measurements_from_extraction(self, extraction_dict: Dict[str, Any]) -> "SolutionPackageBuilder":
        """Extract measurements from extraction dict and add to package."""
        doc = extraction_dict.get("doc", {})
        scale_info = doc.get("scale", {})
        transform = doc.get("transform", {})
        
        measurements = MeasurementData(
            scale_found=scale_info.get("found", False),
            scale_ratio=scale_info.get("ratio"),
            scale_confidence=scale_info.get("confidence", 0.0),
            points_to_m=transform.get("points_to_m", 1.0),
            measurements=extraction_dict.get("measurements", []),
        )
        
        return self.add_measurements(measurements)
    
    def add_preview(
        self,
        image: Any,  # numpy array or PIL Image
        max_size: int = 512,
    ) -> "SolutionPackageBuilder":
        """Add preview thumbnail."""
        try:
            import numpy as np
            from PIL import Image
            
            # Convert numpy to PIL if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    # Grayscale
                    pil_img = Image.fromarray(image, mode="L")
                elif image.shape[2] == 3:
                    # BGR to RGB
                    pil_img = Image.fromarray(image[:, :, ::-1], mode="RGB")
                else:
                    pil_img = Image.fromarray(image)
            else:
                pil_img = image
            
            # Resize maintaining aspect ratio
            pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            dst = self.package_dir / self.manifest.preview_file
            pil_img.save(dst, "PNG")
            self._files["preview"] = dst
            self.manifest.steps_completed.append("preview")
            
        except Exception as e:
            logger.warning(f"Could not create preview: {e}")
            self.manifest.warnings.append(f"Preview generation failed: {e}")
        
        return self
    
    def add_json(self, filename: str, data: dict) -> "SolutionPackageBuilder":
        """Write an arbitrary JSON file into the package directory."""
        dst = self.package_dir / filename
        dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._files[filename] = dst
        return self

    def add_warning(self, warning: str) -> "SolutionPackageBuilder":
        """Add a warning message."""
        self.manifest.warnings.append(warning)
        return self
    
    def add_error(self, error: str) -> "SolutionPackageBuilder":
        """Add an error message."""
        self.manifest.errors.append(error)
        return self
    
    def set_quality(self, score: float, label: str) -> "SolutionPackageBuilder":
        """Set quality metrics."""
        self.manifest.quality_score = score
        self.manifest.quality_label = label
        return self
    
    def set_pipeline_meta(
        self,
        core_version: str,
        pipeline_config: Optional[Dict[str, Any]] = None,
    ) -> "SolutionPackageBuilder":
        """Set pipeline metadata for reproducibility."""
        self.manifest.core_version = core_version
        self.manifest.pipeline_config = dict(pipeline_config or {})
        return self
    
    def build(self, create_zip: bool = True) -> str:
        """
        Build the solution package.
        
        Returns:
            Path to the package (zip file if create_zip=True, directory otherwise)
        """
        # Calculate processing time
        self.manifest.processing_time_s = (datetime.now() - self._start_time).total_seconds()
        
        # Write manifest
        meta_path = self.package_dir / "meta.json"
        meta_path.write_text(
            json.dumps(self.manifest.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        logger.info(f"Solution package created: {self.package_dir}")
        logger.info(f"  Files: {list(self._files.keys())}")
        logger.info(f"  Quality: {self.manifest.quality_label} ({self.manifest.quality_score:.2f})")
        logger.info(f"  Processing time: {self.manifest.processing_time_s:.2f}s")
        
        if self.manifest.errors:
            logger.warning(f"  Errors: {self.manifest.errors}")
        
        if create_zip:
            zip_path = self.output_dir / f"{self.package_name}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in self.package_dir.iterdir():
                    zf.write(file, file.name)
            
            # Optionally clean up directory
            # shutil.rmtree(self.package_dir)
            
            return str(zip_path)
        
        return str(self.package_dir)


def load_solution_package(package_path: str) -> Dict[str, Any]:
    """
    Load a solution package from disk.
    
    Args:
        package_path: Path to zip file or directory
    
    Returns:
        Dict with loaded package contents
    """
    path = Path(package_path)
    
    if path.suffix == ".zip":
        # Extract to temp directory
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(temp_dir)
        package_dir = temp_dir
    else:
        package_dir = path
    
    result: Dict[str, Any] = {}
    
    # Load meta.json
    meta_path = package_dir / "meta.json"
    if meta_path.exists():
        result["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    
    # Load extraction.json
    extraction_path = package_dir / "extraction.json"
    if extraction_path.exists():
        result["extraction"] = json.loads(extraction_path.read_text(encoding="utf-8"))
    
    # Load semantics.json
    semantics_path = package_dir / "semantics.json"
    if semantics_path.exists():
        result["semantics"] = json.loads(semantics_path.read_text(encoding="utf-8"))
    
    # Load measurements.json
    measurements_path = package_dir / "measurements.json"
    if measurements_path.exists():
        result["measurements"] = json.loads(measurements_path.read_text(encoding="utf-8"))
    
    # Model path (don't load, just provide path)
    model_path = package_dir / "model.gltf"
    if model_path.exists():
        result["model_path"] = str(model_path)
    
    # Preview path
    preview_path = package_dir / "preview.png"
    if preview_path.exists():
        result["preview_path"] = str(preview_path)
    
    result["package_dir"] = str(package_dir)
    
    return result


def create_solution_package_from_pipeline(
    source_path: str,
    extraction_dict: Dict[str, Any],
    semantics_dict: Dict[str, Any],
    model_path: str,
    output_dir: str,
    preview_image: Optional[Any] = None,
) -> str:
    """
    Convenience function to create a complete solution package.
    
    Returns path to created package.
    """
    package_name = Path(source_path).stem + "_solution"
    
    builder = SolutionPackageBuilder(output_dir, package_name)
    builder.set_source(source_path)
    builder.add_extraction(extraction_dict)
    builder.add_semantics(semantics_dict)
    builder.add_measurements_from_extraction(extraction_dict)
    builder.add_model(model_path)
    
    if preview_image is not None:
        builder.add_preview(preview_image)
    
    return builder.build(create_zip=True)
