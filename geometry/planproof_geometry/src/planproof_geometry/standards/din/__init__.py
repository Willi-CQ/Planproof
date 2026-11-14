"""DIN standards as structured JSON + typed Python accessor.

Three JSON files ship with the package (critical / high / medium priority
norms). `load_din_standards()` returns a `DinStandards` dataclass with
converted-to-metres values ready for use by the engineering-rules validator.
"""
from planproof_geometry.standards.din.loader import (
    DinStandards,
    load_din_standards,
)

__all__ = ["DinStandards", "load_din_standards"]
