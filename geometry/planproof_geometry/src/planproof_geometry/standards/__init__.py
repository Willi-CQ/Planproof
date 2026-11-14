"""German construction standards (DIN) as structured reference data.

Used by the engineering-rules validator (v2 pipeline) to snap detected
dimensions to standard values and flag non-conformities.
"""
from planproof_geometry.standards.din import load_din_standards

__all__ = ["load_din_standards"]
