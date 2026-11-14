"""Plan2BIM V3 — Pixel-BIM + Gravity-Well-Refinement.

See .brain/Plans/Plan2BIM-V3-Pixel-BIM.md for the full architecture.

V3 replaces the V2 wall-graph pipeline with three phases:
  1. Label refinement (SAMRefiner + Active-Contour fallback)
  2. Pixel-footprint extrusion (cv2 + shapely + trimesh, no centerlines)
  3. Opening Boolean-cut + room-from-label + DIN soft-constraint metadata
"""
