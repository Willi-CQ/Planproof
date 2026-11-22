import { useEffect, useRef, useState } from "react";
import type { Drawing } from "../lib/api";

/** Decode the selected source region only when the user opens its citation. */
export function RasterEvidenceDetail({source, drawing, bounds}: {
  source: Blob; drawing: Drawing; bounds: number[];
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [failed, setFailed] = useState(false);
  const [x0, y0, x1, y1] = bounds;
  useEffect(() => {
    let active = true;
    setFailed(false);
    const margin = Math.max(12, Math.ceil((y1 - y0) * .35));
    const x = Math.max(0, Math.floor(x0) - margin);
    const y = Math.max(0, Math.floor(y0) - margin);
    const w = Math.min(drawing.width, Math.ceil(x1) + margin) - x;
    const h = Math.min(drawing.height, Math.ceil(y1) + margin) - y;
    const scale = Math.min(1, 900 / w, 300 / h);
    const width = Math.max(1, Math.round(w * scale)), height = Math.max(1, Math.round(h * scale));
    createImageBitmap(source, x, y, w, h, {imageOrientation: "none", resizeWidth: width,
      resizeHeight: height, resizeQuality: "high"}).then(bitmap => {
      try {
        if (!active || !canvas.current) return;
        const target = canvas.current;
        target.width = width;
        target.height = height;
        const ctx = target.getContext("2d")!;
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(bitmap, 0, 0);
      } finally { bitmap.close(); }
    }).catch(() => { if (active) setFailed(true); });
    return () => {active = false;};
  }, [source, drawing.width, drawing.height, x0, y0, x1, y1]);
  return failed ? <p>Open the original image to inspect this detail.</p> : (
    <figure className="raster-evidence-detail">
      <figcaption>Detail from the original image</figcaption>
      <canvas ref={canvas} role="img" aria-label="Selected evidence from original pixels" />
    </figure>
  );
}
