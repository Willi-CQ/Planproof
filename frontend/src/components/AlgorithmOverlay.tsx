import { useEffect, useRef } from "react";
import type { EvidenceShape } from "../lib/runs";
import { layerColors } from "./EvidenceInspector";

export function AlgorithmOverlay({
  shapes,
  split,
  width,
  height,
}: {
  shapes: EvidenceShape[];
  split?: number | null;
  width: number;
  height: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const paint = () => {
      const scale = Math.min(
        4096 / Math.max(width, height),
        (canvas.clientWidth * Math.min(devicePixelRatio, 2)) / width,
      );
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(scale, scale);
      ctx.lineWidth = (1.6 * width) / Math.max(canvas.clientWidth, 1);
      for (const shape of shapes) {
        ctx.save();
        if (split !== null && split !== undefined) {
          const x = (width * split) / 100;
          ctx.beginPath();
          if (shape.layer === "ml") ctx.rect(0, 0, x, height);
          else if (shape.layer === "deterministic")
            ctx.rect(x, 0, width - x, height);
          else {
            ctx.restore();
            continue;
          }
          ctx.clip();
        }
        ctx.strokeStyle = layerColors[shape.layer];
        ctx.setLineDash(
          shape.layer === "hybrid"
            ? [ctx.lineWidth * 5, ctx.lineWidth * 3]
            : [],
        );
        ctx.beginPath();
        if (shape.polygons?.length) {
          for (const polygon of shape.polygons)
            for (const ring of [polygon.outer, ...(polygon.holes || [])]) {
              ring.forEach(([x, y], i) =>
                i ? ctx.lineTo(x, y) : ctx.moveTo(x, y),
              );
              ctx.closePath();
            }
        } else {
          const [x0, y0, x1, y1] = shape.bbox;
          ctx.rect(x0, y0, x1 - x0, y1 - y0);
        }
        ctx.stroke();
        ctx.restore();
      }
      if (split !== null && split !== undefined) {
        ctx.setLineDash([]);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth *= 2;
        ctx.beginPath();
        ctx.moveTo((width * split) / 100, 0);
        ctx.lineTo((width * split) / 100, height);
        ctx.stroke();
        ctx.strokeStyle = "#001d00";
        ctx.lineWidth /= 2;
        ctx.beginPath();
        ctx.moveTo((width * split) / 100, 0);
        ctx.lineTo((width * split) / 100, height);
        ctx.stroke();
      }
    };
    const observer = new ResizeObserver(paint);
    observer.observe(canvas);
    paint();
    return () => observer.disconnect();
  }, [shapes, width, height, split]);
  return (
    <canvas
      ref={ref}
      className="algorithm-overlay"
      aria-hidden="true"
      data-testid="algorithm-overlay"
      data-shapes={shapes.length}
      data-split={split ?? "off"}
    />
  );
}
