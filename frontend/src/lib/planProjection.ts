export type PlanPoint = [number, number];

export const overviewCamera = { yaw: -28, tilt: 49, zoom: 1 };

export function projectPlan(width: number, height: number, frame: PlanPoint,
  camera: typeof overviewCamera, center: PlanPoint = [width / 2, height / 2]) {
  const radians = Math.PI / 180;
  const cos = Math.cos(camera.yaw * radians), sin = Math.sin(camera.yaw * radians);
  const squash = Math.cos(camera.tilt * radians), lift = Math.sin(camera.tilt * radians);
  const rangeX = Math.abs(cos) * width + Math.abs(sin) * height;
  const rangeY = (Math.abs(sin) * width + Math.abs(cos) * height) * squash;
  const padding = Math.min(48, frame[0] * .09);
  const scale = Math.min((frame[0] - 2 * padding) / rangeX,
    Math.max(1, frame[1] - 124) / rangeY) * camera.zoom;
  const a = scale * cos, b = scale * sin * squash;
  const c = -scale * sin, d = scale * cos * squash;
  const e = frame[0] / 2 - a * center[0] - c * center[1];
  const f = frame[1] / 2 + 12 - b * center[0] - d * center[1];
  return {
    matrix: `matrix(${a} ${b} ${c} ${d} ${e} ${f})`,
    point: ([x, y]: PlanPoint, z = 0): PlanPoint => [a * x + c * y + e, b * x + d * y + f - z * lift * scale],
    depth: ([x, y]: PlanPoint) => sin * x + cos * y,
    scale,
  };
}

function cross([a, b]: PlanPoint, [x, y]: PlanPoint, [u, v]: PlanPoint) {
  return (x - a) * (v - b) - (y - b) * (u - a);
}

export function simpleFootprint(vertices: PlanPoint[]) {
  const n = vertices.length;
  if (n < 3 || new Set(vertices.map(p => p.join(","))).size !== n) return false;
  let twiceArea = 0;
  for (let i = 0; i < n; i++) {
    const a = vertices[i], b = vertices[(i + 1) % n];
    twiceArea += a[0] * b[1] - a[1] * b[0];
    for (let j = i + 2; j < n; j++) {
      if (i === 0 && j === n - 1) continue;
      const c = vertices[j], d = vertices[(j + 1) % n];
      const overlaps = (axis: number) => Math.max(Math.min(a[axis], b[axis]), Math.min(c[axis], d[axis]))
        <= Math.min(Math.max(a[axis], b[axis]), Math.max(c[axis], d[axis]));
      if (overlaps(0) && overlaps(1) && cross(a, b, c) * cross(a, b, d) <= 0
          && cross(c, d, a) * cross(c, d, b) <= 0) return false;
    }
  }
  return Math.abs(twiceArea) > 1;
}
