import { ZoomIn, ZoomOut, RotateCcw, MessageSquarePlus, X } from 'lucide-react';
import type { PlanSheet, PlanZone, UploadedPlanFile } from '../lib/plansData';
import { MAX_ZOOM, MIN_ZOOM, clampZoom } from '../lib/stoaEngine';

interface PlanCanvas2DProps {
  plan: PlanSheet;
  uploadedPlan: UploadedPlanFile | null;
  onClearUpload: () => void;
  selectedZoneId: string | null;
  onSelectZone: (zoneId: string | null) => void;
  zoom: number;
  onZoomChange: (nextZoom: number) => void;
  onAskAboutZone: (zone: PlanZone) => void;
}

function SheetBaseGeometry({ planId }: { planId: string }) {
  if (planId === 'plan-a101') {
    return (
      <g stroke="#0F172A" fill="none">
        {/* Outer Building Perimeter */}
        <rect x="75" y="70" width="650" height="380" strokeWidth="3.5" fill="none" />
        {/* Stair Core (Top Left) */}
        <rect x="75" y="70" width="130" height="125" strokeWidth="2.5" fill="rgba(148, 163, 184, 0.08)" />
        {[0, 1, 2, 3, 4, 5, 6].map((i) => (
          <line
            key={i}
            x1={90 + i * 14}
            y1="86"
            x2={90 + i * 14}
            y2="178"
            stroke="#64748B"
            strokeWidth="1"
          />
        ))}
        <text x="102" y="108" fill="#475569" stroke="none" fontSize="10" fontFamily="IBM Plex Mono, monospace">
          STAIR 01
        </text>

        {/* Corridor Walls */}
        <line x1="205" y1="195" x2="725" y2="195" strokeWidth="2.5" />
        <line x1="75" y1="267" x2="725" y2="267" strokeWidth="2.5" />

        {/* Animated Egress Flow Path along COR-01 */}
        <line
          x1="540"
          y1="231"
          x2="245"
          y2="231"
          stroke="#1E40AF"
          strokeWidth="1.4"
          strokeDasharray="6 6"
          className="stoa-anim-flow"
          opacity="0.7"
        />
        <g className="stoa-anim-traverse">
          <circle cx="268" cy="231" r="5" fill="#1E40AF" stroke="#FAF7EF" strokeWidth="1.5" />
          <circle cx="268" cy="231" r="10" fill="none" stroke="#1E40AF" strokeWidth="1" opacity="0.45" />
        </g>

        {/* Door D-04 Swing Arc & Animated Door Leaf */}
        <path d="M 195 150 A 38 38 0 0 1 233 188" stroke="#1E40AF" strokeWidth="1.2" strokeDasharray="3 2" />
        <line
          x1="195"
          y1="150"
          x2="195"
          y2="188"
          stroke="#1E40AF"
          strokeWidth="2.2"
          className="stoa-anim-door"
        />

        {/* Riser Shaft Cross */}
        <rect x="110" y="280" width="125" height="115" strokeWidth="2" fill="rgba(148, 163, 184, 0.1)" />
        <line x1="110" y1="280" x2="235" y2="395" stroke="#94A3B8" strokeWidth="1" />
        <line x1="235" y1="280" x2="110" y2="395" stroke="#94A3B8" strokeWidth="1" />

        {/* Acoustic Meeting Suite */}
        <rect x="430" y="280" width="220" height="135" strokeWidth="2" fill="rgba(148, 163, 184, 0.06)" />
        <rect x="475" y="325" width="130" height="46" rx="4" stroke="#64748B" strokeWidth="1.2" />

        {/* Corridor Dimension Line */}
        <line x1="595" y1="198" x2="595" y2="264" stroke="#1E40AF" strokeWidth="1" />
        <text x="602" y="235" fill="#1E40AF" stroke="none" fontSize="10" fontFamily="IBM Plex Mono, monospace">
          1,850 mm
        </text>
      </g>
    );
  }

  if (planId === 'plan-m301') {
    return (
      <g stroke="#0F172A" fill="none">
        {/* Plantroom Boundary */}
        <rect x="75" y="70" width="650" height="380" strokeWidth="3.5" fill="none" />
        {/* Inertia Plinth & Chiller CH-01 with Rotating Cooling Fans */}
        <rect x="115" y="105" width="220" height="145" strokeWidth="1.5" strokeDasharray="5 3" fill="rgba(148, 163, 184, 0.08)" />
        <rect x="140" y="128" width="170" height="98" strokeWidth="2.5" fill="none" />
        {[185, 265].map((cx) => (
          <g key={cx}>
            <circle cx={cx} cy="177" r="24" stroke="#334155" strokeWidth="1.5" />
            <g className="stoa-anim-spin">
              <line x1={cx - 17} y1="177" x2={cx + 17} y2="177" stroke="#1E40AF" strokeWidth="1.6" />
              <line x1={cx} y1="160" x2={cx} y2="194" stroke="#1E40AF" strokeWidth="1.6" />
              <circle cx={cx} cy="177" r="4" fill="#1E40AF" stroke="none" />
            </g>
          </g>
        ))}

        {/* Primary 316L Header Manifold + Animated Flow */}
        <line x1="310" y1="162" x2="530" y2="162" stroke="#1E40AF" strokeWidth="3.5" />
        <line
          x1="310"
          y1="162"
          x2="580"
          y2="162"
          stroke="#93C5FD"
          strokeWidth="1.6"
          strokeDasharray="6 8"
          className="stoa-anim-flow"
        />
        <line x1="310" y1="205" x2="530" y2="205" stroke="#0F172A" strokeWidth="3.5" />
        <line
          x1="580"
          y1="205"
          x2="310"
          y2="205"
          stroke="#CBD5E1"
          strokeWidth="1.4"
          strokeDasharray="6 8"
          className="stoa-anim-flow"
        />
        <line x1="415" y1="150" x2="415" y2="335" stroke="#1E40AF" strokeWidth="3" />
        <line x1="465" y1="195" x2="465" y2="335" stroke="#0F172A" strokeWidth="3" />

        {/* Plate Heat Exchangers HX-01 */}
        {[0, 1, 2].map((i) => (
          <rect
            key={i}
            x={580}
            y={128 + i * 48}
            width="110"
            height="34"
            strokeWidth="2"
            fill="rgba(148, 163, 184, 0.08)"
          />
        ))}

        {/* Pump Bay PMP-01 with Rotating Impellers */}
        <rect x="165" y="315" width="260" height="110" strokeWidth="1.5" strokeDasharray="4 3" fill="rgba(148, 163, 184, 0.08)" />
        {[235, 335].map((cx) => (
          <g key={cx}>
            <circle cx={cx} cy="370" r="22" strokeWidth="2" fill="none" />
            <g className="stoa-anim-spin">
              <line x1={cx - 14} y1="362" x2={cx + 14} y2="378" stroke="#0F172A" strokeWidth="1.6" />
              <line x1={cx - 14} y1="378" x2={cx + 14} y2="362" stroke="#0F172A" strokeWidth="1.6" />
            </g>
          </g>
        ))}

        {/* Clearance dimension */}
        <line x1="90" y1="105" x2="90" y2="250" stroke="#1E40AF" strokeWidth="1" />
        <text x="95" y="96" fill="#1E40AF" stroke="none" fontSize="10" fontFamily="IBM Plex Mono, monospace">
          1,200 mm AISLE
        </text>
      </g>
    );
  }

  // plan-s204 Structural CLT Framing
  return (
    <g stroke="#0F172A" fill="none">
      <rect x="75" y="70" width="650" height="380" strokeWidth="3.5" fill="none" />
      {/* Primary Glulam Beams along Gridlines */}
      <line x1="75" y1="145" x2="725" y2="145" stroke="#334155" strokeWidth="4" />
      <line x1="75" y1="242" x2="725" y2="242" stroke="#334155" strokeWidth="4" />
      <line x1="75" y1="355" x2="725" y2="355" stroke="#334155" strokeWidth="4" />

      {/* CLT Deck Panel Joints */}
      {[180, 290, 400, 510, 620].map((x) => (
        <line key={x} x1={x} y1="70" x2={x} y2="450" stroke="#CBD5E1" strokeWidth="1" />
      ))}

      {/* Glulam Columns 360x360 */}
      {[160, 400, 640].map((cx) =>
        [145, 242, 355].map((cy) => (
          <rect
            key={`${cx}-${cy}`}
            x={cx - 8}
            y={cy - 8}
            width="16"
            height="16"
            fill="#0F172A"
            stroke="none"
          />
        ))
      )}

      {/* Span Arrow + Animated Inspection Caliper */}
      <line x1="160" y1="185" x2="395" y2="185" stroke="#1E40AF" strokeWidth="1.2" />
      <g className="stoa-anim-span">
        <line x1="185" y1="173" x2="185" y2="197" stroke="#1E40AF" strokeWidth="1.6" />
        <circle cx="185" cy="185" r="3.5" fill="#1E40AF" stroke="none" />
      </g>
      <text x="235" y="178" fill="#1E40AF" stroke="none" fontSize="10" fontFamily="IBM Plex Mono, monospace">
        7,200 mm CLT SPAN
      </text>

      {/* Cross Bracing BR-01 with Animated Load Path */}
      <line x1="130" y1="305" x2="305" y2="415" stroke="#475569" strokeWidth="1.5" strokeDasharray="5 3" className="stoa-anim-flow" />
      <line x1="305" y1="305" x2="130" y2="415" stroke="#475569" strokeWidth="1.5" strokeDasharray="5 3" className="stoa-anim-flow" />

      {/* Perimeter Halfen Anchors */}
      {[320, 345, 370, 395, 420].map((y) => (
        <rect key={y} x="717" y={y} width="8" height="10" fill="#1E40AF" stroke="none" />
      ))}
    </g>
  );
}

export function PlanCanvas2D({
  plan,
  uploadedPlan,
  onClearUpload,
  selectedZoneId,
  onSelectZone,
  zoom,
  onZoomChange,
  onAskAboutZone,
}: PlanCanvas2DProps) {
  const selectedZone =
    plan.zones.find((z) => z.id === selectedZoneId) ?? null;

  return (
    <section
      aria-label="2D Plan Viewer"
      className="flex flex-col rounded-lg border border-slate-200 bg-white overflow-hidden md:h-full"
    >
      {/* Plan Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-[#F8FAFC] px-4 py-2.5">
        <div className="min-w-0">
          {uploadedPlan ? (
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-semibold text-[#1E40AF]">
                CUSTOM UPLOAD
              </span>
              <span className="truncate text-sm font-semibold text-slate-900">
                {uploadedPlan.name}
              </span>
              <button
                type="button"
                onClick={onClearUpload}
                aria-label="Remove uploaded 2D plan"
                className="inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-2 py-0.5 text-xs font-medium text-slate-700 hover:bg-slate-100 cursor-pointer"
              >
                <X className="h-3 w-3" aria-hidden="true" />
                <span>Reset sheet</span>
              </button>
            </div>
          ) : (
            <div>
              <div className="flex flex-wrap items-center gap-2 font-mono text-xs text-slate-500 tabular-nums">
                <span className="font-semibold text-[#1E40AF]">{plan.code}</span>
                <span aria-hidden="true">·</span>
                <span>{plan.discipline}</span>
                <span aria-hidden="true">·</span>
                <span>{plan.revision}</span>
                <span aria-hidden="true">·</span>
                <span>Scale {plan.scale}</span>
              </div>
              <h1 className="truncate font-display text-base font-semibold text-slate-900">
                {plan.title}
              </h1>
            </div>
          )}
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-1 shrink-0" role="group" aria-label="2D plan zoom controls">
          <button
            type="button"
            onClick={() => onZoomChange(clampZoom(zoom - 25))}
            disabled={zoom <= MIN_ZOOM}
            aria-label="Zoom out 2D plan"
            className="inline-flex min-h-9 min-w-9 items-center justify-center rounded border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40 cursor-pointer"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <span
            data-testid="zoom-level"
            className="min-w-12 text-center font-mono text-xs font-medium text-slate-700 tabular-nums"
          >
            {zoom}%
          </span>
          <button
            type="button"
            onClick={() => onZoomChange(clampZoom(zoom + 25))}
            disabled={zoom >= MAX_ZOOM}
            aria-label="Zoom in 2D plan"
            className="inline-flex min-h-9 min-w-9 items-center justify-center rounded border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40 cursor-pointer"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => onZoomChange(100)}
            disabled={zoom === 100}
            aria-label="Reset zoom to 100%"
            className="inline-flex min-h-9 items-center gap-1 rounded border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40 cursor-pointer"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Interactive Scanned 2D Viewport */}
      <div className="relative flex-1 overflow-auto bg-[#DDE1E6] p-5 sm:p-6">
        <div
          className="mx-auto flex items-center justify-center transition-transform duration-150 origin-top-left"
          style={{
            width: `${zoom}%`,
            minWidth: '100%',
          }}
        >
          {uploadedPlan ? (
            <div className="relative inline-block -rotate-[0.65deg] overflow-hidden rounded-[2px] border border-[#C6C1B5] bg-[#F5F1E6] p-3 shadow-[0_6px_20px_rgba(15,23,42,0.14)]">
              <img
                src={uploadedPlan.previewUrl}
                alt={`Scanned 2D plan: ${uploadedPlan.name}`}
                referrerPolicy="no-referrer"
                className="max-h-[520px] w-auto object-contain contrast-[1.06] sepia-[0.15] brightness-[0.98]"
              />
              <div
                aria-hidden="true"
                className="stoa-anim-scan pointer-events-none absolute inset-x-0 top-0 h-0.5 bg-[#1E40AF]/35"
              />
              <div
                aria-hidden="true"
                className="pointer-events-none absolute inset-0 shadow-[inset_0_0_28px_rgba(68,58,42,0.12)]"
              />
            </div>
          ) : (
            <div className="relative w-full max-w-[820px] -rotate-[0.6deg] overflow-hidden rounded-[2px] border border-[#C6C1B5] bg-[#F6F2E7] p-2.5 shadow-[0_8px_24px_rgba(15,23,42,0.15)]">
              <svg
                viewBox="0 0 800 520"
                role="img"
                aria-label={`Scanned 2D CAD drawing for ${plan.code} ${plan.title}`}
                className="w-full bg-[#FAF7EF]"
              >
                <defs>
                  {/* Paper grain / scanner toner texture */}
                  <filter id="stoa-scan-paper" x="0%" y="0%" width="100%" height="100%">
                    <feTurbulence
                      type="fractalNoise"
                      baseFrequency="0.75"
                      numOctaves="3"
                      seed="7"
                      result="noise"
                    />
                    <feColorMatrix
                      type="matrix"
                      values="0 0 0 0 0.25  0 0 0 0 0.22  0 0 0 0 0.16  0 0 0 0.055 0"
                      in="noise"
                      result="grain"
                    />
                  </filter>

                  {/* Subtle scanner edge vignette */}
                  <radialGradient id="stoa-scan-vignette" cx="50%" cy="50%" r="70%">
                    <stop offset="72%" stopColor="#FAF7EF" stopOpacity="0" />
                    <stop offset="100%" stopColor="#8C8270" stopOpacity="0.16" />
                  </radialGradient>
                </defs>

                {/* Subtle archival sheet border & crop ticks */}
                <rect x="8" y="8" width="784" height="504" fill="none" stroke="#B8B1A3" strokeWidth="0.9" />
                <line x1="400" y1="0" x2="400" y2="520" stroke="#D8D1C2" strokeWidth="1" opacity="0.55" />
                <line x1="0" y1="260" x2="800" y2="260" stroke="#D8D1C2" strokeWidth="1" opacity="0.55" />

                {/* Structural Gridlines */}
                {[
                  { x: 160, label: 'A' },
                  { x: 320, label: 'B' },
                  { x: 480, label: 'C' },
                  { x: 640, label: 'D' },
                ].map((g) => (
                  <g key={g.label} opacity="0.92">
                    <line
                      x1={g.x}
                      y1="42"
                      x2={g.x}
                      y2="470"
                      stroke="#CBD5E1"
                      strokeDasharray="4 4"
                      strokeWidth="1"
                    />
                    <circle cx={g.x} cy="28" r="11" fill="#FAF7EF" stroke="#64748B" strokeWidth="1" />
                    <text
                      x={g.x}
                      y="32"
                      textAnchor="middle"
                      fontSize="10"
                      fontWeight="600"
                      fill="#1E293B"
                      fontFamily="IBM Plex Mono, monospace"
                    >
                      {g.label}
                    </text>
                  </g>
                ))}

                {[
                  { y: 145, label: '1' },
                  { y: 242, label: '2' },
                  { y: 355, label: '3' },
                ].map((g) => (
                  <g key={g.label} opacity="0.92">
                    <line
                      x1="46"
                      y1={g.y}
                      x2="750"
                      y2={g.y}
                      stroke="#CBD5E1"
                      strokeDasharray="4 4"
                      strokeWidth="1"
                    />
                    <circle cx="30" cy={g.y} r="11" fill="#FAF7EF" stroke="#64748B" strokeWidth="1" />
                    <text
                      x="30"
                      y={g.y + 4}
                      textAnchor="middle"
                      fontSize="10"
                      fontWeight="600"
                      fill="#1E293B"
                      fontFamily="IBM Plex Mono, monospace"
                    >
                      {g.label}
                    </text>
                  </g>
                ))}

                {/* Architectural/MEP/Structural Vector Layer */}
                <g opacity="0.9">
                  <SheetBaseGeometry planId={plan.id} />
                </g>

                {/* Interactive Selectable Plan Zones */}
                {plan.zones.map((zone) => {
                  const isSelected = zone.id === selectedZoneId;
                  return (
                    <g
                      key={zone.id}
                      role="button"
                      tabIndex={0}
                      aria-label={`Select zone ${zone.tag} ${zone.title}`}
                      aria-pressed={isSelected}
                      onClick={() => onSelectZone(isSelected ? null : zone.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          onSelectZone(isSelected ? null : zone.id);
                        }
                      }}
                      className="cursor-pointer focus:outline-none"
                    >
                      {isSelected && (
                        <rect
                          x={zone.x}
                          y={zone.y}
                          width={zone.w}
                          height={zone.h}
                          rx="4"
                          fill="none"
                          stroke="#1E40AF"
                          strokeWidth="1.5"
                          className="stoa-anim-beacon pointer-events-none"
                        />
                      )}
                      <rect
                        x={zone.x}
                        y={zone.y}
                        width={zone.w}
                        height={zone.h}
                        rx="3"
                        fill={isSelected ? 'rgba(30, 64, 175, 0.14)' : 'rgba(30, 64, 175, 0.04)'}
                        stroke={isSelected ? '#1E40AF' : '#64748B'}
                        strokeWidth={isSelected ? '2.2' : '1'}
                        strokeDasharray={isSelected ? undefined : '4 3'}
                      />
                      <rect
                        x={zone.x + 6}
                        y={zone.y + 6}
                        width="68"
                        height="20"
                        rx="3"
                        fill={isSelected ? '#1E40AF' : '#1E293B'}
                      />
                      <text
                        x={zone.x + 40}
                        y={zone.y + 19}
                        textAnchor="middle"
                        fontSize="10"
                        fontWeight="600"
                        fill="#FFFFFF"
                        fontFamily="IBM Plex Mono, monospace"
                      >
                        {zone.tag}
                      </text>
                    </g>
                  );
                })}

                {/* Physical Archival Ink Stamp */}
                <g transform="translate(24, 464) rotate(-2.5)" opacity="0.45">
                  <rect x="0" y="0" width="152" height="28" rx="2" fill="none" stroke="#991B1B" strokeWidth="1.4" />
                  <text
                    x="76"
                    y="17"
                    textAnchor="middle"
                    fontSize="8.5"
                    fontWeight="600"
                    letterSpacing="0.8"
                    fill="#991B1B"
                    fontFamily="IBM Plex Mono, monospace"
                  >
                    ISSUED FOR CONSTRUCTION
                  </text>
                </g>

                {/* CAD Sheet Title Block */}
                <g transform="translate(530, 466)">
                  <rect x="0" y="0" width="220" height="36" fill="#F4EFE2" stroke="#94A3B8" strokeWidth="1" />
                  <text x="10" y="15" fontSize="9.5" fontWeight="600" fill="#1E293B" fontFamily="IBM Plex Mono, monospace">
                    {plan.code} · {plan.revision} · {plan.scale}
                  </text>
                  <text x="10" y="28" fontSize="9" fill="#475569" fontFamily="Public Sans, sans-serif">
                    {plan.project}
                  </text>
                </g>

                {/* Scanner Grain & Edge Vignette Overlay */}
                <rect x="0" y="0" width="800" height="520" filter="url(#stoa-scan-paper)" pointerEvents="none" />
                <rect x="0" y="0" width="800" height="520" fill="url(#stoa-scan-vignette)" pointerEvents="none" />

                {/* Subtle Optical Scan Sweep Line */}
                <line
                  x1="0"
                  y1="14"
                  x2="800"
                  y2="14"
                  stroke="#1E40AF"
                  strokeWidth="1.2"
                  opacity="0.3"
                  className="stoa-anim-scan"
                  pointerEvents="none"
                />
              </svg>
            </div>
          )}
        </div>
      </div>

      {/* Quick Zone Index + Active Callout Inspector Strip */}
      {!uploadedPlan && (
        <div className="border-t border-slate-200 bg-white px-4 py-3">
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="2D plan callout zones">
            <span className="mr-1 text-xs font-medium text-slate-500">
              Tagged Zones:
            </span>
            {plan.zones.map((zone) => {
              const isSelected = zone.id === selectedZoneId;
              return (
                <button
                  key={zone.id}
                  type="button"
                  aria-pressed={isSelected}
                  onClick={() => onSelectZone(isSelected ? null : zone.id)}
                  className={`inline-flex min-h-8 items-center gap-1.5 rounded border px-2.5 py-1 font-mono text-xs font-medium whitespace-nowrap shrink-0 transition-colors cursor-pointer ${
                    isSelected
                      ? 'border-[#1E40AF] bg-[#1E40AF] text-white'
                      : 'border-slate-200 bg-[#F8FAFC] text-slate-700 hover:border-[#1E40AF]/50 hover:text-[#1E40AF]'
                  }`}
                >
                  <span>{zone.tag}</span>
                  <span className="opacity-70">· {zone.gridRef}</span>
                </button>
              );
            })}
          </div>

          {selectedZone && (
            <div
              data-testid="zone-detail-strip"
              className="mt-2.5 flex flex-col gap-2 rounded-md border border-slate-200 bg-[#F8FAFC] p-3 sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="min-w-0">
                <div className="font-mono text-xs text-[#1E40AF]">
                  {selectedZone.tag} · {selectedZone.gridRef} · {selectedZone.title}
                </div>
                <p className="mt-0.5 text-xs leading-relaxed text-slate-700">
                  {selectedZone.spec}
                </p>
              </div>
              <button
                type="button"
                onClick={() => onAskAboutZone(selectedZone)}
                className="inline-flex min-h-9 items-center gap-1.5 self-start rounded bg-[#0F172A] px-3 py-1.5 text-xs font-medium text-white whitespace-nowrap shrink-0 hover:bg-[#1E40AF] transition-colors cursor-pointer sm:self-center"
              >
                <MessageSquarePlus className="h-3.5 w-3.5" aria-hidden="true" />
                <span>Ask about {selectedZone.tag}</span>
              </button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
