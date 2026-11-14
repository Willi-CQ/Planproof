export interface PlanZone {
  id: string;
  tag: string;
  title: string;
  gridRef: string;
  x: number;
  y: number;
  w: number;
  h: number;
  spec: string;
}

export interface PlanSheet {
  id: string;
  code: string;
  shortLabel: string;
  title: string;
  project: string;
  discipline: 'Architectural' | 'MEP' | 'Structural';
  revision: string;
  scale: string;
  sheetNotes: string[];
  starterPrompts: string[];
  zones: PlanZone[];
}

export interface PlanChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface UploadedPlanFile {
  name: string;
  mimeType: string;
  base64Data: string;
  previewUrl: string;
}

export const COMPANY_PLANS: PlanSheet[] = [
  {
    id: 'plan-a101',
    code: 'A-101',
    shortLabel: 'A-101 · Level 01',
    title: 'Level 01 Core, Corridor & Fire Compartment Plan',
    project: 'Aarhus Harbour Timber Office',
    discipline: 'Architectural',
    revision: 'Rev 05',
    scale: '1:100 @ A1',
    sheetNotes: [
      'All structural grid dimensions are in millimetres (mm) from gridline centrelines.',
      'Fire compartment walls along Gridline B shall maintain continuous EI 60 sealing to underside of CLT slab.',
      'Door schedule references D-01 through D-14; magnetic hold-open devices tie to Zone 1 fire alarm loop.',
    ],
    starterPrompts: [
      'What is the clear corridor width along Gridline B?',
      'Check fire rating and hardware on Door D-04.',
      'What acoustic rating separates the meeting suite from open office?',
    ],
    zones: [
      {
        id: 'Z1',
        tag: 'COR-01',
        title: 'Main Egress Corridor (Grid B–C)',
        gridRef: 'Grid B-2',
        x: 245,
        y: 195,
        w: 310,
        h: 72,
        spec: 'Minimum clear unobstructed egress width 1,850 mm between finished acoustic wall linings. Floor finish: honed terrazzo over 40 mm acoustic screed.',
      },
      {
        id: 'Z2',
        tag: 'D-04',
        title: 'Stair Core Fire Door Assembly D-04',
        gridRef: 'Grid B-1',
        x: 150,
        y: 118,
        w: 110,
        h: 78,
        spec: 'Double-leaf EI2 60-C5Sm fire and smoke door, 2,100 x 2,250 mm clear opening, fitted with fail-safe EN 1155 magnetic hold-open closers.',
      },
      {
        id: 'Z3',
        tag: 'MTG-104',
        title: 'Client Briefing & Acoustic Suite',
        gridRef: 'Grid C-2',
        x: 430,
        y: 280,
        w: 220,
        h: 135,
        spec: 'Double-glazed demountable partition rated Rw 48 dB with full-height acoustic baffle above suspended timber slats.',
      },
      {
        id: 'Z4',
        tag: 'RIS-02',
        title: 'West MEP & Sprinkler Riser Shaft',
        gridRef: 'Grid A-2',
        x: 110,
        y: 280,
        w: 125,
        h: 115,
        spec: '1,600 x 1,400 mm structural shaft with EI 90 fire-stopped slab penetrations and 800 mm clear front access hatch.',
      },
    ],
  },
  {
    id: 'plan-m301',
    code: 'M-301',
    shortLabel: 'M-301 · Plantroom',
    title: 'Room 2B Chilled Water & Heat Recovery Layout',
    project: 'Nordhavn Data Centre — Phase II',
    discipline: 'MEP',
    revision: 'Rev 04',
    scale: '1:50 @ A1',
    sheetNotes: [
      'Secondary Hall B cooling loop operates at 18.0 °C supply / 28.5 °C return.',
      'All primary headers in Room 2B require 316L stainless flanged spools rated PN16 and hydro-tested to 1.5x working pressure for 4 hours.',
      'Maintain minimum 1,200 mm tube-pull and maintenance aisle around Chiller Skids CH-01 and CH-02.',
    ],
    starterPrompts: [
      'What maintenance clearance is required around CH-01?',
      'Verify pressure test and pipe grade for Header H-2B.',
      'Check supply and return temperatures for Plate HX-01.',
    ],
    zones: [
      {
        id: 'Z1',
        tag: 'CH-01',
        title: 'Chiller Skid CH-01 & Inertia Plinth',
        gridRef: 'Grid A-1',
        x: 115,
        y: 105,
        w: 220,
        h: 145,
        spec: '6.4 m x 2.2 m C40/50 concrete inertia plinth (21-day wet cure, flatness <= 2.0 mm/3.0 m). Minimum 1,200 mm service clearance on all four sides.',
      },
      {
        id: 'Z2',
        tag: 'H-2B',
        title: 'Primary 316L Distribution Header',
        gridRef: 'Grid B-2',
        x: 365,
        y: 140,
        w: 175,
        h: 190,
        spec: 'DN300 316L stainless steel PN16 flanged manifold. Hydrostatic test 1.5x working pressure for 4 hours prior to closed-cell elastomeric insulation.',
      },
      {
        id: 'Z3',
        tag: 'HX-01',
        title: 'N+1 Plate Heat Exchanger Bank',
        gridRef: 'Grid C-2',
        x: 565,
        y: 115,
        w: 145,
        h: 165,
        spec: 'Three plate heat exchangers (2 duty + 1 standby) delivering 18.0 °C supply / 28.5 °C return to Hall B CRAH units.',
      },
      {
        id: 'Z4',
        tag: 'PMP-01',
        title: 'Variable-Speed Primary Pump Bay',
        gridRef: 'Grid B-3',
        x: 165,
        y: 315,
        w: 260,
        h: 110,
        spec: 'Twin inline VSD centrifugal pumps on spring-viscous isolators with 950 mm crane hoist beam centerline above pump axes.',
      },
    ],
  },
  {
    id: 'plan-s204',
    code: 'S-204',
    shortLabel: 'S-204 · Framing',
    title: 'Level 03 CLT Floor Deck & Glulam Grid Plan',
    project: 'Aarhus Harbour Timber Office',
    discipline: 'Structural',
    revision: 'Rev 03',
    scale: '1:100 @ A1',
    sheetNotes: [
      'Primary glulam columns along Grids 1–3 carry REI 90 rating (beta_0 = 0.65 mm/min charring rate).',
      'Exposed beam-to-column steel knife plates must be recessed >= 25 mm and plugged with hardwood dowels.',
      'Perimeter Halfen channels require 5% witnessed pull-out testing before unitised facade hanging.',
    ],
    starterPrompts: [
      'What span and thickness apply to the CLT deck panels?',
      'Check steel knife plate recess depth at Grid B-2.',
      'Where are Halfen facade anchor pull tests required?',
    ],
    zones: [
      {
        id: 'Z1',
        tag: 'COL-B2',
        title: 'Primary Glulam Column Grid B-2',
        gridRef: 'Grid B-2',
        x: 345,
        y: 195,
        w: 110,
        h: 95,
        spec: '360 x 360 mm GL30c glulam column rated REI 90 (beta_0 = 0.65 mm/min). Recessed 25 mm steel knife-plate connection with oak dowel plugs.',
      },
      {
        id: 'Z2',
        tag: 'CLT-03',
        title: '7-Ply CLT Structural Deck Span',
        gridRef: 'Grid B-1',
        x: 120,
        y: 100,
        w: 210,
        h: 165,
        spec: '240 mm 7-ply spruce CLT panels spanning 7,200 mm clear between primary glulam beams; core moisture must stay below 15.5% before sealing.',
      },
      {
        id: 'Z3',
        tag: 'HAF-E',
        title: 'Perimeter Slab-Edge Halfen Anchors',
        gridRef: 'Grid C-3',
        x: 475,
        y: 310,
        w: 215,
        h: 110,
        spec: 'Cast-in Halfen channels at 1,500 mm centres along east curtain wall edge; 5% (min 12 anchors per elevation) witnessed pull-out tested.',
      },
      {
        id: 'Z4',
        tag: 'BR-01',
        title: 'Lateral Steel Tension Cross-Bracing',
        gridRef: 'Grid A-3',
        x: 120,
        y: 300,
        w: 195,
        h: 120,
        spec: 'M30 grade 8.8 diagonal tie rods anchored to concrete core shear wall; pre-tensioned to 35 kN after Level 04 deck placement.',
      },
    ],
  },
];
