import type { PlanSheet, PlanZone } from './plansData';

export interface ParsedZoneCitation {
  zoneId: string;
  label: string;
  zone: PlanZone | null;
}

export type InlineRun =
  | { type: 'text'; text: string; bold?: boolean }
  | { type: 'citation'; citation: ParsedZoneCitation };

export type MessageBlock =
  | { type: 'paragraph'; runs: InlineRun[] }
  | { type: 'bullet'; runs: InlineRun[] };

const STOP_WORDS = new Set([
  'the', 'and', 'for', 'with', 'that', 'this', 'from', 'what', 'when', 'where',
  'which', 'who', 'are', 'was', 'were', 'have', 'has', 'had', 'does', 'did',
  'into', 'onto', 'about', 'check', 'verify', 'show', 'tell', 'give', 'all',
  'any', 'per', 'must', 'shall', 'how', 'can', 'our', 'your', 'plan', 'drawing'
]);

export const MIN_ZOOM = 75;
export const MAX_ZOOM = 200;

export function clampZoom(value: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Math.round(value)));
}

export function tokenizeQuery(input: string): string[] {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, ' ')
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length >= 2 && !STOP_WORDS.has(t));
}

export function retrieveRelevantZones(
  query: string,
  zones: PlanZone[],
  limit = 2
): PlanZone[] {
  if (!zones.length) return [];
  const tokens = tokenizeQuery(query);
  if (!tokens.length) return zones.slice(0, limit);

  const scored = zones.map((zone) => {
    const haystack = `${zone.id} ${zone.tag} ${zone.title} ${zone.gridRef} ${zone.spec}`.toLowerCase();
    let score = 0;
    for (const token of tokens) {
      if (zone.tag.toLowerCase().includes(token) || zone.id.toLowerCase() === token) score += 6;
      if (zone.title.toLowerCase().includes(token) || zone.gridRef.toLowerCase().includes(token)) score += 4;
      if (haystack.includes(token)) score += 3;
    }
    return { zone, score };
  });

  scored.sort((a, b) => b.score - a.score);
  const matched = scored.filter((item) => item.score > 0).map((item) => item.zone);
  return (matched.length > 0 ? matched : zones).slice(0, limit);
}

function splitBoldRuns(text: string): InlineRun[] {
  const runs: InlineRun[] = [];
  const boldRegex = /\*\*([^*]+)\*\*/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = boldRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      runs.push({ type: 'text', text: text.slice(lastIndex, match.index) });
    }
    runs.push({ type: 'text', text: match[1], bold: true });
    lastIndex = boldRegex.lastIndex;
  }

  if (lastIndex < text.length) {
    runs.push({ type: 'text', text: text.slice(lastIndex) });
  }

  return runs;
}

/**
 * Parses `[[ZONE_ID|Label]]` or `[[ZONE_ID]]` tokens alongside markdown bullets and **bold** spans.
 */
export function parsePlanMessage(content: string, zones: PlanZone[]): MessageBlock[] {
  const zoneById = new Map(zones.map((z) => [z.id.toUpperCase(), z]));
  const zoneByTag = new Map(zones.map((z) => [z.tag.toUpperCase(), z]));
  const citationRegex = /\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g;

  const lines = content.split(/\r?\n/);
  const blocks: MessageBlock[] = [];

  for (const rawLine of lines) {
    const trimmed = rawLine.trim();
    if (!trimmed) continue;

    const bulletMatch = trimmed.match(/^(?:[*\-•])\s+(.*)$/);
    const isBullet = Boolean(bulletMatch);
    const lineText = bulletMatch ? bulletMatch[1] : trimmed;

    const runs: InlineRun[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;
    citationRegex.lastIndex = 0;

    while ((match = citationRegex.exec(lineText)) !== null) {
      if (match.index > lastIndex) {
        runs.push(...splitBoldRuns(lineText.slice(lastIndex, match.index)));
      }

      const rawKey = match[1].trim().toUpperCase();
      const rawLabel = match[2]?.trim();
      const matchedZone = zoneById.get(rawKey) ?? zoneByTag.get(rawKey) ?? null;

      runs.push({
        type: 'citation',
        citation: {
          zoneId: matchedZone?.id ?? rawKey,
          label: rawLabel || (matchedZone ? `${matchedZone.tag} · ${matchedZone.gridRef}` : rawKey),
          zone: matchedZone,
        },
      });

      lastIndex = citationRegex.lastIndex;
    }

    if (lastIndex < lineText.length) {
      runs.push(...splitBoldRuns(lineText.slice(lastIndex)));
    }

    blocks.push({
      type: isBullet ? 'bullet' : 'paragraph',
      runs,
    });
  }

  return blocks;
}

export function buildPlanFallbackReply(query: string, plan: PlanSheet): string {
  const tokens = tokenizeQuery(query);

  // Also check general sheet notes
  const matchingNotes = plan.sheetNotes.filter((note) =>
    tokens.some((t) => note.toLowerCase().includes(t))
  );

  const relevantZones = retrieveRelevantZones(query, plan.zones, 2);
  const topZone = relevantZones[0];

  const topHaystack = topZone
    ? `${topZone.id} ${topZone.tag} ${topZone.title} ${topZone.gridRef} ${topZone.spec}`.toLowerCase()
    : '';
  const hasZoneMatch =
    Boolean(topZone) &&
    (tokens.length === 0 || tokens.some((t) => topHaystack.includes(t)));

  if (!hasZoneMatch && matchingNotes.length === 0) {
    const tags = plan.zones.map((z) => `${z.tag} (${z.title})`).join(', ');
    return `Sheet ${plan.code} does not specify details for "${query.trim()}". Tagged zones on this drawing cover: ${tags}.`;
  }

  const lines: string[] = [];
  if (hasZoneMatch && topZone) {
    lines.push(
      `On sheet **${plan.code}** (${plan.revision}), **${topZone.title}** at ${topZone.gridRef} specifies:`
    );
    lines.push(`- ${topZone.spec} [[${topZone.id}|${topZone.tag} · ${topZone.gridRef}]]`);
  }

  if (matchingNotes.length > 0) {
    lines.push(`- General Sheet Note (${plan.code}): ${matchingNotes[0]}`);
  }

  return lines.join('\n');
}

export function validatePlanImageFile(file: {
  type: string;
  size: number;
}): { valid: boolean; error?: string } {
  const allowed = ['image/png', 'image/jpeg', 'image/webp'];
  if (!allowed.includes(file.type)) {
    return { valid: false, error: 'Upload a 2D plan image in PNG, JPG, or WebP format.' };
  }
  if (file.size > 6 * 1024 * 1024) {
    return { valid: false, error: '2D plan image must be under 6 MB.' };
  }
  return { valid: true };
}
