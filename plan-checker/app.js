const $ = (selector) => document.querySelector(selector);
const fileInput = $('#fileInput');
const dropzone = $('#dropzone');
const drawing = $('#drawingWrap');
const initialDrawing = drawing.innerHTML;
const accepted = ['application/pdf', 'image/png', 'image/jpeg'];
const counts = { deterministic: 'Déterministe', ml: 'ML' };
const reasons = {
  analyzer_unavailable: 'Au moins un moteur est indisponible.',
  non_independent: 'Les résultats ont une source commune.',
  ambiguous_matches: 'Plusieurs correspondances sont possibles.',
  unmatched_detections: 'Certaines propositions ne correspondent pas.',
  coverage_unknown: 'La couverture de la page reste inconnue.',
};

let selectedFile;
let analysis;
let assistantAvailable = false;
let busy = false;
let currentJsonTab = 'cross';

function setSonaState(state, label) {
  const indicator = $('#sonaState');
  indicator.dataset.state = state;
  indicator.textContent = label;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = response.headers.get('content-type')?.includes('application/json') ? await response.json() : null;
  if (!response.ok) {
    throw new Error(typeof payload?.detail === 'string' ? payload.detail : 'Le serveur ne répond pas. Réessayez dans un instant.');
  }
  if (!payload) throw new Error('Réponse inattendue du serveur.');
  return payload;
}

function toast(message) {
  const element = $('#toast');
  element.textContent = message;
  element.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove('show'), 3500);
}

function updateControls() {
  fileInput.disabled = busy;
  for (const id of ['dropzone', 'replaceFile', 'removeFile', 'newAnalysis', 'runDemo']) {
    $(`#${id}`).disabled = busy;
  }
  $('#pageNumber').disabled = busy || selectedFile?.type !== 'application/pdf';
  $('#assistantPermission').disabled = busy || !assistantAvailable;
  $('#runAnalysis').disabled = busy || !selectedFile;
}

function resetResult() {
  analysis = null;
  $('#results').hidden = true;
  $('#assistantContext').hidden = true;
  $('#decisionBadge').textContent = 'En attente';
  $('#decisionBadge').className = 'decision-badge review';
  $('#decisionTitle').textContent = 'Lancez une analyse';
  $('#decisionCopy').textContent = 'Les résultats apparaîtront ici.';
  $('#decisionSymbol').textContent = '…';
  for (const id of ['metricRooms', 'metricArea', 'metricIssues', 'detValue', 'mlValue']) {
    $(`#${id}`).textContent = '—';
  }
  $('#assistantContextTitle').textContent = selectedFile?.name || 'Aucun plan analysé';
  $('#assistantPage').textContent = '';
  setSonaState('idle', 'En attente');
  $('#reviewMessage').textContent = selectedFile
    ? 'Analysez le plan pour afficher les zones à inspecter.'
    : 'Lancez le plan d’exemple pour voir comment Sona explique un désaccord.';
  $('#assistantReferences').replaceChildren();
  $('#zoneInspection').hidden = true;
  $('#planLegend').hidden = true;
  $('#evidenceList').replaceChildren();
  $('#jsonPanel').hidden = true;
  $('#jsonButton').setAttribute('aria-expanded', 'false');
  $('#jsonButton').disabled = true;
}

function setFile(file) {
  if (busy || !file) return;
  if (/\.svg$/i.test(file.name)) return toast('Pour un plan SVG, utilisez le PDF associé ou exportez le dessin en PNG.');
  if (!accepted.includes(file.type)) return toast('Choisissez un PDF, PNG ou JPEG.');
  if (file.size > 10 * 1024 * 1024) return toast('Le fichier dépasse la limite de 10 Mo.');
  selectedFile = file;
  $('#fileName').textContent = file.name;
  $('#fileSize').textContent = `${Math.max(1, Math.round(file.size / 1024))} Ko`;
  $('#fileType').textContent = file.type.split('/')[1].toUpperCase();
  $('#fileRow').hidden = false;
  $('#uploadOptions').hidden = false;
  $('#replaceFile').hidden = false;
  $('#pageInputLabel').hidden = file.type !== 'application/pdf';
  dropzone.hidden = true;
  $('#pageNumber').value = 1;
  $('#assistantPermission').checked = false;
  const placeholder = document.createElement('div');
  placeholder.className = 'pdf-placeholder';
  placeholder.textContent = 'La page analysée apparaîtra ici.';
  drawing.replaceChildren(placeholder);
  setPreviewLabel('FICHIER IMPORTÉ', '');
  resetResult();
  updateControls();
}

function resetApp() {
  selectedFile = undefined;
  fileInput.value = '';
  dropzone.hidden = false;
  $('#fileRow').hidden = true;
  $('#uploadOptions').hidden = true;
  $('#replaceFile').hidden = true;
  $('#assistantPermission').checked = false;
  drawing.innerHTML = initialDrawing;
  setPreviewLabel('Aperçu d’un plan d’étage', '');
  resetResult();
  updateControls();
}

function setPreviewLabel(label, details) {
  const line = $('.preview-topline');
  line.querySelector('span:first-child').textContent = label;
  line.querySelector('span:last-child').textContent = details;
}

function paintZone(image, item, width, height) {
  const canvas = $('#zoneCrop');
  const [left, top, right, bottom] = item.bbox_xyxy;
  const margin = Math.max(20, Math.max(right - left, bottom - top) * 0.9);
  const x = Math.max(0, left - margin);
  const y = Math.max(0, top - margin);
  const w = Math.min(width, right + margin) - x;
  const h = Math.min(height, bottom + margin) - y;
  const scale = Math.min(640 / w, 370 / h);
  canvas.width = Math.max(1, Math.round(w * scale));
  canvas.height = Math.max(1, Math.round(h * scale));
  const context = canvas.getContext('2d');
  context.drawImage(image, x, y, w, h, 0, 0, canvas.width, canvas.height);
  context.strokeStyle = item.provider === 'ml' ? '#256487' : '#a45424';
  context.lineWidth = 3;
  context.strokeRect((left - x) * scale, (top - y) * scale, (right - left) * scale, (bottom - top) * scale);
}

function renderEvidence(result) {
  const { width, height, page_index: index } = result.source;
  const wrapper = document.createElement('div');
  wrapper.className = 'evidence-canvas';
  wrapper.style.aspectRatio = `${width} / ${height}`;
  wrapper.style.width = `min(100%, ${330 * width / height}px)`;
  const image = document.createElement('img');
  image.src = result.image_url;
  image.alt = `Plan source, page ${index + 1}`;
  wrapper.append(image);

  const primary = result.disputed_evidence.length ? result.disputed_evidence : result.unverified_evidence;
  const evidence = [...(primary.length ? primary : result.matched_evidence)];
  const cited = new Set(result.assistant?.evidence_ids);
  evidence.sort((a, b) => Number(cited.has(b.id)) - Number(cited.has(a.id)));
  const list = $('#evidenceList');
  list.replaceChildren();
  const references = $('#assistantReferences');
  references.replaceChildren();
  let visibleMessage = $('#reviewMessage').textContent;
  $('#planLegend').hidden = evidence.length === 0;
  $('#zoneInspection').hidden = true;
  const markers = [];
  const showOverview = evidence.length <= 24;
  const buttons = [];
  const largeList = evidence.length > 8;
  const focusDetail = () => {
    if (window.matchMedia('(max-width: 720px)').matches) {
      $('#zoneInspection').scrollIntoView({ block: 'nearest' });
    }
  };
  const selector = document.createElement('select');
  selector.id = 'evidenceZone';
  let currentZone = 0;
  function showZone(position) {
    const item = evidence[position];
    if (!item) return;
    currentZone = position;
    markers.forEach((marker, index) => {
      marker.hidden = !showOverview && index !== position;
      marker.classList.toggle('evidence-marker--active', index === position);
      marker.setAttribute('aria-pressed', String(index === position));
    });
    buttons.forEach((button, index) => button.setAttribute('aria-pressed', String(index === position)));
    selector.value = String(position);
    const label = `Zone ${position + 1} · ${counts[item.provider]}`;
    $('#zoneTitle').textContent = label;
    $('#zoneStatus').textContent = cited.has(item.id) ? 'Citée par Sona'
      : primary.length ? 'À vérifier' : 'Détections concordantes';
    $('#zoneCrop').setAttribute('aria-label', `Agrandissement de la zone ${position + 1} du plan, proposition ${counts[item.provider]}`);
    $('#zonePosition').textContent = `${position + 1} / ${evidence.length}`;
    $('#previousZone').disabled = position === 0;
    $('#nextZone').disabled = position === evidence.length - 1;
    if (image.complete && image.naturalWidth) {
      paintZone(image, item, width, height);
      $('#zoneInspection').hidden = false;
    }
  }
  for (const [position, item] of evidence.entries()) {
    const [left, top, right, bottom] = item.bbox_xyxy;
    const marker = document.createElement('button');
    marker.type = 'button';
    marker.className = `evidence-marker evidence-marker--${item.provider}`;
    marker.classList.toggle('evidence-marker--cited', cited.has(item.id));
    marker.hidden = !showOverview && position !== 0;
    marker.setAttribute('aria-label', `Inspecter la zone ${position + 1} · ${counts[item.provider]}`);
    marker.addEventListener('click', () => {
      showZone(position);
      focusDetail();
    });
    Object.assign(marker.style, {
      left: `${left / width * 100}%`, top: `${top / height * 100}%`,
      width: `${(right - left) / width * 100}%`, height: `${(bottom - top) / height * 100}%`,
    });
    wrapper.append(marker);
    markers.push(marker);
    const label = `Zone ${position + 1} · ${counts[item.provider]}${cited.has(item.id) ? ' · citée' : ''}`;
    if (cited.has(item.id)) {
      const escapedId = item.id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const referencePattern = new RegExp(`(?<![\\w:-])${escapedId}(?![\\w:-])`, 'g');
      visibleMessage = visibleMessage.replace(referencePattern, `zone ${position + 1}`);
      const reference = document.createElement('button');
      reference.type = 'button';
      reference.textContent = `Voir la zone ${position + 1}`;
      reference.addEventListener('click', () => {
        showZone(position);
        $('#zoneInspection').scrollIntoView({ block: 'nearest' });
      });
      references.append(reference);
    }
    if (largeList) {
      selector.add(new Option(label, String(position)));
      continue;
    }
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'text-button';
    button.textContent = label;
    button.setAttribute('aria-pressed', String(position === 0));
    button.addEventListener('click', () => {
      showZone(position);
      focusDetail();
    });
    buttons.push(button);
    list.append(button);
  }
  $('#reviewMessage').textContent = visibleMessage;
  if (largeList) {
    const label = document.createElement('label');
    label.htmlFor = selector.id;
    label.textContent = `Zones du plan (${evidence.length})`;
    selector.addEventListener('change', () => {
      showZone(Number(selector.value));
      focusDetail();
    });
    list.append(label, selector);
  }
  drawing.replaceChildren(wrapper);
  image.addEventListener('load', () => {
    if (analysis === result) showZone(currentZone);
  });
  $('#previousZone').onclick = () => showZone(currentZone - 1);
  $('#nextZone').onclick = () => showZone(currentZone + 1);
  showZone(0);
  const link = document.createElement('a');
  link.href = result.image_url;
  link.target = '_blank';
  link.rel = 'noopener';
  link.textContent = 'Ouvrir la page source ↗';
  list.append(link);
  setPreviewLabel(result.mode === 'synthetic_demo' ? 'PLAN SYNTHÉTIQUE' : 'PAGE ANALYSÉE', `PAGE ${index + 1}`);
}

function jsonForTab() {
  if (currentJsonTab === 'det') return analysis.deterministic;
  if (currentJsonTab === 'ml') return analysis.ml;
  const {
    status, answer, reasons, deterministic, ml, matched_pairs, matched_evidence, disputed_evidence,
    unverified_evidence, schema_version, request_id, source, image_url, mode, assistant,
  } = analysis;
  return {
    status, answer, reasons, deterministic, ml, matched_pairs, matched_evidence, disputed_evidence,
    unverified_evidence, schema_version, request_id, source, image_url, mode,
    assistant: assistant && {
      status: assistant.status, message: assistant.message, evidence_ids: assistant.evidence_ids,
    },
  };
}

function render(result, fileName) {
  analysis = result;
  $('#results').hidden = false;
  $('#assistantContext').hidden = false;
  const supported = result.status === 'SUPPORTED';
  const labels = { SUPPORTED: 'Concordant', NEEDS_REVIEW: 'À vérifier', INSUFFICIENT_EVIDENCE: 'Preuves insuffisantes' };
  $('#decisionBadge').textContent = labels[result.status] || result.status;
  $('#decisionBadge').className = `decision-badge ${supported ? 'consistent' : 'review'}`;
  $('#decisionSymbol').textContent = supported ? '✓' : '!';
  $('#decisionTitle').textContent = supported ? 'Les détections concordent' : 'Total de portes indéterminé';
  const noAgreement = result.status === 'NEEDS_REVIEW' && result.matched_pairs.length === 0;
  $('#decisionCopy').textContent = noAgreement
    ? 'Aucune proposition de porte n’a pu être appariée. Sélectionnez une zone pour inspecter le plan.'
    : result.reasons.map((reason) => reasons[reason] || reason).join(' ')
      || 'Les deux moteurs couvrent la page et les positions correspondent.';
  $('#metricRooms').textContent = result.answer ?? 'À vérifier';
  $('#metricArea').textContent = result.matched_pairs.length;
  $('#metricIssues').textContent = result.disputed_evidence.length + result.unverified_evidence.length;
  for (const [key, id] of [['deterministic', 'detValue'], ['ml', 'mlValue']]) {
    const number = result[key].detected_count;
    $(`#${id}`).textContent = number == null ? 'Moteur indisponible'
      : number === 0 ? 'Aucune proposition détectée'
      : `${number} ${number === 1 ? 'proposition' : 'propositions'}`;
  }
  $('#assistantContextTitle').textContent = fileName;
  $('#assistantPage').textContent = `Page ${result.source.page_index + 1}`;
  const assistant = result.assistant;
  const observed = Boolean(assistant?.evidence_ids?.length);
  if (assistant?.status === 'answered') {
    setSonaState('answered', observed ? 'Zone citée' : 'Synthèse des résultats');
  } else if (assistant?.status === 'unavailable') {
    setSonaState('unavailable', 'Explication indisponible');
  } else {
    setSonaState('local', 'Revue du plan');
  }
  const assistantStates = {
    not_needed: 'Ce résultat ne nécessite pas d’explication supplémentaire.',
    not_authorized: 'Activez l’assistant si vous souhaitez une explication.',
    unavailable: 'L’assistant est indisponible. Vous pouvez toujours consulter le résultat et les zones.',
  };
  $('#reviewMessage').textContent = assistant?.message || assistantStates[assistant?.status] || assistantStates.unavailable;
  renderEvidence(result);
  $('#jsonOutput').textContent = JSON.stringify(jsonForTab(), null, 2);
  $('#jsonButton').disabled = false;
}

async function analyze(demo = false) {
  if (busy || (!demo && !selectedFile)) return;
  busy = true;
  updateControls();
  const permission = assistantAvailable && (demo || $('#assistantPermission').checked);
  const trigger = $(demo ? '#runDemo' : '#runAnalysis');
  const label = trigger.innerHTML;
  resetResult();
  trigger.textContent = 'Analyse en cours…';
  setSonaState('busy', 'Analyse en cours');
  $('#reviewMessage').textContent = permission
    ? 'Sona prépare l’explication et les zones à retrouver sur le plan.'
    : 'Les deux moteurs analysent le plan.';
  try {
    let path = '/api/v1/demo/door-count';
    const params = new URLSearchParams();
    params.set('allow_assistant_review', String(permission));
    if (demo) params.set('scenario', 'disputed');
    if (!demo) {
      const form = new FormData();
      form.append('file', selectedFile);
      form.append('page', $('#pageNumber').value || '1');
      const upload = await api('/api/documents', { method: 'POST', body: form });
      path = `/api/v1/documents/${upload.document_id}/analyze-doors`;
    }
    const report = await api(`${path}?${params}`, { method: 'POST' });
    render(report, demo ? 'Plan synthétique' : selectedFile.name);
    if (window.matchMedia('(max-width: 720px)').matches) {
      $('#planPreview').scrollIntoView({ block: 'start' });
    }
  } catch (error) {
    resetResult();
    setSonaState('unavailable', 'Analyse interrompue');
    $('#reviewMessage').textContent = error.message;
    toast(error.message);
  } finally {
    busy = false;
    trigger.innerHTML = label;
    updateControls();
  }
}

async function readConfig() {
  try {
    const config = await api('/api/v1/engines');
    $('#modeLabel').textContent = config.assistant ? 'Sona disponible' : 'Vérification locale';
    $('#detState').textContent = config.deterministic ? 'Configuré' : 'Indisponible';
    $('#mlState').textContent = config.ml ? 'Configuré' : 'Indisponible';
    assistantAvailable = Boolean(config.assistant);
    $('#assistantState').textContent = assistantAvailable ? 'Disponible' : 'Indisponible';
    updateControls();
  } catch {
    $('#modeLabel').textContent = 'API indisponible';
    $('#assistantState').textContent = 'Indisponible';
  }
}

fileInput.addEventListener('change', () => setFile(fileInput.files?.[0]));
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (event) => event.preventDefault());
dropzone.addEventListener('drop', (event) => { event.preventDefault(); setFile(event.dataTransfer.files?.[0]); });
$('#replaceFile').addEventListener('click', () => fileInput.click());
$('#removeFile').addEventListener('click', resetApp);
$('#newAnalysis').addEventListener('click', resetApp);
$('#runAnalysis').addEventListener('click', () => analyze());
$('#runDemo').addEventListener('click', () => analyze(true));
$('#jsonButton').addEventListener('click', () => {
  $('#jsonPanel').hidden = !$('#jsonPanel').hidden;
  $('#jsonButton').setAttribute('aria-expanded', String(!$('#jsonPanel').hidden));
});
document.querySelectorAll('.json-tab').forEach((tab) => tab.addEventListener('click', () => {
  currentJsonTab = tab.dataset.json;
  document.querySelectorAll('.json-tab').forEach((item) => item.classList.toggle('active', item === tab));
  $('#jsonOutput').textContent = JSON.stringify(jsonForTab(), null, 2);
}));
$('#copyJson').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($('#jsonOutput').textContent);
    toast('JSON copié.');
  } catch {
    toast('Copie indisponible dans ce navigateur.');
  }
});
resetApp();
readConfig();
