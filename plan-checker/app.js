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
let busy = false;
let currentJsonTab = 'cross';

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Échec de la requête.');
  }
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
  for (const id of ['reasonQuestion', 'reasonMode', 'reasonPermission', 'specialistPermission']) {
    const control = document.getElementById(id);
    if (control) control.disabled = busy;
  }
  for (const id of ['dropzone', 'replaceFile', 'removeFile', 'newAnalysis', 'runDemo']) {
    $(`#${id}`).disabled = busy;
  }
  $('#pageNumber').disabled = busy || selectedFile?.type !== 'application/pdf';
  $('#runAnalysis').disabled = busy || !selectedFile;
}

function resetResult() {
  analysis = null;
  for (const id of ['reasonAnswer', 'reasonTrace', 'reasonJson']) {
    const element = document.getElementById(id);
    if (element) element.textContent = '';
  }
  const questionStatus = document.getElementById('reasonStatus');
  if (questionStatus) questionStatus.textContent = 'Ask a question about the selected drawing.';
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
  $('#reviewMessage').textContent = 'L’avis apparaît ici après la revue d’une zone en désaccord.';
  $('#evidenceList').replaceChildren();
  $('#jsonPanel').hidden = true;
  $('#jsonButton').disabled = true;
}

function setFile(file) {
  if (busy || !file) return;
  if (!accepted.includes(file.type)) return toast('Choisissez un PDF, PNG ou JPEG.');
  if (file.size > 10 * 1024 * 1024) return toast('Le fichier dépasse la limite de 10 Mo.');
  selectedFile = file;
  $('#fileName').textContent = file.name;
  $('#fileSize').textContent = `${Math.max(1, Math.round(file.size / 1024))} Ko`;
  $('#fileType').textContent = file.type.split('/')[1].toUpperCase();
  $('#fileRow').hidden = false;
  dropzone.hidden = true;
  $('#pageNumber').value = 1;
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
  $('#geminiPermission').checked = false;
  drawing.innerHTML = initialDrawing;
  setPreviewLabel('PLAN D’EXEMPLE', '');
  resetResult();
  updateControls();
}

function setPreviewLabel(label, details) {
  const line = $('.preview-topline');
  line.querySelector('span:first-child').textContent = label;
  line.querySelector('span:last-child').textContent = details;
}

function renderEvidence(result) {
  const { width, height, page_index: index } = result.source;
  const wrapper = document.createElement('div');
  wrapper.className = 'evidence-canvas';
  wrapper.style.aspectRatio = `${width} / ${height}`;
  wrapper.style.width = `min(100%, ${220 * width / height}px)`;
  const image = document.createElement('img');
  image.src = result.image_url;
  image.alt = `Plan source, page ${index + 1}`;
  wrapper.append(image);

  const primary = result.disputed_evidence.length ? result.disputed_evidence : result.unverified_evidence;
  const evidence = primary.length ? primary : result.matched_evidence;
  const list = $('#evidenceList');
  list.replaceChildren();
  for (const [position, item] of evidence.entries()) {
    const [left, top, right, bottom] = item.bbox_xyxy;
    const marker = document.createElement('span');
    marker.className = 'evidence-marker';
    marker.hidden = position !== 0;
    Object.assign(marker.style, {
      left: `${left / width * 100}%`, top: `${top / height * 100}%`,
      width: `${(right - left) / width * 100}%`, height: `${(bottom - top) / height * 100}%`,
    });
    wrapper.append(marker);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'text-button';
    button.textContent = `Zone ${position + 1} · ${counts[item.provider]}`;
    button.setAttribute('aria-pressed', String(position === 0));
    button.addEventListener('click', () => {
      wrapper.querySelectorAll('.evidence-marker').forEach((box) => { box.hidden = box !== marker; });
      list.querySelectorAll('button').forEach((other) => other.setAttribute('aria-pressed', String(other === button)));
    });
    list.append(button);
  }
  drawing.replaceChildren(wrapper);
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
  return analysis;
}

function render(result, fileName) {
  analysis = result;
  const supported = result.status === 'SUPPORTED';
  const labels = { SUPPORTED: 'Concordant', NEEDS_REVIEW: 'À vérifier', INSUFFICIENT_EVIDENCE: 'Preuves insuffisantes' };
  $('#decisionBadge').textContent = labels[result.status] || result.status;
  $('#decisionBadge').className = `decision-badge ${supported ? 'consistent' : 'review'}`;
  $('#decisionSymbol').textContent = supported ? '✓' : '!';
  $('#decisionTitle').textContent = supported ? 'Les détections concordent' : 'Total de portes indéterminé';
  $('#decisionCopy').textContent = result.reasons.map((reason) => reasons[reason] || reason).join(' ')
    || 'Les deux moteurs couvrent la page et les positions correspondent.';
  $('#metricRooms').textContent = result.answer ?? '—';
  $('#metricArea').textContent = result.matched_pairs.length;
  $('#metricIssues').textContent = result.disputed_evidence.length + result.unverified_evidence.length;
  for (const [key, id] of [['deterministic', 'detValue'], ['ml', 'mlValue']]) {
    const number = result[key].detected_count;
    $(`#${id}`).textContent = number == null ? 'Indisponible' : `${number} ${number === 1 ? 'proposition' : 'propositions'}`;
  }
  $('#assistantContextTitle').textContent = fileName;
  $('#assistantPage').textContent = `Page ${result.source.page_index + 1}`;
  const advisory = result.gemini.advisory;
  const reviewStates = {
    not_needed: 'Aucun appel Gemini pour ce résultat.',
    not_authorized: 'Analyse Gemini non autorisée pour cette demande.',
    disabled: 'Gemini est désactivé.',
    outside_scope: 'La zone de désaccord est trop large pour la revue.',
    unavailable: 'La revue Gemini est indisponible.',
  };
  $('#reviewMessage').textContent = advisory
    ? `${advisory.reason} ${advisory.evidence_ids.length ? `Références : ${advisory.evidence_ids.join(', ')}.` : ''}`
    : reviewStates[result.gemini.status];
  renderEvidence(result);
  $('#jsonOutput').textContent = JSON.stringify(jsonForTab(), null, 2);
  $('#jsonButton').disabled = false;
}

async function analyze(demo = false) {
  if (busy || (!demo && !selectedFile)) return;
  busy = true;
  updateControls();
  const permission = $('#geminiPermission').checked;
  $('#runAnalysis').textContent = 'Analyse…';
  try {
    let path = '/api/v1/demo/door-count';
    const params = new URLSearchParams({ allow_gemini_review: String(permission) });
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
  } catch (error) {
    resetResult();
    toast(error.message);
  } finally {
    busy = false;
    $('#runAnalysis').textContent = 'Analyser le plan →';
    updateControls();
  }
}

async function readConfig() {
  try {
    const [config, agent] = await Promise.all([api('/api/v1/engines'), api('/api/v1/agent')]);
    $('#modeLabel').textContent = config.deterministic || config.ml ? 'Moteurs configurés' : 'En attente des moteurs';
    $('#detState').textContent = config.deterministic ? 'Configuré' : 'Indisponible';
    $('#mlState').textContent = config.ml ? 'Configuré' : 'Indisponible';
    $('#geminiState').textContent = config.gemini || agent.configured ? 'Configured' : 'Not configured';
    $('#geminiPermission').disabled = !config.gemini;
  } catch {
    $('#modeLabel').textContent = 'API indisponible';
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
$('#jsonButton').addEventListener('click', () => { $('#jsonPanel').hidden = !$('#jsonPanel').hidden; });
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
// Question-aware coordinator. The earlier fixed pipeline remains a comparison endpoint.
counts.gemini = 'Gemini evidence';
const reasoningPanel = document.createElement('section');
reasoningPanel.className = 'reasoning-panel';
reasoningPanel.innerHTML = `<h3>Ask about this drawing</h3>
  <p>Gemini reads the question and chooses whether to answer directly, inspect details, or request structural evidence.</p>
  <label for="reasonQuestion">Question about the selected page</label>
  <textarea id="reasonQuestion" rows="2" maxlength="2000" placeholder="How many doors are shown on this page?"></textarea>
  <label for="reasonMode">Reasoning mode</label>
  <select id="reasonMode"><option value="auto">Agent chooses useful tools</option><option value="visual_only">Visual inspection and feedback, without algorithms</option><option value="direct">Direct Gemini answer, without tools</option></select>
  <label class="reason-permission"><input id="reasonPermission" type="checkbox"> Allow Gemini to inspect this drawing</label>
  <label class="reason-permission"><input id="specialistPermission" type="checkbox"> Also allow Nemotron detail inspection and feedback</label>
  <button id="reasonButton" class="button button-primary" type="button">Ask Gemini</button>
  <p id="reasonStatus" role="status" aria-live="polite">Upload a drawing to begin.</p>
  <p id="agentConfiguration"></p>
  <div id="reasonAnswer"></div>
  <div id="reasonTrace"></div>
  <details><summary>Tool calls and evidence JSON</summary><pre id="reasonJson"></pre></details>`;
$('#runAnalysis').closest('.plan-actions').after(reasoningPanel);
$('#pageNumber').addEventListener('change', () => {
  resetResult();
  drawing.innerHTML = initialDrawing;
  setPreviewLabel('Selected page changed', 'Ask again to inspect this page.');
});
api('/api/v1/agent').then(config => {
  $('#agentConfiguration').textContent = config.configured ? `Configured model: ${config.model}` : 'Gemini configuration is still needed on the server.';
}).catch(() => { $('#agentConfiguration').textContent = 'Agent status unavailable.'; });
let reasoningFile, reasoningPage, reasoningDocument;
$('#reasonButton').addEventListener('click', async () => {
  if (busy) return;
  if (!selectedFile) { toast('Choose a drawing first.'); return; }
  const question = $('#reasonQuestion').value.trim();
  if (!question) { $('#reasonStatus').textContent = 'Enter a question first.'; return; }
  if (!$('#reasonPermission').checked) { $('#reasonStatus').textContent = 'Enable drawing inspection to call Gemini.'; return; }
  busy = true;
  updateControls();
  $('#reasonButton').disabled = true;
  $('#reasonStatus').textContent = 'Gemini is inspecting the question and drawing. Tool-assisted questions may take a few minutes.';
  $('#reasonAnswer').textContent = '';
  $('#reasonTrace').textContent = '';
  $('#reasonJson').textContent = '';
  drawing.querySelectorAll('.evidence-marker').forEach(marker => marker.remove());
  $('#evidenceList').replaceChildren();
  try {
    const page = $('#pageNumber').value || '1';
    if (reasoningFile !== selectedFile || reasoningPage !== page) {
      const form = new FormData(); form.append('file', selectedFile); form.append('page', page);
      reasoningDocument = await api('/api/documents', {method:'POST', body:form});
      reasoningFile = selectedFile; reasoningPage = page;
    }
    const result = await api(`/api/v1/documents/${reasoningDocument.document_id}/ask`, {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question,allow_gemini:true,allow_nemotron:$('#specialistPermission').checked,mode:$('#reasonMode').value})
    });
    $('#reasonStatus').textContent = result.status === 'ok'
      ? `Answer ready · ${result.algorithm_used ? 'structural evidence used' : 'visual reasoning'} · ${result.nemotron_calls} Nemotron calls · ${result.seconds}s`
      : result.status === 'not_configured' ? 'Gemini key/model are not configured on the server yet.' : `The agent could not produce a validated answer (${result.status}).`;
    if (result.reliability?.recovery_succeeded) $('#reasonStatus').textContent += ' · Recovered after an interrupted model response.';
    if (result.status !== 'ok' && result.reliability?.last_error) {
      const issue = result.reliability.last_error;
      const wait = result.reliability.retry_after_seconds;
      $('#reasonStatus').textContent += result.reliability.retryable
        ? ` The model service was interrupted (${issue.category}). ${wait ? `Wait at least ${Math.ceil(wait)} seconds before retrying.` : 'You can retry when the service is available.'} No answer is asserted.`
        : ` The request stopped (${issue.category}). No answer is asserted.`;
    }
    $('#reasonAnswer').textContent = result.answer?.status === 'answered'
      ? `${result.answer.answer} ${result.answer.unit || ''}\n${result.answer.explanation}`
      : result.answer?.explanation || 'No answer is asserted.';
    const trace = (result.tool_trace || []).map(t => `${t.name} · ${t.result.status === 'unavailable' ? 'unavailable' : 'returned evidence'} · ${t.seconds}s`);
    const attempts = result.reliability?.provider_attempts || [];
    const recoveryTrace = attempts.map(a => `Gemini attempt ${a.attempt} · ${a.phase} · ${a.status || a.category}${a.http_status ? ` HTTP ${a.http_status}` : ''}${a.finish_reason ? ` ${a.finish_reason}` : ''} · ${a.seconds}s`);
    $('#reasonTrace').textContent = [result.answer?.coverage, ...trace, ...recoveryTrace].filter(Boolean).join('\n');
    $('#reasonJson').textContent = JSON.stringify(result,null,2);
    const citations = (result.canonical_citations || []).map((c,i)=>({id:`citation-${i+1}`,provider:'gemini',bbox_xyxy:c.canonical_bbox}));
    renderEvidence({...result,mode:'executed_engines',matched_evidence:citations,disputed_evidence:[],unverified_evidence:[]});
  } catch (error) {
    $('#reasonStatus').textContent = error.message;
  } finally {
    busy = false; updateControls(); $('#reasonButton').disabled = false;
  }
});
