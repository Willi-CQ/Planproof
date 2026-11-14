const $ = (selector) => document.querySelector(selector);
const fileInput = $('#fileInput');
const dropzone = $('#dropzone');
const fileRow = $('#fileRow');
const planPreview = $('#planPreview');
const toast = $('#toast');
let selectedFile = null;
let analysis = null;
let currentJsonTab = 'cross';
let objectUrl = null;
let config = { mode: 'demo', gemini: false, deterministic: false, ml: false, jev: false };

const demoData = {
  gemini: {
    plan_id: 'demo-t3-001', plan_type: 'appartement', scale: '1:100',
    rooms: [
      { id: 'room-01', label: 'Séjour', area_m2: 24.8, polygon: [[0,0],[5.1,0],[5.1,4.86],[0,4.86]] },
      { id: 'room-02', label: 'Chambre 01', area_m2: 12.4, polygon: [[5.1,0],[8.2,0],[8.2,4],[5.1,4]] },
      { id: 'room-03', label: 'Chambre 02', area_m2: 10.2, polygon: [[0,4.86],[3.6,4.86],[3.6,7.7],[0,7.7]] },
      { id: 'room-04', label: 'Cuisine', area_m2: 8.1, polygon: [[5.1,5.4],[8.2,5.4],[8.2,8],[5.1,8]] },
      { id: 'room-05', label: "Salle d'eau", area_m2: 4.7, polygon: [[5.1,4],[8.2,4],[8.2,5.4],[5.1,5.4]] }
    ],
    objects: { walls: 14, doors: 5, windows: 5 }, total_area_m2: 68.2,
    uncertainties: ['Nombre de fenêtres à confirmer : une ouverture est partiellement masquée.']
  },
  deterministic: {
    source: 'deterministic-demo', schema_version: '1.0', room_count: 5,
    area_m2: 68.2, counts: { walls: 14, doors: 5, windows: 4 },
    coordinates: { unit: 'm', origin: 'top_left', x_axis: 'right', y_axis: 'down' },
    confidence: 0.97
  },
  ml: {
    source: 'ml-demo', model_version: '0.0-demo', room_count: 5,
    area_m2: 67.8, counts: { walls: 14, doors: 5, windows: 5 },
    detections: [
      { label: 'window', x: 1.55, y: 0, confidence: 0.98 },
      { label: 'window', x: 7.1, y: 0, confidence: 0.94 },
      { label: 'window', x: 8.2, y: 2.1, confidence: 0.91 },
      { label: 'window', x: 6.8, y: 8, confidence: 0.87 },
      { label: 'window', x: 0, y: 6.2, confidence: 0.72 }
    ], confidence: 0.89
  }
};

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove('show'), 2600);
}

function prettySize(bytes) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} Ko` : `${(bytes / 1024 / 1024).toFixed(1).replace('.', ',')} Mo`;
}

function setFile(file) {
  if (!file) return;
  const accepted = ['application/pdf', 'image/png', 'image/jpeg', 'image/webp', 'image/bmp', 'image/tiff'];
  if (!accepted.includes(file.type)) {
    showToast('Format non pris en charge. Choisissez un PDF ou une image.');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showToast('Le fichier dépasse la limite de 20 Mo.');
    return;
  }
  selectedFile = file;
  $('#fileName').textContent = file.name;
  $('#fileSize').textContent = `${prettySize(file.size)} · prêt à analyser`;
  $('#fileType').textContent = file.type === 'application/pdf' ? 'PDF' : file.type.split('/')[1].toUpperCase();
  fileRow.hidden = false;
  dropzone.hidden = true;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  if (file.type.startsWith('image/')) {
    $('#drawingWrap').innerHTML = '';
    const image = document.createElement('img');
    image.src = objectUrl;
    image.alt = `Aperçu de ${file.name}`;
    image.className = 'uploaded-image';
    $('#drawingWrap').append(image);
    $('#drawingWrap').classList.add('uploaded-preview');
  } else {
    $('#drawingWrap').innerHTML = `<div class="pdf-placeholder"><span>PDF</span><b>${escapeHtml(file.name)}</b><small>Aperçu disponible après analyse</small></div>`;
    $('#drawingWrap').classList.add('uploaded-preview');
  }
  $('.preview-topline span:first-child').innerHTML = '<i class="tiny-dot"></i> FICHIER IMPORTÉ';
  $('.preview-topline span:last-child').textContent = file.type === 'application/pdf' ? 'DOCUMENT PDF' : 'IMAGE';
  $('#assistantContextTitle').textContent = file.name;
  analysis = null;
  showToast('Plan importé. Lancez l’analyse pour continuer.');
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
}

function resetApp() {
  selectedFile = null;
  analysis = null;
  fileInput.value = '';
  fileRow.hidden = true;
  dropzone.hidden = false;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  $('#drawingWrap').classList.remove('uploaded-preview');
  $('#drawingWrap').innerHTML = initialPlanDrawing;
  $('.preview-topline span:first-child').innerHTML = '<i class="tiny-dot"></i> PLAN D’EXEMPLE';
  $('.preview-topline span:last-child').textContent = 'ÉCHELLE 1:100';
  resetResultView();
  showToast('Nouvelle analyse prête.');
}

const initialPlanDrawing = $('#drawingWrap').innerHTML;

async function readConfig() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) return;
    config = await response.json();
    updateIntegrationStatus();
  } catch {
    // The static preview remains usable in demo mode if the server is not running.
  }
}

function updateIntegrationStatus() {
  const real = config.mode === 'live';
  $('#modeLabel').textContent = real ? 'Connecteurs actifs' : 'Mode démo';
  $('#geminiState').textContent = config.gemini ? 'Connecté' : 'Démo';
  $('#detState').textContent = config.deterministic ? 'Connecté' : 'Démo';
  $('#mlState').textContent = config.ml ? 'Connecté' : 'Démo';
  $('#jevState').textContent = config.jev ? 'Connecté' : 'Optionnel';
  document.querySelector('.online-dot').style.background = config.gemini ? '#78a480' : '#cf9c55';
}

async function fileAsBase64(file) {
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
  return String(dataUrl).split(',')[1];
}

async function runAnalysis() {
  const button = $('#runAnalysis');
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> Analyse…';
  document.body.classList.add('loading-state');
  try {
    if (config.mode === 'live') {
      if (!selectedFile) throw new Error('Importez un PDF ou une image avant de lancer une analyse réelle.');
      const data = await fileAsBase64(selectedFile);
      const response = await fetch('/api/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file: { name: selectedFile.name, mimeType: selectedFile.type, data } })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Échec de l’analyse.');
      analysis = payload;
    } else {
      await new Promise((resolve) => setTimeout(resolve, 850));
      analysis = demoData;
      analysis.crossCheck = crossCheck(demoData.deterministic, demoData.ml, demoData.gemini);
      analysis.mode = 'demo';
    }
    renderAnalysis(analysis);
    showToast(config.mode === 'live' ? 'Analyse terminée.' : 'Simulation terminée · résultats de démonstration.');
  } catch (error) {
    showToast(error.message || 'Une erreur est survenue.');
  } finally {
    document.body.classList.remove('loading-state');
    button.disabled = false;
    button.innerHTML = 'Analyser le plan <span class="button-arrow">→</span>';
  }
}

function getRoomsCount(gemini, det, ml) {
  return Number(gemini?.rooms?.length ?? det?.room_count ?? ml?.room_count ?? 0);
}

function getArea(gemini, det, ml) {
  const number = Number(gemini?.total_area_m2 ?? det?.area_m2 ?? ml?.area_m2 ?? 0);
  return number.toLocaleString('fr-FR', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
}

function crossCheck(det, ml, gemini) {
  const discrepancies = [];
  const fields = [
    ['room_count', 'Nombre de pièces', det?.room_count, ml?.room_count],
    ['area_m2', 'Surface', det?.area_m2, ml?.area_m2],
    ['windows', 'Fenêtres', det?.counts?.windows, ml?.counts?.windows],
    ['doors', 'Portes', det?.counts?.doors, ml?.counts?.doors],
    ['walls', 'Murs', det?.counts?.walls, ml?.counts?.walls]
  ];
  for (const [key, label, left, right] of fields) {
    if (left == null || right == null) continue;
    const differs = key === 'area_m2' ? Math.abs(Number(left) - Number(right)) > 1 : Number(left) !== Number(right);
    if (differs) discrepancies.push({ field: key, label, deterministic: left, ml: right });
  }
  const unresolved = discrepancies.length > 0 || (gemini?.uncertainties?.length || 0) > 0;
  return { decision: unresolved ? 'review_required' : 'consistent', discrepancies, reason: unresolved ? 'Les résultats divergent ou comportent une incertitude à confirmer.' : 'Les deux analyses sont concordantes.' };
}

function renderAnalysis(result) {
  const det = result.deterministic || {};
  const ml = result.ml || {};
  const gemini = result.gemini || {};
  const cross = result.crossCheck || crossCheck(det, ml, gemini);
  const diff = cross.discrepancies || [];
  const needsReview = cross.decision !== 'consistent' && cross.decision !== 'accepted';
  const windowDiff = diff.find((item) => item.field === 'windows');
  $('#decisionBadge').className = `decision-badge ${needsReview ? 'review' : 'consistent'}`;
  $('#decisionBadge').innerHTML = `<span>${needsReview ? '!' : '✓'}</span> ${needsReview ? 'À vérifier' : 'Concordant'}`;
  $('#decisionTitle').textContent = result.mode === 'demo'
    ? 'Écart simulé sur les fenêtres'
    : needsReview ? 'Un écart détecté entre les analyses' : 'Les analyses sont concordantes';
  const decisionText = needsReview
    ? (windowDiff ? `Écart sur les fenêtres : ${windowDiff.deterministic} pour le déterministe, ${windowDiff.ml} pour le modèle ML.` : cross.reason || 'Une validation humaine est recommandée.')
    : 'Les mesures et détections des deux algorithmes correspondent.';
  $('#decisionCopy').textContent = result.mode === 'demo'
    ? `${decisionText} Simulation : le fichier choisi n’a pas été analysé.`
    : decisionText;
  $('#decisionCopy').parentElement.parentElement.style.background = needsReview ? '#fffaf2' : '#f3f8f3';
  $('#decisionCopy').parentElement.parentElement.style.borderColor = needsReview ? '#f2e7d4' : '#e0ecdf';
  $('#decisionSymbol').textContent = needsReview ? '!' : '✓';
  $('#decisionSymbol').style.background = needsReview ? '#f5e4c9' : '#e2efe4';
  $('#decisionSymbol').style.color = needsReview ? '#ae773f' : '#4f815b';
  $('#metricRooms').textContent = getRoomsCount(gemini, det, ml) || '—';
  $('#metricArea').innerHTML = `${getArea(gemini, det, ml) || '—'} <em>m²</em>`;
  $('#metricIssues').textContent = diff.length + (gemini.uncertainties?.length || 0);
  $('#detValue').textContent = `${det.counts?.windows ?? '—'} fenêtres`;
  $('#mlValue').textContent = `${ml.counts?.windows ?? '—'} fenêtres`;
  $('#detValue').style.color = windowDiff ? '#a36b36' : '';
  $('#mlValue').style.color = windowDiff ? '#a36b36' : '';
  const resultChecks = document.querySelectorAll('.algo-result .result-check');
  resultChecks[0].className = 'result-check ok'; resultChecks[0].textContent = '✓';
  resultChecks[1].className = `result-check ${diff.length ? 'warn' : 'ok'}`; resultChecks[1].textContent = diff.length ? '!' : '✓';
  $('#jsonOutput').textContent = JSON.stringify(jsonForTab(currentJsonTab), null, 2);
  updateChatIntro(windowDiff, needsReview);
}

function jsonForTab(tab) {
  if (!analysis) return {};
  if (tab === 'det') return analysis.deterministic;
  if (tab === 'ml') return analysis.ml;
  return analysis.crossCheck || crossCheck(analysis.deterministic, analysis.ml, analysis.gemini);
}

function updateChatIntro(windowDiff, needsReview) {
  const messages = $('#chatBody');
  const paragraphs = messages.querySelectorAll('.assistant-message .message-content p');
  if (paragraphs.length >= 2) {
    paragraphs[0].textContent = analysis?.mode === 'demo'
      ? 'Voici une simulation avec les données d’exemple intégrées.'
      : 'Bonjour Johan ! J’ai comparé les deux analyses du plan.';
    paragraphs[1].textContent = needsReview
      ? `Un écart apparaît${windowDiff ? ` sur les fenêtres : le modèle ML en détecte ${windowDiff.ml}, tandis que l’algorithme déterministe en compte ${windowDiff.deterministic}` : ' dans les résultats'}.`
      : 'Les deux algorithmes sont concordants sur les mesures et les objets détectés.';
  }
  const hint = messages.querySelector('.message-hint');
  if (hint) hint.textContent = analysis?.mode === 'demo'
    ? 'Les résultats affichés sont fictifs et ne proviennent pas du fichier importé.'
    : 'Vous pouvez demander une vérification complémentaire avec JEV.';
}

function resetResultView() {
  $('#decisionBadge').className = 'decision-badge review';
  $('#decisionBadge').innerHTML = '<span>!</span> À vérifier';
  $('#decisionTitle').textContent = 'Un écart détecté entre les analyses';
  $('#decisionCopy').textContent = 'Les algorithmes ne sont pas entièrement d’accord sur le nombre d’ouvertures.';
  $('#decisionCopy').parentElement.parentElement.style.background = '#fffaf2';
  $('#decisionCopy').parentElement.parentElement.style.borderColor = '#f2e7d4';
  $('#decisionSymbol').textContent = '!'; $('#decisionSymbol').style.background = ''; $('#decisionSymbol').style.color = '';
  $('#metricRooms').textContent = '5'; $('#metricArea').innerHTML = '68,2 <em>m²</em>'; $('#metricIssues').textContent = '1';
  $('#detValue').textContent = '4 fenêtres'; $('#mlValue').textContent = '5 fenêtres';
  const resultChecks = document.querySelectorAll('.algo-result .result-check');
  resultChecks[0].className = 'result-check ok'; resultChecks[0].textContent = '✓';
  resultChecks[1].className = 'result-check warn'; resultChecks[1].textContent = '!';
  $('#jsonPanel').hidden = true;
  $('#assistantContextTitle').textContent = 'Plan appartement T3';
}

function addChatMessage(text, role, content = text) {
  const body = $('#chatBody');
  const element = document.createElement('div');
  if (role === 'user') {
    element.className = 'user-message';
    const p = document.createElement('p'); p.textContent = text; element.append(p);
  } else {
    element.className = 'assistant-reply';
    const avatar = document.createElement('span'); avatar.className = 'message-avatar'; avatar.textContent = '✳';
    const reply = document.createElement('div'); reply.className = 'reply-text'; reply.textContent = content;
    element.append(avatar, reply);
  }
  body.append(element); body.scrollTop = body.scrollHeight;
}

async function sendChat(question) {
  const clean = question.trim();
  if (!clean) return;
  addChatMessage(clean, 'user');
  $('#chatInput').value = '';
  if (config.mode === 'live') {
    try {
      const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: clean, analysis }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Erreur assistant.');
      addChatMessage(payload.answer, 'assistant');
    } catch (error) { addChatMessage(error.message, 'assistant'); }
  } else {
    const answer = clean.toLowerCase().includes('pièce')
      ? 'Dans cette simulation, les 5 pièces repérées sont le séjour, deux chambres, la cuisine et la salle d’eau. Aucun écart de comptage n’est remonté sur les pièces.'
      : clean.toLowerCase().includes('décision')
        ? 'Le contrôle rapproche les champs comparables des deux JSON (pièces, surface, murs, portes et fenêtres). Un écart dépasse le seuil configuré : la décision devient « à vérifier » et garde les deux valeurs pour examen humain.'
        : 'La simulation montre un désaccord sur les fenêtres (4 contre 5). La décision reste « à vérifier » jusqu’à validation. Connectez Gemini pour poser des questions sur votre plan réel.';
    setTimeout(() => addChatMessage(answer, 'assistant'), 220);
  }
}

async function requestJev() {
  if (!config.jev) {
    $('#modalTitle').textContent = 'JEV n’est pas configuré';
    $('#modalBody').textContent = 'Ajoutez JEV_ENDPOINT et JEV_TOKEN dans le fichier .env pour activer cette vérification complémentaire.';
    $('#modalBackdrop').hidden = false;
    return;
  }
  $('#jevButton').disabled = true;
  $('#jevButton').innerHTML = '<span class="spinner"></span> Vérification…';
  try {
    const response = await fetch('/api/jev', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ analysis }) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'La vérification JEV a échoué.');
    analysis.jev = payload.result;
    $('#jsonPanel').hidden = false;
    currentJsonTab = 'cross';
    $('#jsonOutput').textContent = JSON.stringify({ crossCheck: analysis.crossCheck, jev: analysis.jev }, null, 2);
    showToast('Résultat JEV ajouté au dossier.');
  } catch (error) { showToast(error.message); }
  finally { $('#jevButton').disabled = false; $('#jevButton').innerHTML = '<span>✳</span> Compléter avec JEV <span class="button-arrow">↗</span>'; }
}

fileInput.addEventListener('change', () => setFile(fileInput.files?.[0]));
dropzone.addEventListener('click', () => fileInput.click());
$('#replaceFile').addEventListener('click', () => fileInput.click());
$('#removeFile').addEventListener('click', () => { selectedFile = null; fileInput.value = ''; resetApp(); });
$('#runAnalysis').addEventListener('click', runAnalysis);
$('#newAnalysis').addEventListener('click', resetApp);
$('#jevButton').addEventListener('click', requestJev);
$('#jsonButton').addEventListener('click', () => { $('#jsonPanel').hidden = !$('#jsonPanel').hidden; if (!$('#jsonPanel').hidden) { if (!analysis) analysis = { ...demoData, crossCheck: crossCheck(demoData.deterministic, demoData.ml, demoData.gemini), mode: 'demo' }; $('#jsonOutput').textContent = JSON.stringify(jsonForTab(currentJsonTab), null, 2); } });
document.querySelectorAll('.json-tab').forEach((button) => button.addEventListener('click', () => {
  document.querySelectorAll('.json-tab').forEach((tab) => tab.classList.toggle('active', tab === button));
  currentJsonTab = button.dataset.json;
  $('#jsonOutput').textContent = JSON.stringify(jsonForTab(currentJsonTab), null, 2);
}));
$('#copyJson').addEventListener('click', async () => { try { await navigator.clipboard.writeText($('#jsonOutput').textContent); showToast('JSON copié.'); } catch { showToast('Copie indisponible dans ce navigateur.'); } });
$('#chatForm').addEventListener('submit', (event) => { event.preventDefault(); sendChat($('#chatInput').value); });
$('.suggestions').addEventListener('click', (event) => { const button = event.target.closest('.suggestion-chip'); if (button) sendChat(button.textContent); });
$('#modalClose').addEventListener('click', () => { $('#modalBackdrop').hidden = true; });
$('#modalAction').addEventListener('click', () => { $('#modalBackdrop').hidden = true; });
$('#modalBackdrop').addEventListener('click', (event) => { if (event.target === $('#modalBackdrop')) $('#modalBackdrop').hidden = true; });
$('#dropzone').addEventListener('dragover', (event) => { event.preventDefault(); dropzone.classList.add('dragover'); });
$('#dropzone').addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
$('#dropzone').addEventListener('drop', (event) => { event.preventDefault(); dropzone.classList.remove('dragover'); setFile(event.dataTransfer.files?.[0]); });
$('#assistantContextTitle').textContent = 'Plan appartement T3';
readConfig();
