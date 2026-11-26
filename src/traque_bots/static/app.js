const form = document.querySelector("#analysis-form");
const fileInput = document.querySelector("#csv-file");
const uploadButton = document.querySelector("#upload-button");
const demoButton = document.querySelector("#demo-button");
const statusMessage = document.querySelector("#status-message");

let currentPredictions = [];
let currentAnalysis = null;
let currentLabels = {};
let currentPage = 1;
const pageSize = 50;

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  document.querySelector("#file-name").textContent = file ? file.name : "Choisir un fichier CSV";
  uploadButton.disabled = !file;
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (fileInput.files[0]) runAnalysis(new FormData(form));
});

demoButton.addEventListener("click", () => runAnalysis(new FormData()));
document.querySelector("#search-input").addEventListener("input", () => {
  currentPage = 1;
  renderTable();
});
document.querySelector("#export-button").addEventListener("click", exportVisibleResults);
document.querySelector("#previous-page").addEventListener("click", () => {
  currentPage = Math.max(1, currentPage - 1);
  renderTable();
});
document.querySelector("#next-page").addEventListener("click", () => {
  currentPage += 1;
  renderTable();
});

async function runAnalysis(payload) {
  setLoading(true);
  setStatus("Analyse en cours — entraînement et évaluation sur des données mises à part.", false);
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: payload });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "L'analyse n'a pas pu aboutir.");
    currentPredictions = result.predictions;
    currentAnalysis = result.analysis;
    currentLabels = currentAnalysis.class_labels;
    currentPage = 1;
    document.querySelector("#search-input").value = "";
    renderDashboard(result);
    setStatus(
      result.source === "synthetic"
        ? "Démonstration terminée. Résultats issus uniquement de données synthétiques."
        : currentAnalysis.completion_message,
      false,
    );
    document.querySelector("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setLoading(false);
  }
}

function renderDashboard(result) {
  const { summary, analysis } = result;
  document.querySelector("#results-title").textContent = analysis.title;
  document.querySelector("#metric-entities-label").textContent = `${analysis.entity_name}s évalués`;
  document.querySelector("#metric-accounts").textContent = formatNumber(summary.evaluated_accounts);
  document.querySelector("#metric-accounts-note").textContent =
    `${formatNumber(summary.train_accounts)} ${analysis.unit_plural} utilisés pour l'entraînement`;
  document.querySelector("#metric-accuracy").textContent = formatPercent(summary.accuracy);
  document.querySelector("#metric-f1").textContent = formatPercent(summary.macro_f1);
  document.querySelector("#metric-highlight-label").textContent = analysis.metric_highlight_label;
  document.querySelector("#metric-review").textContent = formatNumber(summary.highlighted_count);
  document.querySelector("#metric-highlight-note").textContent = analysis.metric_highlight_note;

  const sourceNames = {
    synthetic: "DÉMO SYNTHÉTIQUE",
    normalized: "JEU NORMALISÉ",
    bot_dataset: "BOT DETECTION",
    fake_news_dataset: "FAKE NEWS",
  };
  const runSource = document.querySelector("#run-source");
  runSource.classList.add("ready");
  runSource.innerHTML = `<span class="status-dot"></span>${sourceNames[result.source] || "JEU IMPORTÉ"}`;

  document.querySelector("#signals-title").textContent = analysis.signal_title;
  document.querySelector("#signals-description").textContent =
    `Scores moyens sur les ${analysis.unit_plural} évalués`;
  document.querySelector("#distribution-title").textContent = "Répartition des classes prédites";
  document.querySelector("#guard-title").textContent = analysis.guard_title;
  document.querySelector("#guard-description").textContent = analysis.guard_description;
  document.querySelector("#table-title").textContent = analysis.table_title;
  document.querySelector("#table-disclaimer").textContent = analysis.guard_description;

  renderSignals(summary.average_scores, analysis.signal_definitions);
  renderRecommendations(summary.recommendations, summary.evaluated_accounts);
  renderModelComparison(summary.model_comparison || [], analysis);
  renderTableColumns();
  const displayed = summary.displayed_accounts;
  document.querySelector("#table-caption").textContent = displayed < summary.evaluated_accounts
    ? `Affichage des ${formatNumber(displayed)} premiers sur ${formatNumber(summary.evaluated_accounts)} ${analysis.unit_plural}`
    : `${formatNumber(summary.evaluated_accounts)} ${analysis.unit_plural} évalués`;
  document.querySelector("#search-input").disabled = false;
  document.querySelector("#export-button").disabled = currentPredictions.length === 0;
  renderTable();
}

function renderModelComparison(models, analysis) {
  const panel = document.querySelector("#model-comparison-panel");
  panel.hidden = models.length === 0;
  if (models.length === 0) return;
  document.querySelector("#model-comparison-body").innerHTML = models.map((model) => {
    const selected = model.name === analysis.selected_model;
    return `<tr><td class="account-id">${escapeHtml(model.name)}${selected ? " · retenu après validation" : ""}</td><td>${formatPercent(model.accuracy)}</td><td>${formatPercent(model.macro_f1)}</td></tr>`;
  }).join("");
  document.querySelector("#model-note").textContent = analysis.model_note;
}

function renderSignals(scores, definitions) {
  document.querySelector("#signal-bars").innerHTML = definitions.map(({ key, label }) => {
    const value = Number(scores[key]) || 0;
    return `<div class="signal-row"><span class="signal-label">${escapeHtml(label)}</span><div class="signal-track" role="meter" aria-label="${escapeHtml(label)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(value * 100)}"><div class="signal-fill" style="width:${Math.max(1, value * 100)}%"></div></div><span class="signal-value">${formatPercent(value)}</span></div>`;
  }).join("");
}

function renderRecommendations(counts, total) {
  document.querySelector("#recommendation-list").innerHTML = Object.entries(counts).map(([key, count]) => {
    const share = total ? count / total : 0;
    const label = currentLabels[key] || key;
    return `<div class="recommendation-row ${escapeHtml(key)}"><span class="recommendation-marker"></span><span>${escapeHtml(label)}<small class="distribution-share"> · ${formatPercent(share)}</small></span><span class="recommendation-count">${formatNumber(count)}</span></div>`;
  }).join("");
}

function renderTableColumns() {
  const headers = [
    currentAnalysis.entity_name.toUpperCase(),
    ...currentAnalysis.signal_definitions.map(({ label }) => label.toUpperCase()),
    currentAnalysis.prediction_column,
    currentAnalysis.reference_column,
  ];
  document.querySelector("#table-columns").innerHTML = headers
    .map((heading) => `<th>${escapeHtml(heading)}</th>`)
    .join("");
}

function renderTable() {
  const query = document.querySelector("#search-input").value.trim().toLowerCase();
  const filtered = currentPredictions.filter((row) => row.entity_id.toLowerCase().includes(query));
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  currentPage = Math.min(currentPage, pageCount);
  const start = (currentPage - 1) * pageSize;
  const visible = filtered.slice(start, start + pageSize);
  const body = document.querySelector("#results-body");
  const columnCount = currentAnalysis
    ? currentAnalysis.signal_definitions.length + 3
    : 7;
  if (visible.length === 0) {
    body.innerHTML = `<tr class="placeholder-row"><td colspan="${columnCount}">${currentPredictions.length ? "Aucun résultat ne correspond à cette recherche." : "Aucun résultat à afficher. Lancez une analyse pour commencer."}</td></tr>`;
  } else {
    body.innerHTML = visible.map((row) => {
      const prediction = escapeHtml(row.recommendation);
      const scores = currentAnalysis.signal_definitions
        .map(({ key }) => scoreCell(row[key]))
        .join("");
      return `<tr><td class="account-id">${escapeHtml(row.entity_id)}</td>${scores}<td><span class="badge ${prediction}">${escapeHtml(currentLabels[row.recommendation] || prediction)}</span></td><td><span class="reference-label">${escapeHtml(currentLabels[row.expected] || row.expected)}</span></td></tr>`;
    }).join("");
  }
  const count = document.querySelector("#result-count");
  count.textContent = `${formatNumber(filtered.length)} ${currentAnalysis.unit_plural} · ${formatNumber(visible.length)} sur cette page`;
  document.querySelector("#page-info").textContent = `Page ${formatNumber(currentPage)} / ${formatNumber(pageCount)}`;
  document.querySelector("#previous-page").disabled = currentPage <= 1;
  document.querySelector("#next-page").disabled = currentPage >= pageCount;
}

function scoreCell(score) {
  const value = Math.min(1, Math.max(0, Number(score) || 0));
  return `<td><div class="score-cell"><div class="mini-track"><div class="mini-fill" style="width:${value * 100}%"></div></div><span class="score-number">${formatPercent(value)}</span></div></td>`;
}

function exportVisibleResults() {
  const query = document.querySelector("#search-input").value.trim().toLowerCase();
  const rows = currentPredictions.filter((row) => row.entity_id.toLowerCase().includes(query));
  if (rows.length === 0) return;
  const columns = [
    "entity_id",
    ...currentAnalysis.signal_definitions.map(({ key }) => key),
    "recommendation",
    "expected",
    "human_review_required",
  ];
  const csv = [
    columns.join(","),
    ...rows.map((row) => columns.map((column) => csvValue(row[column])).join(",")),
  ].join("\r\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" }));
  link.download = "sentinelle-resultats.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

function csvValue(value) {
  const text = value === undefined ? "" : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

function formatNumber(value) {
  return new Intl.NumberFormat("fr-FR").format(value);
}

function formatPercent(value) {
  return new Intl.NumberFormat("fr-FR", { style: "percent", maximumFractionDigits: 1 }).format(value);
}

function setStatus(message, isError) {
  statusMessage.hidden = false;
  statusMessage.classList.toggle("error", isError);
  statusMessage.textContent = message;
}

function setLoading(isLoading) {
  demoButton.disabled = isLoading;
  uploadButton.disabled = isLoading || !fileInput.files[0];
  demoButton.innerHTML = isLoading ? "Analyse en cours…" : '<span class="play-icon">▶</span>Lancer la démo';
  demoButton.setAttribute("aria-busy", String(isLoading));
}
