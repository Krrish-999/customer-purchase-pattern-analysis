/* Supermarket Data Mining Platform - Frontend Client Logic
   Supports: Data Preprocessing, Classification, Regression, Clustering, and Association Rule Mining.
*/
const API = "";
const COLORS = ["#3b82f6", "#8b5cf6", "#10b981", "#14b8a6", "#f59e0b", "#ec4899", "#0ea5e9", "#a855f7", "#22c55e", "#f97316", "#6366f1", "#06b6d4", "#e11d48", "#84cc16", "#d946ef"];
const YEAR_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"];
const charts = {};
const loaded = {};
let cachedColumns = [];
let clusteringMode = "rows";
const comparisonState = {
  userRuns: { classification: {}, regression: {}, clustering: {}, rules: null, preprocessing: null },
  activeBenchmark: null
};

const $ = id => document.getElementById(id);
const nf = n => (n === null || n === undefined) ? "–" : Number(n).toLocaleString();
const pc = (v, d = 1) => (v === null || v === undefined) ? "–" : (v * 100).toFixed(d) + "%";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function toast(msg, bad) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (bad ? " err" : "");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => t.classList.add("hidden"), 4500);
}

async function api(path, opts) {
  const r = await fetch(API + path, opts);
  let j = {};
  try { j = await r.json(); } catch (e) { }
  if (!r.ok) throw new Error(j.error || ("Request failed (" + r.status + ")"));
  return j;
}

function chart(id, cfg) {
  if (charts[id]) charts[id].destroy();
  const base = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { usePointStyle: true, boxWidth: 8 } } }
  };
  cfg.options = Object.assign({}, base, cfg.options || {});
  charts[id] = new Chart($(id), cfg);
}

const gridOpt = { grid: { color: "#e2e8f044" } };
const kpi = (cls, val, label) => `<div class="kpi ${cls}"><b>${val}</b><span>${label}</span></div>`;

/* ---------- Navigation ---------- */
const TITLES = {
  dashboard: ["Dashboard", "Supermarket Data Mining Learning & Analysis Platform"],
  overview: ["Dashboard", "Supermarket Data Mining Learning & Analysis Platform"],
  dataset: ["Dataset Explorer", "Active data source, column schema, and raw data preview"],
  preprocessing: ["Data Preprocessing", "Interactive data cleaning, categorical encoding, scaling, and feature selection"],
  classification: ["Classification", "Supervised classification algorithms, evaluation metrics, and confusion matrix"],
  regression: ["Regression", "Continuous target prediction, polynomial modeling, and error metrics"],
  clustering: ["Clustering", "Unsupervised clustering algorithms and RFM customer segmentation"],
  rules: ["Association Rules", "Apriori frequent itemset mining, support, confidence, and lift"],
  algorithms: ["Rule Mining Theory", "Apriori (implemented) vs DHP & CHARM (conceptual comparison)"],
  insights: ["Business Insights", "Actionable retail recommendations generated from mined patterns"],
  yearly: ["Yearly Sales", "Year-over-year sales trends and seasonality calculated live"],
  comparison: ["Data Mining Comparison & Analysis", "Cross-Technique and Intra-Technique Performance Benchmarking & Evaluation Dashboard"]
};

function show(sec) {
  if (sec === "overview") sec = "dashboard";
  document.querySelectorAll(".sec").forEach(s => s.classList.toggle("active", s.id === "sec-" + sec));
  document.querySelectorAll("#nav button").forEach(b => b.classList.toggle("active", b.dataset.sec === sec));
  if (TITLES[sec]) {
    $("pageTitle").textContent = TITLES[sec][0];
    $("pageSub").textContent = TITLES[sec][1];
  }
  const f = {
    dashboard: loadOverview,
    overview: loadOverview,
    dataset: loadDataset,
    preprocessing: loadPreprocessing,
    classification: loadClassification,
    regression: loadRegression,
    clustering: loadClustering,
    rules: loadRules,
    algorithms: loadAlgorithms,
    insights: loadInsights,
    yearly: loadYearly,
    comparison: loadComparison
  }[sec];
  if (f) {
    if (!loaded[sec]) {
      loaded[sec] = true;
      f().catch(e => { loaded[sec] = false; toast(e.message, true); });
    } else {
      setTimeout(() => { Object.values(charts).forEach(c => { try { c.resize(); } catch(e){} }); }, 50);
    }
  }
}
document.querySelectorAll("#nav button").forEach(b => b.onclick = () => show(b.dataset.sec));

function invalidateDownstream() {
  ["overview", "dashboard", "classification", "regression", "clustering", "rules", "insights", "comparison"].forEach(k => loaded[k] = false);
  cachedColumns = [];
}

/* ---------- Column Fetcher Helper ---------- */
async function getColumns(force = false) {
  if (!force && cachedColumns.length) return cachedColumns;
  try {
    const d = await api("/api/dm/columns");
    cachedColumns = d.columns || [];
    return cachedColumns;
  } catch (e) {
    return [];
  }
}

/* ---------- Dashboard / Overview ---------- */
async function loadOverview() {
  const d = await api("/api/overview"), k = d.kpis;
  $("kpis").innerHTML = kpi("k-blue", nf(k.transactions), "Total Transactions") +
    kpi("k-purple", nf(k.customers), "Unique Customers") +
    kpi("k-green", nf(k.products), "Products") +
    kpi("k-teal", nf(k.categories), "Categories") +
    kpi("k-orange", nf(k.items_sold), "Items Sold") +
    kpi("k-pink", k.avg_basket, "Average Basket Size");

  chart("chTop", {
    type: "bar",
    data: { labels: d.top_items.labels, datasets: [{ label: "Transactions containing item", data: d.top_items.values, backgroundColor: COLORS, borderRadius: 6 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { ticks: { maxRotation: 60, minRotation: 45 } }, y: gridOpt } }
  });
  chart("chSize", {
    type: "bar",
    data: { labels: d.size_distribution.labels, datasets: [{ label: "Transactions", data: d.size_distribution.values, backgroundColor: "#8b5cf6", borderRadius: 6 }] },
    options: { plugins: { legend: { display: false } }, scales: { y: gridOpt } }
  });
  if (d.daily.labels.length) {
    chart("chDaily", {
      type: "line",
      data: { labels: d.daily.labels, datasets: [{ label: "Transactions per day", data: d.daily.values, borderColor: "#10b981", backgroundColor: "#10b98122", fill: true, tension: .35, pointRadius: 2 }] },
      options: { scales: { x: { ticks: { maxTicksLimit: 10 } }, y: gridOpt } }
    });
  } else {
    noData("chDaily", "No date column in this dataset");
  }
  chart("chFreq", {
    type: "doughnut",
    data: { labels: d.frequency.labels, datasets: [{ data: d.frequency.values, backgroundColor: ["#ec4899", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6"], borderWidth: 2 }] },
    options: { cutout: "58%", plugins: { legend: { position: "bottom" } } }
  });
}

function noData(id, msg) {
  if (charts[id]) charts[id].destroy();
  const c = $(id), x = c.getContext("2d");
  x.clearRect(0, 0, c.width, c.height);
  x.fillStyle = "#64748b";
  x.font = "14px sans-serif";
  x.fillText(msg, 20, 40);
}

/* ---------- Dataset Explorer ---------- */
function renderDataset(d, cols) {
  const s = d.stats;
  $("srcName").textContent = d.source;
  $("dsStats").innerHTML = kpi("k-blue", nf(s.rows), "Rows") +
    kpi("k-purple", nf(s.columns), "Columns") +
    kpi("k-green", nf(s.transactions), "Transactions") +
    kpi("k-teal", nf(s.customers), "Customers") +
    kpi("k-orange", nf(s.products), "Products") +
    kpi("k-pink", nf(s.categories), "Categories");

  $("dsInfo").innerHTML = [
    ["Source", d.source],
    ["Columns", d.columns.join(", ")],
    ["Date range", s.date_from ? s.date_from + " → " + s.date_to : "No date column"],
    ["Items sold", nf(s.items_sold)],
    ["Avg basket size (products/transaction)", s.avg_basket],
    ["Total revenue", "$" + nf(s.revenue)]
  ].map(([a, b]) => `<div class="kv"><span>${a}</span><b style="text-align:right;max-width:60%">${esc(b)}</b></div>`).join("");

  const p = d.preprocessing || {};
  const rows = [
    ["Raw rows read", nf(p.raw_rows)],
    ["Rows with missing id/product removed", nf(p.missing_removed)],
    ["Rows with invalid quantity removed", nf(p.invalid_quantity_removed)],
    ["Duplicate rows removed", nf(p.duplicates_removed)],
    ["Final rows used", nf(p.final_rows)]
  ];
  $("prep").innerHTML = rows.map(([a, b]) => `<div class="kv"><span>${a}</span><b>${b}</b></div>`).join("") +
    (p.derived_columns && p.derived_columns.length ? `<div class="kv"><span>Columns derived</span><b>${esc(p.derived_columns.join(", "))}</b></div>` : "") +
    ((p.notes || []).map(n => `<p class="muted">• ${esc(n)}</p>`).join("")) +
    `<p class="muted">Cleaned data is stored in SQLite tables and ready for all Data Mining techniques.</p>`;

  // Schema Table
  if (cols && cols.length) {
    $("dsColsTable").innerHTML = "<tr><th>Column Name</th><th>Type</th><th>Group</th><th>Missing Values</th><th>Unique Values</th><th>Sample Values</th><th>Min / Max / Mean</th></tr>" +
      cols.map(c => `<tr>
        <td><b>${esc(c.name)}</b></td>
        <td><code>${esc(c.dtype)}</code></td>
        <td><span class="tag ${c.group === 'numeric' ? 't-hi' : c.group === 'categorical' ? 't-mid' : 't-lo'}">${c.group}</span></td>
        <td>${c.missing} (${c.missing_pct}%)</td>
        <td>${nf(c.unique)}</td>
        <td>${esc((c.samples || []).join(", "))}</td>
        <td>${c.group === 'numeric' && c.min !== undefined ? `${c.min} / ${c.max} / ${c.mean}` : "–"}</td>
      </tr>`).join("");
  }

  // Preview Table
  $("previewTable").innerHTML = "<tr>" + d.columns.map(c => `<th>${c}</th>`).join("") + "</tr>" +
    d.preview.map(r => "<tr>" + d.columns.map(c => `<td>${esc(r[c])}</td>`).join("") + "</tr>").join("");
}

async function loadDataset() {
  const [d, cols] = await Promise.all([api("/api/dataset"), getColumns()]);
  renderDataset(d, cols);
}

$("btnUpload").onclick = async () => {
  const f = $("fileInput").files[0];
  if (!f) return toast("Choose a CSV, XLSX or XLS file first.", true);
  const fd = new FormData();
  fd.append("file", f);
  $("btnUpload").disabled = true;
  $("btnUpload").textContent = "Processing…";
  try {
    const res = await api("/api/upload", { method: "POST", body: fd });
    invalidateDownstream();
    const cols = await getColumns(true);
    renderDataset(res, cols);
    toast("Dataset loaded and preprocessed: " + f.name);
  } catch (e) {
    toast(e.message, true);
  } finally {
    $("btnUpload").disabled = false;
    $("btnUpload").textContent = "Upload & Analyze";
  }
};

$("btnReset").onclick = async () => {
  try {
    const res = await api("/api/reset", { method: "POST" });
    invalidateDownstream();
    const cols = await getColumns(true);
    renderDataset(res, cols);
    toast("Default supermarket dataset reloaded.");
  } catch (e) {
    toast(e.message, true);
  }
};

/* ---------- Data Preprocessing Workbench ---------- */
function initPrepToolTabs() {
  const tabs = document.querySelectorAll("#prepToolTabs .tool-btn");
  tabs.forEach(btn => {
    btn.onclick = () => {
      tabs.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll("#sec-preprocessing .tool-panel").forEach(p => p.classList.remove("active"));
      const target = $(btn.dataset.tab);
      if (target) target.classList.add("active");
    };
  });

  $("impStrategy").onchange = e => {
    $("impValWrap").style.display = e.target.value === "constant" ? "block" : "none";
  };
  $("fsMethod").onchange = e => {
    const isK = e.target.value === "select_k_best";
    $("fsTargetWrap").style.display = isK ? "block" : "none";
    $("fsKWrap").style.display = isK ? "block" : "none";
    $("fsThreshWrap").style.display = isK ? "none" : "block";
  };
}

function switchPrepPreview(view) {
  $("btnTabAfter").classList.toggle("active", view === "after");
  $("btnTabBefore").classList.toggle("active", view === "before");
  $("wrapPrepAfter").classList.toggle("hidden", view !== "after");
  $("wrapPrepBefore").classList.toggle("hidden", view !== "before");
}

function renderTable(tableEl, records) {
  if (!records || !records.length) {
    tableEl.innerHTML = "<tr><td>No records</td></tr>";
    return;
  }
  const cols = Object.keys(records[0]);
  tableEl.innerHTML = "<tr>" + cols.map(c => `<th>${esc(c)}</th>`).join("") + "</tr>" +
    records.map(r => "<tr>" + cols.map(c => `<td>${esc(r[c])}</td>`).join("") + "</tr>").join("");
}

function renderPrepStatus(status) {
  comparisonState.userRuns.preprocessing = status;
  const b = status.before, a = status.after;
  $("pRowB").textContent = nf(b.rows);
  $("pRowA").textContent = nf(a.rows);
  $("pColB").textContent = nf(b.columns);
  $("pColA").textContent = nf(a.columns);
  $("pMisB").textContent = nf(b.total_missing);
  $("pMisA").textContent = nf(a.total_missing);
  $("pDupB").textContent = nf(b.duplicate_rows);
  $("pDupA").textContent = nf(a.duplicate_rows);

  // History list
  const hist = status.history || [];
  $("prepHistoryEmpty").style.display = hist.length ? "none" : "block";
  $("prepStepsList").innerHTML = hist.map(h => `<li>
    <b>Step ${h.step}:</b> <code>${esc(h.operation)}</code> — ${esc(h.message)} <small class="muted">(${h.time})</small>
  </li>`).join("");

  // Previews
  renderTable($("prepTableAfter"), status.preview_after);
  renderTable($("prepTableBefore"), status.preview_before);

  // Populate form controls from active preprocessed columns
  const cols = a.columns_info || [];
  populatePrepControls(cols);
}

function populatePrepControls(cols) {
  const numCols = cols.filter(c => c.group === "numeric");
  const catCols = cols.filter(c => c.group === "categorical");

  // Impute Column select
  $("impCol").innerHTML = `<option value="all">All Columns</option>` + cols.map(c => `<option value="${esc(c.name)}">${esc(c.name)} (${c.missing} missing)</option>`).join("");
  // Convert Column select
  $("convCol").innerHTML = cols.map(c => `<option value="${esc(c.name)}">${esc(c.name)} [${c.dtype}]</option>`).join("");
  // Feature selection target
  $("fsTarget").innerHTML = cols.map(c => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("");
  // Text cleaning
  $("cleanTextCol").innerHTML = catCols.map(c => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("");
  // Drop columns select
  $("dropCol").innerHTML = cols.map(c => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("");

  // Chips: Categorical Encoding
  $("encChips").innerHTML = catCols.length ? catCols.map(c => `<label class="feature-chip selected">
    <input type="checkbox" name="encCol" value="${esc(c.name)}" checked> ${esc(c.name)}
  </label>`).join("") : "<span class='muted'>No categorical columns found.</span>";

  // Chips: Feature Scaling
  $("scaleChips").innerHTML = numCols.length ? numCols.map(c => `<label class="feature-chip selected">
    <input type="checkbox" name="scaleCol" value="${esc(c.name)}" checked> ${esc(c.name)}
  </label>`).join("") : "<span class='muted'>No numerical columns found.</span>";

  // Chips: Outlier Handling
  $("outlierChips").innerHTML = numCols.length ? numCols.map(c => `<label class="feature-chip selected">
    <input type="checkbox" name="outlierCol" value="${esc(c.name)}" checked> ${esc(c.name)}
  </label>`).join("") : "<span class='muted'>No numerical columns found.</span>";
}

async function loadPreprocessing() {
  initPrepToolTabs();
  const status = await api("/api/preprocessing/status");
  renderPrepStatus(status);
}

async function applyPrep(op, params, btn) {
  btn.disabled = true;
  btn.textContent = "Applying…";
  try {
    const res = await api("/api/preprocessing/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ operation: op, params: params })
    });
    renderPrepStatus(res);
    cachedColumns = [];
    toast("Preprocessing operation applied: " + op);
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = btn.dataset.origText || "Apply";
  }
}

// Preprocessing Action Buttons
$("btnApplyImpute").onclick = function() {
  this.dataset.origText = "Apply Imputation";
  applyPrep("impute_missing", {
    column: $("impCol").value,
    strategy: $("impStrategy").value,
    fill_value: $("impVal").value
  }, this);
};

$("btnApplyDedupe").onclick = function() {
  this.dataset.origText = "Remove Duplicates";
  const sub = $("dupSubset").value === "all" ? null : $("dupSubset").value.split(",");
  applyPrep("remove_duplicates", { subset: sub }, this);
};

$("btnApplyConvert").onclick = function() {
  this.dataset.origText = "Convert Data Type";
  applyPrep("convert_type", {
    column: $("convCol").value,
    target_type: $("convType").value
  }, this);
};

$("btnApplyEncode").onclick = function() {
  this.dataset.origText = "Encode Variables";
  const selected = Array.from(document.querySelectorAll("input[name='encCol']:checked")).map(cb => cb.value);
  if (!selected.length) return toast("Select at least one column to encode.", true);
  applyPrep("encode_categorical", {
    columns: selected,
    method: $("encMethod").value,
    drop_first: $("encDropFirst").checked
  }, this);
};

$("btnApplyScale").onclick = function() {
  this.dataset.origText = "Scale Selected Features";
  const selected = Array.from(document.querySelectorAll("input[name='scaleCol']:checked")).map(cb => cb.value);
  if (!selected.length) return toast("Select at least one numeric feature to scale.", true);
  applyPrep("scale_features", {
    columns: selected,
    method: $("scaleMethod").value
  }, this);
};

$("btnApplyOutliers").onclick = function() {
  this.dataset.origText = "Detect & Handle Outliers";
  const selected = Array.from(document.querySelectorAll("input[name='outlierCol']:checked")).map(cb => cb.value);
  if (!selected.length) return toast("Select at least one numeric column for outliers.", true);
  applyPrep("handle_outliers", {
    columns: selected,
    method: $("outlierMethod").value,
    action: $("outlierAction").value,
    threshold: parseFloat($("outlierThreshold").value)
  }, this);
};

$("btnApplyFs").onclick = function() {
  this.dataset.origText = "Select Features";
  applyPrep("feature_selection", {
    method: $("fsMethod").value,
    target_column: $("fsTarget").value,
    k: parseInt($("fsK").value, 10),
    threshold: parseFloat($("fsThresh").value)
  }, this);
};

$("btnApplyDrop").onclick = function() {
  this.dataset.origText = "Drop Selected Column(s)";
  const selected = Array.from($("dropCol").selectedOptions).map(o => o.value);
  if (!selected.length) return toast("Select at least one column to drop.", true);
  applyPrep("drop_columns", { columns: selected }, this);
};

$("btnApplyCleanText").onclick = function() {
  this.dataset.origText = "Clean Text";
  applyPrep("clean_text", {
    columns: [$("cleanTextCol").value],
    trim: $("cleanTrim").checked,
    lowercase: $("cleanLower").checked
  }, this);
};

$("btnPrepReset").onclick = async () => {
  try {
    const res = await api("/api/preprocessing/reset", { method: "POST" });
    renderPrepStatus(res);
    cachedColumns = [];
    toast("Preprocessing pipeline reset to original dataset.");
  } catch (e) {
    toast(e.message, true);
  }
};

$("btnPrepCommit").onclick = async () => {
  try {
    const res = await api("/api/preprocessing/commit", { method: "POST" });
    invalidateDownstream();
    toast(res.message);
  } catch (e) {
    toast(e.message, true);
  }
};

/* ---------- Classification Workbench ---------- */
function initClassificationControls(cols) {
  // Populate target dropdown: recommend Category or similar categorical/discrete column
  let targetHtml = "";
  cols.forEach(c => {
    const isCat = c.group === "categorical";
    const isRec = c.name === "Category";
    targetHtml += `<option value="${esc(c.name)}" ${isRec ? "selected" : ""}>${esc(c.name)} ${isRec ? "★ (Recommended)" : `[${c.unique} classes]`}</option>`;
  });
  $("clfTarget").innerHTML = targetHtml;

  // Populate feature chips
  renderFeatureChips("clf", cols, ["Quantity", "UnitPrice", "TotalAmount"]);

  $("clfAlgo").onchange = e => {
    const a = e.target.value;
    $("wrapDtDepth").style.display = (a === "decision_tree" || a === "random_forest") ? "block" : "none";
    $("wrapKnnK").style.display = a === "knn" ? "block" : "none";
    $("wrapRfTrees").style.display = a === "random_forest" ? "block" : "none";
    $("wrapC").style.display = (a === "logistic_regression" || a === "svm") ? "block" : "none";
  };
}

function renderFeatureChips(prefix, cols, defaultChecked) {
  const container = $(prefix + "FeatureChips");
  container.innerHTML = cols.map(c => {
    const checked = defaultChecked.includes(c.name) || (c.group === "numeric" && !["TransactionID", "CustomerID"].includes(c.name));
    return `<label class="feature-chip ${checked ? 'selected' : ''}" data-type="${c.group}">
      <input type="checkbox" name="${prefix}Feat" value="${esc(c.name)}" ${checked ? "checked" : ""} onchange="this.parentElement.classList.toggle('selected', this.checked)">
      ${esc(c.name)} <small class="muted">(${c.group})</small>
    </label>`;
  }).join("");
}

function selectAllFeatures(prefix, check) {
  document.querySelectorAll(`input[name='${prefix}Feat']`).forEach(cb => {
    cb.checked = check;
    cb.parentElement.classList.toggle("selected", check);
  });
}

function selectNumericFeatures(prefix) {
  document.querySelectorAll(`input[name='${prefix}Feat']`).forEach(cb => {
    const isNum = cb.parentElement.dataset.type === "numeric";
    cb.checked = isNum;
    cb.parentElement.classList.toggle("selected", isNum);
  });
}

async function loadClassification() {
  const cols = await getColumns();
  initClassificationControls(cols);
}

$("btnRunClf").onclick = async () => {
  const selectedFeats = Array.from(document.querySelectorAll("input[name='clfFeat']:checked")).map(cb => cb.value);
  const target = $("clfTarget").value;
  if (!target) return toast("Select a target column.", true);
  if (!selectedFeats.length) return toast("Select at least one feature column.", true);

  const body = {
    algorithm: $("clfAlgo").value,
    target_column: target,
    feature_columns: selectedFeats,
    params: {
      sample_size: parseInt($("clfSample").value, 10),
      test_size: parseFloat($("clfTestSize").value),
      dt_max_depth: parseInt($("clfDtDepth").value, 10),
      knn_n_neighbors: parseInt($("clfKnnK").value, 10),
      rf_n_estimators: parseInt($("clfRfTrees").value, 10),
      rf_max_depth: parseInt($("clfDtDepth").value, 10),
      lr_c: parseFloat($("clfC").value),
      svm_c: parseFloat($("clfC").value)
    },
    use_preprocessed: true
  };

  const btn = $("btnRunClf");
  btn.disabled = true;
  btn.textContent = "Training Model…";
  $("clfRunStatus").textContent = "Training and evaluating classifier on test set…";

  try {
    const res = await api("/api/classification/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    renderClassificationResults(res);
    toast(`Classification completed! Test Accuracy: ${pc(res.metrics.accuracy)}`);
  } catch (e) {
    $("clfRunStatus").textContent = "Error: " + e.message;
    toast(e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Train & Evaluate Classifier";
  }
};

function renderClassificationResults(res) {
  comparisonState.userRuns.classification[res.algorithm] = res;
  $("clfResultsCard").style.display = "block";
  $("clfRunStatus").textContent = `Finished in ${res.training_time_s}s using ${res.algorithm_name} on ${nf(res.train_samples)} training and ${nf(res.test_samples)} test samples.`;

  const m = res.metrics;
  $("clfKpis").innerHTML = kpi("k-blue", pc(m.accuracy), "Overall Accuracy") +
    kpi("k-purple", pc(m.f1_macro), "Macro F1-Score") +
    kpi("k-green", pc(m.precision_macro), "Macro Precision") +
    kpi("k-teal", pc(m.recall_macro), "Macro Recall") +
    kpi("k-orange", res.classes.length, "Target Classes") +
    kpi("k-pink", res.training_time_s + "s", "Training Time");

  // Summary bullets
  $("clfSummary").innerHTML = res.summary.map(s => `<li>${s}</li>`).join("");

  // Confusion Matrix
  const classes = res.classes;
  const cm = res.confusion_matrix;
  let cmHtml = `<tr><th rowspan="2" style="vertical-align:bottom">Actual Class (Row)</th><th colspan="${classes.length}">Predicted Class (Column)</th></tr><tr>`;
  classes.forEach(c => cmHtml += `<th>${esc(c)}</th>`);
  cmHtml += "</tr>";

  cm.forEach((row, i) => {
    cmHtml += `<tr><td class="actual-label">${esc(classes[i])}</td>`;
    row.forEach((cnt, j) => {
      const isDiag = (i === j);
      const cls = cnt === 0 ? "cm-zero" : isDiag ? "cm-match" : "cm-mismatch";
      cmHtml += `<td class="${cls}">${cnt}</td>`;
    });
    cmHtml += "</tr>";
  });
  $("clfCmTable").innerHTML = cmHtml;

  // Feature Importances Chart
  const fis = res.feature_importances || [];
  if (fis.length) {
    $("clfChartTitle").textContent = "🏆 Feature Importance / Influence";
    chart("chClfImportance", {
      type: "bar",
      data: {
        labels: fis.map(f => f.feature),
        datasets: [{ label: "Relative Importance", data: fis.map(f => f.importance), backgroundColor: COLORS, borderRadius: 6 }]
      },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: gridOpt } }
    });
  } else {
    noData("chClfImportance", "Feature importance not applicable for " + res.algorithm_name);
  }

  // Per-Class Metrics Table
  $("clfClassTable").innerHTML = "<tr><th>Class Name</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support (Test Instances)</th></tr>" +
    res.per_class_metrics.map(r => `<tr>
      <td><b>${esc(r.class_name)}</b></td>
      <td>${pc(r.precision)}</td>
      <td>${pc(r.recall)}</td>
      <td><span class="tag ${r.f1 >= 0.7 ? 't-hi' : r.f1 >= 0.4 ? 't-mid' : 't-lo'}">${pc(r.f1)}</span></td>
      <td>${nf(r.support)}</td>
    </tr>`).join("");

  // Sample Test Predictions Preview
  $("clfPredsTable").innerHTML = "<tr><th>#</th><th>Input Features</th><th>Actual Class</th><th>Predicted Class</th><th>Status</th></tr>" +
    res.sample_predictions.map((p, i) => {
      const featStr = Object.entries(p.features).map(([k, v]) => `${k}: ${v}`).join(", ");
      const tag = p.correct ? `<span class="tag t-hi">✓ Correct</span>` : `<span class="tag t-lo" style="background:#fee2e2;color:#991b1b">✗ Incorrect</span>`;
      return `<tr>
        <td>${i + 1}</td>
        <td><code>${esc(featStr)}</code></td>
        <td><b>${esc(p.actual)}</b></td>
        <td><b>${esc(p.predicted)}</b></td>
        <td>${tag}</td>
      </tr>`;
    }).join("");
}

/* ---------- Regression Workbench ---------- */
function initRegressionControls(cols) {
  const numCols = cols.filter(c => c.group === "numeric");
  // Populate target dropdown
  let targetHtml = "";
  numCols.forEach(c => {
    const isRec = c.name === "TotalAmount";
    targetHtml += `<option value="${esc(c.name)}" ${isRec ? "selected" : ""}>${esc(c.name)} ${isRec ? "★ (Recommended Target)" : ""}</option>`;
  });
  $("regTarget").innerHTML = targetHtml;

  // Populate feature chips
  renderFeatureChips("reg", numCols, ["Quantity", "UnitPrice"]);

  $("regAlgo").onchange = e => {
    const a = e.target.value;
    $("wrapPolyDeg").style.display = a === "polynomial" ? "block" : "none";
    $("wrapRegTreeDepth").style.display = (a === "decision_tree" || a === "random_forest") ? "block" : "none";
    $("wrapRegRfTrees").style.display = a === "random_forest" ? "block" : "none";
  };
}

async function loadRegression() {
  const cols = await getColumns();
  initRegressionControls(cols);
}

$("btnRunReg").onclick = async () => {
  const selectedFeats = Array.from(document.querySelectorAll("input[name='regFeat']:checked")).map(cb => cb.value);
  const target = $("regTarget").value;
  if (!target) return toast("Select a target column.", true);
  if (!selectedFeats.length) return toast("Select at least one feature column.", true);

  const body = {
    algorithm: $("regAlgo").value,
    target_column: target,
    feature_columns: selectedFeats,
    params: {
      sample_size: parseInt($("regSample").value, 10),
      test_size: parseFloat($("regTestSize").value),
      poly_degree: parseInt($("regPolyDeg").value, 10),
      dt_max_depth: parseInt($("regTreeDepth").value, 10),
      rf_n_estimators: parseInt($("regRfTrees").value, 10),
      rf_max_depth: parseInt($("regTreeDepth").value, 10)
    },
    use_preprocessed: true
  };

  const btn = $("btnRunReg");
  btn.disabled = true;
  btn.textContent = "Fitting Regressor…";
  $("regRunStatus").textContent = "Training regression model and generating predictions…";

  try {
    const res = await api("/api/regression/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    renderRegressionResults(res);
    toast(`Regression finished! R² Score: ${res.metrics.r2.toFixed(4)}`);
  } catch (e) {
    $("regRunStatus").textContent = "Error: " + e.message;
    toast(e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Train & Predict Regression";
  }
};

function renderRegressionResults(res) {
  comparisonState.userRuns.regression[res.algorithm] = res;
  $("regResultsCard").style.display = "block";
  $("regRunStatus").textContent = `Fitted ${res.algorithm_name} in ${res.training_time_s}s on ${nf(res.train_samples)} training samples.`;

  const m = res.metrics;
  $("regKpis").innerHTML = kpi("k-blue", m.r2.toFixed(4), "R² Score (Variance Explained)") +
    kpi("k-purple", m.mae.toFixed(2), "Mean Absolute Error (MAE)") +
    kpi("k-green", m.rmse.toFixed(2), "Root Mean Squared Error (RMSE)") +
    kpi("k-teal", m.mse.toFixed(2), "Mean Squared Error (MSE)") +
    kpi("k-orange", res.test_samples, "Test Instances") +
    kpi("k-pink", res.training_time_s + "s", "Training Time");

  $("regSummary").innerHTML = res.summary.map(s => `<li>${s}</li>`).join("");

  // Actual vs Predicted Scatter Plot
  const pts = res.chart_points.map(p => ({ x: p.x, y: p.y }));
  const refMin = res.reference_line.min;
  const refMax = res.reference_line.max;

  chart("chRegScatter", {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Predictions (Actual vs Predicted)",
          data: pts,
          backgroundColor: "#8b5cf6",
          pointRadius: 4
        },
        {
          type: "line",
          label: "Ideal Fit (y = x)",
          data: [{ x: refMin, y: refMin }, { x: refMax, y: refMax }],
          borderColor: "#ef4444",
          borderDash: [6, 6],
          borderWidth: 2,
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      scales: {
        x: { ...gridOpt, title: { display: true, text: "Actual Value" } },
        y: { ...gridOpt, title: { display: true, text: "Predicted Value" } }
      }
    }
  });

  // Feature Importance / Coefficients
  const fis = res.feature_importances || [];
  if (fis.length) {
    $("regChartTitle").textContent = fis[0].is_coef ? "🏆 Regression Coefficients" : "🏆 Feature Importance";
    chart("chRegImportance", {
      type: "bar",
      data: {
        labels: fis.map(f => f.feature),
        datasets: [{ label: fis[0].is_coef ? "Coefficient" : "Importance", data: fis.map(f => f.importance), backgroundColor: COLORS, borderRadius: 6 }]
      },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: gridOpt } }
    });
  } else {
    noData("chRegImportance", "No coefficients available");
  }

  // Sample Predictions Table
  $("regPredsTable").innerHTML = "<tr><th>#</th><th>Input Features</th><th>Actual Value</th><th>Predicted Value</th><th>Absolute Error</th><th>% Error</th></tr>" +
    res.sample_predictions.map((p, i) => {
      const featStr = Object.entries(p.features).map(([k, v]) => `${k}: ${v}`).join(", ");
      const tag = p.pct_error <= 10 ? `<span class="tag t-hi">${p.pct_error}%</span>` : p.pct_error <= 25 ? `<span class="tag t-mid">${p.pct_error}%</span>` : `<span class="tag t-lo">${p.pct_error}%</span>`;
      return `<tr>
        <td>${i + 1}</td>
        <td><code>${esc(featStr)}</code></td>
        <td><b>${p.actual}</b></td>
        <td><b>${p.predicted}</b></td>
        <td>${p.error}</td>
        <td>${tag}</td>
      </tr>`;
    }).join("");
}

/* ---------- Clustering Workbench ---------- */
function setClusteringMode(mode) {
  clusteringMode = mode;
  $("btnModeRows").classList.toggle("active", mode === "rows");
  $("btnModeRfm").classList.toggle("active", mode === "rfm");
  getColumns().then(cols => initClusteringControls(cols));
}

function initClusteringControls(cols) {
  if (clusteringMode === "rfm") {
    // Customer RFM Features
    const rfmFeats = [
      { name: "Frequency", group: "numeric" },
      { name: "Monetary", group: "numeric" },
      { name: "Recency", group: "numeric" }
    ];
    renderFeatureChips("clu", rfmFeats, ["Frequency", "Monetary", "Recency"]);
  } else {
    // Transaction Row Level Features
    const numCols = cols.filter(c => c.group === "numeric");
    renderFeatureChips("clu", numCols, ["Quantity", "UnitPrice", "TotalAmount"]);
  }

  $("cluAlgo").onchange = e => {
    const a = e.target.value;
    $("wrapCluK").style.display = a === "kmeans" || a === "agglomerative" ? "block" : "none";
    $("wrapCluLinkage").style.display = a === "agglomerative" ? "block" : "none";
    $("wrapCluEps").style.display = a === "dbscan" ? "block" : "none";
    $("wrapCluMinSamples").style.display = a === "dbscan" ? "block" : "none";
  };
}

async function loadClustering() {
  const cols = await getColumns();
  initClusteringControls(cols);
}

$("btnRunClu").onclick = async () => {
  const selectedFeats = Array.from(document.querySelectorAll("input[name='cluFeat']:checked")).map(cb => cb.value);
  if (selectedFeats.length < 2) return toast("Select at least 2 features to cluster.", true);

  const body = {
    algorithm: $("cluAlgo").value,
    feature_columns: selectedFeats,
    mode: clusteringMode,
    params: {
      n_clusters: parseInt($("cluK").value, 10),
      linkage: $("cluLinkage").value,
      eps: parseFloat($("cluEps").value),
      min_samples: parseInt($("cluMinSamples").value, 10),
      scale_features: $("cluScale").checked,
      sample_size: parseInt($("cluSample").value, 10)
    },
    use_preprocessed: true
  };

  const btn = $("btnRunClu");
  btn.disabled = true;
  btn.textContent = "Clustering…";
  $("cluRunStatus").textContent = "Executing clustering algorithm and computing silhouette metrics…";

  try {
    const res = await api("/api/clustering/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    renderClusteringResults(res);
    toast(`Clustering finished! ${res.n_clusters} clusters identified.`);
  } catch (e) {
    $("cluRunStatus").textContent = "Error: " + e.message;
    toast(e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Execute Clustering Algorithm";
  }
};

function renderClusteringResults(res) {
  comparisonState.userRuns.clustering[res.algorithm] = res;
  $("cluResultsCard").style.display = "block";
  $("cluRunStatus").textContent = `Completed ${res.algorithm_name} in ${res.training_time_s}s on ${nf(res.total_samples)} instances.`;

  $("cluKpis").innerHTML = kpi("k-blue", res.n_clusters, "Clusters Found") +
    kpi("k-purple", res.silhouette_score !== null ? res.silhouette_score.toFixed(4) : "–", "Silhouette Score") +
    kpi("k-green", res.inertia ? nf(res.inertia) : "–", "Inertia (WCSS)") +
    kpi("k-orange", res.n_noise, "Noise Points (DBSCAN)") +
    kpi("k-teal", nf(res.total_samples), "Clustered Samples") +
    kpi("k-pink", res.training_time_s + "s", "Execution Time");

  $("cluSummary").innerHTML = res.summary.map(s => `<li>${s}</li>`).join("");

  // 2D Projection Scatter Plot
  $("cluScatterTitle").textContent = `📍 2D Projection (${res.scatter_axes.x_label} vs ${res.scatter_axes.y_label})`;
  $("cluScatterSub").textContent = res.pca_explained ? `Projected via PCA (PC1 explains ${(res.pca_explained[0]*100).toFixed(1)}% variance, PC2 explains ${(res.pca_explained[1]*100).toFixed(1)}%).` : "Direct 2D feature projection.";

  const scatterPts = res.scatter_data.map(p => ({
    x: p.x,
    y: p.y,
    cluster: p.cluster,
    cluster_id: p.cluster_id
  }));

  // Group by cluster for legend
  const clusterNames = Object.keys(res.cluster_sizes);
  const datasets = clusterNames.map((cName, idx) => {
    const pts = scatterPts.filter(p => p.cluster === cName);
    const color = cName.includes("Noise") ? "#94a3b8" : COLORS[idx % COLORS.length];
    return {
      label: cName,
      data: pts,
      backgroundColor: color,
      pointRadius: 4
    };
  });

  chart("chCluScatter", {
    type: "scatter",
    data: { datasets: datasets },
    options: {
      scales: {
        x: { ...gridOpt, title: { display: true, text: res.scatter_axes.x_label } },
        y: { ...gridOpt, title: { display: true, text: res.scatter_axes.y_label } }
      }
    }
  });

  // Cluster Size Doughnut Chart
  chart("chCluPie", {
    type: "doughnut",
    data: {
      labels: clusterNames,
      datasets: [{
        data: Object.values(res.cluster_sizes),
        backgroundColor: clusterNames.map((c, i) => c.includes("Noise") ? "#94a3b8" : COLORS[i % COLORS.length]),
        borderWidth: 2
      }]
    },
    options: { cutout: "55%", plugins: { legend: { position: "bottom" } } }
  });

  // Cluster Profiles & Centroids Table
  const feats = res.features;
  $("cluProfileTable").innerHTML = "<tr><th>Cluster</th><th>Instance Count</th><th>Percentage</th>" +
    feats.map(f => `<th>Mean ${esc(f)}</th>`).join("") + "</tr>" +
    res.cluster_profiles.map(cp => `<tr>
      <td><b>${esc(cp.cluster)}</b></td>
      <td>${nf(cp.count)}</td>
      <td>${cp.pct}%</td>
      ${feats.map(f => `<td>${cp[f] !== undefined ? cp[f] : "–"}</td>`).join("")}
    </tr>`).join("");

  // Sample Assignments Table
  const assignRows = res.sample_assignments || [];
  if (assignRows.length) {
    const cols = Object.keys(assignRows[0]);
    $("cluAssignTable").innerHTML = "<tr><th>#</th>" + cols.map(c => `<th>${esc(c)}</th>`).join("") + "</tr>" +
      assignRows.map((r, i) => `<tr>
        <td>${i + 1}</td>
        ${cols.map(c => c === "AssignedCluster" ? `<td><span class="tag t-hi">${esc(r[c])}</span></td>` : `<td>${esc(r[c])}</td>`).join("")}
      </tr>`).join("");
  }
}

/* ---------- Algorithms Conceptual Comparison (Preserved) ---------- */
async function loadAlgorithms() {
  const [a, cfg] = await Promise.all([api("/api/algorithms"), api("/api/config")]);
  $("algoDisclaimer").textContent = a.disclaimer + " Engine in use: " + cfg.engine + ".";
  $("pSup").value = cfg.defaults.min_support;
  $("pConf").value = cfg.defaults.min_confidence;
  $("pLift").value = cfg.defaults.min_lift;
  $("pLevel").value = cfg.defaults.level;
  $("pLen").value = cfg.defaults.max_len;

  const cl = ["impl", "c2", "c3"];
  $("algoCards").innerHTML = a.algorithms.map((x, i) => `<div class="card algo ${cl[i]}">
    <h4>${x.name} <span class="badge ${x.status === "IMPLEMENTED" ? "green" : "orange"}">${x.status}</span></h4>
    <div class="full">${esc(x.full)}</div>
    <dl>
      <dt>Basic concept</dt><dd>${esc(x.concept)}</dd>
      <dt>Candidate generation</dt><dd>${esc(x.candidate_generation)}</dd>
      <dt>Pruning approach</dt><dd>${esc(x.pruning)}</dd>
      <dt>Purpose</dt><dd>${esc(x.purpose)}</dd>
      <dt>Strengths</dt><dd>${esc(x.strengths)}</dd>
      <dt>Limitations</dt><dd>${esc(x.limitations)}</dd>
    </dl>
  </div>`).join("");

  $("algoTable").innerHTML = "<tr><th>Aspect</th>" + a.algorithms.map(x => `<th>${x.name}</th>`).join("") + "</tr>" +
    a.comparison.map(r => `<tr><td><b>${r.aspect}</b></td>${r.values.map(v => `<td class="wrap">${esc(v)}</td>`).join("")}</tr>`).join("");
}

/* ---------- Association Rules (Preserved) ---------- */
const liftTag = l => `<span class="tag ${l >= 3 ? "t-hi" : l >= 1.5 ? "t-mid" : "t-lo"}">${l.toFixed(2)}</span>`;

function rulesTable(el, rules) {
  el.innerHTML = "<tr><th>#</th><th>Antecedent (if bought)</th><th>Consequent (then bought)</th><th>Support</th><th>Confidence</th><th>Lift</th></tr>" +
    (rules.length ? rules.map((r, i) => `<tr>
      <td>${i + 1}</td>
      <td>${esc(r.antecedents.join(" + "))}</td>
      <td>${esc(r.consequents.join(" + "))}</td>
      <td>${pc(r.support, 2)}</td>
      <td>${pc(r.confidence)}</td>
      <td>${liftTag(r.lift)}</td>
    </tr>`).join("") : `<tr><td colspan="6">No rules for these thresholds.</td></tr>`);
}

$("btnRun").onclick = async () => {
  const body = {
    min_support: +$("pSup").value,
    min_confidence: +$("pConf").value,
    min_lift: +$("pLift").value,
    level: $("pLevel").value,
    max_len: +$("pLen").value
  };
  const b = $("btnRun");
  b.disabled = true;
  b.textContent = "Running Apriori…";
  $("runInfo").textContent = "Preprocessing transactions and mining frequent itemsets…";

  try {
    const r = await api("/api/apriori", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    $("runInfo").textContent = `Done in ${(r.timing.preprocess_s + r.timing.apriori_s).toFixed(1)} s using ${r.engine}.`;
    $("runResults").style.display = "grid";
    $("runRulesCard").style.display = "block";
    $("runSteps").innerHTML = r.steps.map(s => `<li>${esc(s)}</li>`).join("");
    $("runMeta").innerHTML = [
      `${nf(r.n_transactions)} transactions`,
      `${nf(r.n_items)} items`,
      `${nf(r.n_itemsets)} frequent itemsets`,
      `${nf(r.n_rules)} rules`,
      ...Object.entries(r.itemsets_by_size).map(([k, v]) => `size ${k}: ${v}`)
    ].map(c => `<span class="chip">${c}</span>`).join("");

    $("isCount").textContent = `(top ${r.itemsets.length} of ${nf(r.n_itemsets)} by support)`;
    $("itemsetTable").innerHTML = "<tr><th>#</th><th>Itemset</th><th>Size</th><th>Support</th></tr>" +
      r.itemsets.map((x, i) => `<tr><td>${i + 1}</td><td>${esc(x.items.join(" + "))}</td><td>${x.size}</td><td>${pc(x.support, 2)}</td></tr>`).join("");
    $("ruCount").textContent = `(${nf(r.n_rules)} found, showing top ${r.rules.length} by lift)`;
    rulesTable($("algoRulesTable"), r.rules);
    comparisonState.userRuns.rules = r;
    loaded.rules = false;
    loaded.insights = false;
    toast(`Apriori finished: ${nf(r.n_itemsets)} itemsets, ${nf(r.n_rules)} rules.`);
  } catch (e) {
    $("runInfo").textContent = "";
    toast(e.message, true);
  } finally {
    b.disabled = false;
    b.textContent = "Preprocess & Run Apriori";
  }
};

async function loadRules() {
  const d = await api("/api/rules"), r = d.result;
  $("interp").innerHTML = d.interpretation.map(t => `<li>${esc(t)}</li>`).join("");
  $("rulesParams").textContent = `Parameters: support ≥ ${r.params.min_support}, confidence ≥ ${r.params.min_confidence}, lift ≥ ${r.params.min_lift}, ${r.level.toLowerCase()} level. Change them above.`;
  $("ruleCards").innerHTML = r.rules.slice(0, 8).map((x, i) => `<div class="rcard">
    <div class="rn">RULE #${i + 1}</div>
    <div class="rt">${esc(x.antecedents.join(" + "))} → ${esc(x.consequents.join(" + "))}</div>
    <div class="m"><span class="tag t-lo">Support ${pc(x.support, 2)}</span><span class="tag t-mid">Conf ${pc(x.confidence, 0)}</span>${liftTag(x.lift)}</div>
  </div>`).join("") || `<div class="card">No rules for the current thresholds.</div>`;

  rulesTable($("rulesTable"), r.rules);
  const pts = r.rules.map(x => ({
    x: x.support * 100,
    y: x.confidence * 100,
    lift: x.lift,
    label: x.antecedents.join("+") + " → " + x.consequents.join("+")
  }));

  chart("chScatter", {
    type: "scatter",
    data: {
      datasets: [{
        label: "Rules",
        data: pts,
        backgroundColor: pts.map(p => p.lift >= 3 ? "#10b981" : p.lift >= 1.5 ? "#f59e0b" : "#8b5cf6"),
        pointRadius: pts.map(p => Math.min(4 + p.lift, 14))
      }]
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => `${c.raw.label} | supp ${c.raw.x.toFixed(2)}%, conf ${c.raw.y.toFixed(0)}%, lift ${c.raw.lift.toFixed(2)}` } }
      },
      scales: {
        x: { ...gridOpt, title: { display: true, text: "Support (%)" } },
        y: { ...gridOpt, title: { display: true, text: "Confidence (%)" } }
      }
    }
  });
}

/* ---------- Business Insights (Preserved) ---------- */
async function loadInsights() {
  const d = await api("/api/insights");
  const card = c => `<div class="icard i-${c.color}">
    <div class="ty">${esc(c.type)}</div>
    <h4>${esc(c.title)}</h4>
    <p>${esc(c.text)}</p>
    ${c.metrics ? `<div class="im"><span>Support ${pc(c.metrics.support, 2)}</span><span>Confidence ${pc(c.metrics.confidence, 0)}</span><span>Lift ${c.metrics.lift.toFixed(2)}</span></div>` : ""}
  </div>`;
  $("insightCards").innerHTML = d.cards.map(card).join("") + d.extra.map(card).join("") || "<div class='card'>No rules available – lower the thresholds in Algorithms.</div>";
}

/* ---------- Yearly Sales (Preserved) ---------- */
async function loadYearly() {
  const d = await api("/api/yearly");
  $("yearlyMsg").classList.toggle("hidden", d.available && !d.note);
  $("yearlyBody").classList.toggle("hidden", !d.available);
  if (!d.available) { $("yearlyMsg").textContent = d.message; return; }
  $("yearlyMsg").textContent = d.note || "";

  const Y = d.years, cls = ["k-blue", "k-green", "k-purple", "k-orange"];
  let h = Y.map((y, i) => kpi(cls[i % 4], nf(d.items_by_year[y]), `Items Sold in ${y} <br>${nf(d.transactions_by_year[y])} transactions`)).join("");
  h += d.yoy.map(g => kpi(g.items >= 0 ? "k-teal" : "k-pink", (g.items >= 0 ? "+" : "") + g.items + "%", `Items YoY ${g.from} → ${g.to}<br>Revenue ${g.revenue >= 0 ? "+" : ""}${g.revenue}%`)).join("");
  $("yKpis").innerHTML = h;

  const col = (i) => YEAR_COLORS[i % YEAR_COLORS.length];
  chart("chYItems", {
    type: "bar",
    data: { labels: Y.map(String), datasets: [{ label: "Items sold", data: Y.map(y => d.items_by_year[y]), backgroundColor: Y.map((_, i) => col(i)), borderRadius: 8 }] },
    options: { plugins: { legend: { display: false } }, scales: { y: gridOpt } }
  });

  if (d.yoy.length) {
    chart("chYoY", {
      type: "bar",
      data: {
        labels: d.yoy.map(g => `${g.from} → ${g.to}`),
        datasets: [
          { label: "Items %", data: d.yoy.map(g => g.items), backgroundColor: "#3b82f6", borderRadius: 6 },
          { label: "Transactions %", data: d.yoy.map(g => g.transactions), backgroundColor: "#8b5cf6", borderRadius: 6 },
          { label: "Revenue %", data: d.yoy.map(g => g.revenue), backgroundColor: "#10b981", borderRadius: 6 }
        ]
      },
      options: { scales: { y: { ...gridOpt, title: { display: true, text: "% growth" } } } }
    });
  } else {
    noData("chYoY", "Need at least two years for growth");
  }

  const M = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  chart("chMonthly", {
    type: "line",
    data: { labels: M, datasets: Y.map((y, i) => ({ label: String(y), data: d.monthly[y], borderColor: col(i), backgroundColor: col(i), tension: .35, pointRadius: 3 })) },
    options: { scales: { y: gridOpt } }
  });
  chart("chTopYear", {
    type: "bar",
    data: { labels: d.top_items.labels, datasets: Y.map((y, i) => ({ label: String(y), data: d.top_items.series[y], backgroundColor: col(i), borderRadius: 4 })) },
    options: { scales: { x: { ticks: { maxRotation: 55, minRotation: 40 } }, y: gridOpt } }
  });
  chart("chQuarter", {
    type: "bar",
    data: { labels: ["Q1", "Q2", "Q3", "Q4"], datasets: Y.map((y, i) => ({ label: String(y), data: d.quarterly[y], backgroundColor: col(i), borderRadius: 6 })) },
    options: { scales: { y: gridOpt } }
  });
}

/* ---------- Data Mining Comparison & Analysis Module ---------- */
function initComparisonControls(cols) {
  // Target dropdown for Classification
  const clfTarget = $("cmpClfTarget");
  if (clfTarget) {
    let html = "";
    cols.forEach(c => {
      const isRec = c.name === "Category";
      html += `<option value="${esc(c.name)}" ${isRec ? "selected" : ""}>${esc(c.name)} ${isRec ? "★ (Recommended)" : ""}</option>`;
    });
    clfTarget.innerHTML = html;
  }

  // Target dropdown for Regression
  const regTarget = $("cmpRegTarget");
  if (regTarget) {
    const numCols = cols.filter(c => c.group === "numeric");
    let html = "";
    numCols.forEach(c => {
      const isRec = c.name === "TotalAmount";
      html += `<option value="${esc(c.name)}" ${isRec ? "selected" : ""}>${esc(c.name)} ${isRec ? "★ (Recommended)" : ""}</option>`;
    });
    regTarget.innerHTML = html;
  }

  // Preset buttons
  $("btnCmpPresetAll").onclick = () => {
    $("cmpTechPrep").checked = true;
    $("cmpTechClf").checked = true;
    $("cmpTechReg").checked = true;
    $("cmpTechClu").checked = true;
    $("cmpTechRules").checked = true;
    ["cmpClfDt", "cmpClfRf", "cmpClfKnn", "cmpClfLr", "cmpClfNb"].forEach(id => { if ($(id)) $(id).checked = true; });
    ["cmpRegLinear", "cmpRegPoly", "cmpRegDt", "cmpRegRf"].forEach(id => { if ($(id)) $(id).checked = true; });
    ["cmpCluKmeans", "cmpCluAgg", "cmpCluDbscan"].forEach(id => { if ($(id)) $(id).checked = true; });
    if ($("cmpRulesApriori")) $("cmpRulesApriori").checked = true;
    updatePresetActiveState("btnCmpPresetAll");
  };

  $("btnCmpPresetSupervised").onclick = () => {
    $("cmpTechPrep").checked = false;
    $("cmpTechClf").checked = true;
    $("cmpTechReg").checked = true;
    $("cmpTechClu").checked = false;
    $("cmpTechRules").checked = false;
    ["cmpClfDt", "cmpClfRf", "cmpClfKnn", "cmpClfLr", "cmpClfNb"].forEach(id => { if ($(id)) $(id).checked = true; });
    ["cmpRegLinear", "cmpRegPoly", "cmpRegDt", "cmpRegRf"].forEach(id => { if ($(id)) $(id).checked = true; });
    updatePresetActiveState("btnCmpPresetSupervised");
  };

  $("btnCmpPresetUnsupervised").onclick = () => {
    $("cmpTechPrep").checked = false;
    $("cmpTechClf").checked = false;
    $("cmpTechReg").checked = false;
    $("cmpTechClu").checked = true;
    $("cmpTechRules").checked = true;
    ["cmpCluKmeans", "cmpCluAgg", "cmpCluDbscan"].forEach(id => { if ($(id)) $(id).checked = true; });
    if ($("cmpRulesApriori")) $("cmpRulesApriori").checked = true;
    updatePresetActiveState("btnCmpPresetUnsupervised");
  };

  $("btnCmpPresetClear").onclick = () => {
    $("cmpTechPrep").checked = false;
    $("cmpTechClf").checked = false;
    $("cmpTechReg").checked = false;
    $("cmpTechClu").checked = false;
    $("cmpTechRules").checked = false;
    updatePresetActiveState("btnCmpPresetClear");
  };

  function updatePresetActiveState(activeId) {
    ["btnCmpPresetAll", "btnCmpPresetSupervised", "btnCmpPresetUnsupervised", "btnCmpPresetClear"].forEach(id => {
      const b = $(id);
      if (b) b.classList.toggle("active", id === activeId);
    });
  }

  // View navigation tabs
  const viewTabs = document.querySelectorAll("#cmpViewTabs .cmp-tab");
  viewTabs.forEach(btn => {
    btn.onclick = () => {
      viewTabs.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.dataset.tab;
      document.querySelectorAll("#sec-comparison .cmp-section-block").forEach(b => {
        b.classList.toggle("hidden", b.id !== targetId);
      });
      setTimeout(() => {
        ["chCmpClf", "chCmpReg", "chCmpClu", "chCmpSpeed", "chCmpRules", "chCmpPrep"].forEach(id => {
          if (charts[id]) {
            try { charts[id].resize(); } catch (e) { }
          }
        });
      }, 50);
    };
  });
}

async function loadComparison() {
  const cols = await getColumns();
  initComparisonControls(cols);
  if (!comparisonState.activeBenchmark) {
    runComparisonBenchmark();
  } else {
    renderComparisonDashboard(comparisonState.activeBenchmark);
  }
}

$("btnRunComparison").onclick = () => runComparisonBenchmark();

$("btnSyncComparison").onclick = () => {
  const hasUserClf = Object.keys(comparisonState.userRuns.classification).length > 0;
  const hasUserReg = Object.keys(comparisonState.userRuns.regression).length > 0;
  const hasUserClu = Object.keys(comparisonState.userRuns.clustering).length > 0;
  const hasUserRules = comparisonState.userRuns.rules !== null;

  if (!hasUserClf && !hasUserReg && !hasUserClu && !hasUserRules) {
    toast("No previous workbench results found yet. Running full benchmark now…");
    runComparisonBenchmark();
  } else {
    toast("Syncing results from active workbenches and benchmarking any remaining techniques…");
    runComparisonBenchmark();
  }
};

async function runComparisonBenchmark() {
  const prepOn = $("cmpTechPrep") ? $("cmpTechPrep").checked : true;
  const clfOn = $("cmpTechClf") ? $("cmpTechClf").checked : true;
  const regOn = $("cmpTechReg") ? $("cmpTechReg").checked : true;
  const cluOn = $("cmpTechClu") ? $("cmpTechClu").checked : true;
  const rulesOn = $("cmpTechRules") ? $("cmpTechRules").checked : true;

  if (!prepOn && !clfOn && !regOn && !cluOn && !rulesOn) {
    return toast("Please select at least one technique to compare.", true);
  }

  // Gather classification algorithms
  const clfAlgos = [];
  if ($("cmpClfDt") && $("cmpClfDt").checked) clfAlgos.push("decision_tree");
  if ($("cmpClfRf") && $("cmpClfRf").checked) clfAlgos.push("random_forest");
  if ($("cmpClfKnn") && $("cmpClfKnn").checked) clfAlgos.push("knn");
  if ($("cmpClfLr") && $("cmpClfLr").checked) clfAlgos.push("logistic_regression");
  if ($("cmpClfNb") && $("cmpClfNb").checked) clfAlgos.push("naive_bayes");
  if ($("cmpClfSvm") && $("cmpClfSvm").checked) clfAlgos.push("svm");

  // Gather regression algorithms
  const regAlgos = [];
  if ($("cmpRegLinear") && $("cmpRegLinear").checked) regAlgos.push("linear");
  if ($("cmpRegPoly") && $("cmpRegPoly").checked) regAlgos.push("polynomial");
  if ($("cmpRegDt") && $("cmpRegDt").checked) regAlgos.push("decision_tree");
  if ($("cmpRegRf") && $("cmpRegRf").checked) regAlgos.push("random_forest");

  // Gather clustering algorithms
  const cluAlgos = [];
  if ($("cmpCluKmeans") && $("cmpCluKmeans").checked) cluAlgos.push("kmeans");
  if ($("cmpCluAgg") && $("cmpCluAgg").checked) cluAlgos.push("agglomerative");
  if ($("cmpCluDbscan") && $("cmpCluDbscan").checked) cluAlgos.push("dbscan");

  const payload = {
    use_preprocessed: true,
    techniques: {
      preprocessing: { enabled: prepOn },
      classification: {
        enabled: clfOn,
        algorithms: clfAlgos.length ? clfAlgos : ["decision_tree", "random_forest"],
        target_column: $("cmpClfTarget") ? $("cmpClfTarget").value : "Category",
        feature_columns: ["Quantity", "UnitPrice", "TotalAmount"],
        params: { sample_size: 2500, test_size: 0.2 }
      },
      regression: {
        enabled: regOn,
        algorithms: regAlgos.length ? regAlgos : ["linear", "polynomial"],
        target_column: $("cmpRegTarget") ? $("cmpRegTarget").value : "TotalAmount",
        feature_columns: ["Quantity", "UnitPrice"],
        params: { sample_size: 2500, test_size: 0.2, poly_degree: 2, dt_max_depth: 6, rf_n_estimators: 40 }
      },
      clustering: {
        enabled: cluOn,
        algorithms: cluAlgos.length ? cluAlgos : ["kmeans", "agglomerative"],
        mode: $("cmpCluMode") ? $("cmpCluMode").value : "rows",
        params: { sample_size: 2000, n_clusters: 3, scale_features: true }
      },
      rules: {
        enabled: rulesOn,
        params: { min_support: 0.005, min_confidence: 0.25, min_lift: 1.2, level: "Product", max_len: 3 }
      }
    }
  };

  const btn = $("btnRunComparison");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Benchmarking Selected Models…";
  }
  if ($("cmpRunStatus")) {
    $("cmpRunStatus").textContent = "Executing standardized comparative benchmark across selected algorithms…";
  }

  try {
    const data = await api("/api/comparison/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    comparisonState.activeBenchmark = data;
    renderComparisonDashboard(data);
    toast("Comparative analysis generated successfully!");
    if ($("cmpRunStatus")) {
      $("cmpRunStatus").textContent = `Benchmark completed at ${data.timestamp}. Analysis dashboard updated below.`;
    }
  } catch (e) {
    if ($("cmpRunStatus")) $("cmpRunStatus").textContent = "Error: " + e.message;
    toast(e.message, true);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "⚡ Generate Comparative Analysis";
    }
  }
}

function renderComparisonDashboard(data) {
  $("cmpResultsCard").style.display = "block";

  const clf = data.classification;
  const reg = data.regression;
  const clu = data.clustering;
  const rules = data.rules;
  const prep = data.preprocessing;
  const holistic = data.holistic;

  // 1. Best-Performing Algorithm Highlight Cards (Winners Grid)
  let winnerHtml = "";

  // Classification Winner
  if (clf && clf.best_algorithm) {
    const b = clf.best_algorithm;
    winnerHtml += `<div class="winner-card w-clf">
      <div class="winner-badge">🏆 Classification Leader</div>
      <div class="winner-name">${esc(b.name)}</div>
      <div class="muted" style="font-size:12px">Highest Macro F1-Score &amp; Accuracy</div>
      <div class="winner-metrics">
        <span class="winner-stat">Accuracy: <b>${pc(b.accuracy)}</b></span>
        <span class="winner-stat">Macro F1: <b>${b.f1_macro.toFixed(4)}</b></span>
        <span class="winner-stat">Time: <b>${b.training_time_s}s</b></span>
      </div>
    </div>`;
  }

  // Regression Winner
  if (reg && reg.best_algorithm) {
    const b = reg.best_algorithm;
    winnerHtml += `<div class="winner-card w-reg">
      <div class="winner-badge" style="background:#f3e8ff;color:#7e22ce">🏆 Regression Leader</div>
      <div class="winner-name">${esc(b.name)}</div>
      <div class="muted" style="font-size:12px">Maximum Variance Explained (R²)</div>
      <div class="winner-metrics">
        <span class="winner-stat">R² Score: <b>${b.r2.toFixed(4)}</b></span>
        <span class="winner-stat">RMSE: <b>$${b.rmse.toFixed(2)}</b></span>
        <span class="winner-stat">MAE: <b>$${b.mae.toFixed(2)}</b></span>
      </div>
    </div>`;
  }

  // Clustering Winner
  if (clu && clu.best_algorithm) {
    const b = clu.best_algorithm;
    winnerHtml += `<div class="winner-card w-clu">
      <div class="winner-badge" style="background:#ccfbf1;color:#0f766e">🏆 Clustering Leader</div>
      <div class="winner-name">${esc(b.name)}</div>
      <div class="muted" style="font-size:12px">Highest Silhouette Cluster Separation</div>
      <div class="winner-metrics">
        <span class="winner-stat">Silhouette: <b>${b.silhouette_score !== null ? b.silhouette_score.toFixed(4) : "–"}</b></span>
        <span class="winner-stat">Clusters: <b>${b.n_clusters}</b></span>
        <span class="winner-stat">Time: <b>${b.training_time_s}s</b></span>
      </div>
    </div>`;
  }

  // Association Rules Winner
  if (rules && rules.available && rules.top_rules && rules.top_rules.highest_lift) {
    const r = rules.top_rules.highest_lift;
    const a = r.antecedents.join(" + ");
    const c = r.consequents.join(" + ");
    winnerHtml += `<div class="winner-card w-rules">
      <div class="winner-badge" style="background:#ffedd5;color:#c2410c">🏆 Top Association Rule</div>
      <div class="winner-name" style="font-size:14px;word-break:break-word">${esc(a)} → ${esc(c)}</div>
      <div class="muted" style="font-size:12px">Maximum Basket Affinity (Lift)</div>
      <div class="winner-metrics">
        <span class="winner-stat">Lift: <b>${r.lift.toFixed(2)}x</b></span>
        <span class="winner-stat">Confidence: <b>${pc(r.confidence)}</b></span>
        <span class="winner-stat">Support: <b>${pc(r.support, 2)}</b></span>
      </div>
    </div>`;
  }

  // Preprocessing Winner
  if (prep && prep.available) {
    const d = prep.deltas;
    const a = prep.stats_after;
    winnerHtml += `<div class="winner-card w-prep">
      <div class="winner-badge" style="background:#dcfce7;color:#15803d">🏆 Data Quality Baseline</div>
      <div class="winner-name">${d.missing_reduction_pct}% Nulls Cleaned</div>
      <div class="muted" style="font-size:12px">Data Cleanliness &amp; Matrix Integrity</div>
      <div class="winner-metrics">
        <span class="winner-stat">Hygiene: <b>${a.cleanliness_score}%</b></span>
        <span class="winner-stat">Kept: <b>${d.rows_retained_pct}%</b></span>
        <span class="winner-stat">Steps: <b>${d.pipeline_steps}</b></span>
      </div>
    </div>`;
  }

  $("cmpWinnerGrid").innerHTML = winnerHtml || `<div class="card">No models evaluated. Select algorithms above.</div>`;

  // 2. Automated Interpretation & Performance Rationale List
  const explanations = [];
  if (clf && clf.explanation) explanations.push(`<b>Classification:</b> ${clf.explanation}`);
  if (reg && reg.explanation) explanations.push(`<b>Regression:</b> ${reg.explanation}`);
  if (clu && clu.explanation) explanations.push(`<b>Clustering:</b> ${clu.explanation}`);
  if (rules && rules.explanation) explanations.push(`<b>Association Rules:</b> ${rules.explanation}`);
  if (prep && prep.explanation) explanations.push(`<b>Data Preprocessing:</b> ${prep.explanation}`);
  if (holistic && holistic.summary) explanations.push(`<b>Pipeline Lifecycle:</b> ${holistic.summary}`);

  $("cmpExplanationList").innerHTML = explanations.map(e => `<li>${e}</li>`).join("");

  // 3. Render Visualizations Grid
  const clfList = (clf && clf.results) ? clf.results : [];
  const regList = (reg && reg.results) ? reg.results : [];
  const cluList = (clu && clu.results) ? clu.results : [];

  // Chart 1: Classification Comparison
  if (clfList.length) {
    chart("chCmpClf", {
      type: "bar",
      data: {
        labels: clfList.map(r => r.name),
        datasets: [
          { label: "Accuracy (%)", data: clfList.map(r => r.accuracy * 100), backgroundColor: "#3b82f6", borderRadius: 4 },
          { label: "Macro F1 (%)", data: clfList.map(r => r.f1_macro * 100), backgroundColor: "#8b5cf6", borderRadius: 4 },
          { label: "Macro Precision (%)", data: clfList.map(r => r.precision_macro * 100), backgroundColor: "#10b981", borderRadius: 4 },
          { label: "Macro Recall (%)", data: clfList.map(r => r.recall_macro * 100), backgroundColor: "#f59e0b", borderRadius: 4 }
        ]
      },
      options: {
        scales: {
          x: { ticks: { maxRotation: 35, minRotation: 20 } },
          y: { ...gridOpt, min: 0, max: 100, title: { display: true, text: "Score (%)" } }
        }
      }
    });
  } else {
    noData("chCmpClf", "No classification algorithms selected");
  }

  // Chart 2: Regression Comparison (Dual Axis)
  if (regList.length) {
    chart("chCmpReg", {
      type: "bar",
      data: {
        labels: regList.map(r => r.name),
        datasets: [
          { label: "R² Score (Variance Explained)", data: regList.map(r => r.r2), backgroundColor: "#8b5cf6", borderRadius: 5, yAxisID: "y" },
          { label: "RMSE Error ($)", data: regList.map(r => r.rmse), backgroundColor: "#ec4899", borderRadius: 5, yAxisID: "y1" }
        ]
      },
      options: {
        scales: {
          x: { ticks: { maxRotation: 30, minRotation: 15 } },
          y: { ...gridOpt, position: "left", title: { display: true, text: "R² Score (Higher is Better)" } },
          y1: { ...gridOpt, position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "RMSE ($ Lower is Better)" } }
        }
      }
    });
  } else {
    noData("chCmpReg", "No regression algorithms selected");
  }

  // Chart 3: Clustering Silhouette Comparison
  if (cluList.length) {
    chart("chCmpClu", {
      type: "bar",
      data: {
        labels: cluList.map(r => r.name),
        datasets: [{
          label: "Silhouette Score",
          data: cluList.map(r => r.silhouette_score !== null ? r.silhouette_score : 0),
          backgroundColor: ["#14b8a6", "#3b82f6", "#f59e0b", "#ec4899"],
          borderRadius: 6
        }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { maxRotation: 25, minRotation: 10 } },
          y: { ...gridOpt, min: 0, max: 1, title: { display: true, text: "Silhouette Score (Separation & Cohesion)" } }
        }
      }
    });
  } else {
    noData("chCmpClu", "No clustering algorithms selected");
  }

  // Chart 4: Algorithmic Speed Comparison across all techniques
  const speedItems = [
    ...clfList.map(r => ({ name: r.name + " [Clf]", time: r.training_time_s, color: "#3b82f6" })),
    ...regList.map(r => ({ name: r.name + " [Reg]", time: r.training_time_s, color: "#8b5cf6" })),
    ...cluList.map(r => ({ name: r.name + " [Clu]", time: r.training_time_s, color: "#14b8a6" }))
  ];
  if (speedItems.length) {
    chart("chCmpSpeed", {
      type: "bar",
      data: {
        labels: speedItems.map(s => s.name),
        datasets: [{ label: "Execution Time (Seconds)", data: speedItems.map(s => s.time), backgroundColor: speedItems.map(s => s.color), borderRadius: 5 }]
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: { x: { ...gridOpt, title: { display: true, text: "Seconds" } } }
      }
    });
  } else {
    noData("chCmpSpeed", "No speed metrics available");
  }

  // Chart 5: Association Rules Scatter (Lift vs Confidence vs Support)
  if (rules && rules.available && rules.sample_rules && rules.sample_rules.length) {
    const pts = rules.sample_rules.map(r => ({
      x: r.support * 100,
      y: r.confidence * 100,
      lift: r.lift,
      label: `${r.antecedents.join("+")} → ${r.consequents.join("+")}`
    }));
    chart("chCmpRules", {
      type: "scatter",
      data: {
        datasets: [{
          label: "Association Rules",
          data: pts,
          backgroundColor: pts.map(p => p.lift >= 3 ? "#10b981" : p.lift >= 1.8 ? "#f59e0b" : "#8b5cf6"),
          pointRadius: pts.map(p => Math.min(14, 5 + p.lift * 1.5))
        }]
      },
      options: {
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: c => `${c.raw.label} | Supp: ${c.raw.x.toFixed(2)}%, Conf: ${c.raw.y.toFixed(1)}%, Lift: ${c.raw.lift.toFixed(2)}` } }
        },
        scales: {
          x: { ...gridOpt, title: { display: true, text: "Support (%)" } },
          y: { ...gridOpt, title: { display: true, text: "Confidence (%)" } }
        }
      }
    });

    // Association Rules KPIs
    const av = rules.averages || {};
    $("cmpRulesMetricsSummary").innerHTML =
      kpi("k-blue", nf(rules.n_rules), "Total Rules Mined") +
      kpi("k-purple", nf(rules.n_itemsets), "Frequent Itemsets") +
      kpi("k-green", (rules.top_rules.highest_lift ? rules.top_rules.highest_lift.lift.toFixed(2) + "x" : "–"), "Maximum Lift") +
      kpi("k-orange", pc(av.avg_confidence), "Average Confidence");
  } else {
    noData("chCmpRules", "No association rules generated");
  }

  // Chart 6: Preprocessing Transformation Bar
  if (prep && prep.available) {
    const b = prep.stats_before;
    const a = prep.stats_after;
    chart("chCmpPrep", {
      type: "bar",
      data: {
        labels: ["Missing Values", "Duplicate Rows", "Cleanliness Score (%)"],
        datasets: [
          { label: "Baseline (Before)", data: [b.missing_values, b.duplicate_rows, b.cleanliness_score], backgroundColor: "#f87171", borderRadius: 6 },
          { label: "Cleaned (After)", data: [a.missing_values, a.duplicate_rows, a.cleanliness_score], backgroundColor: "#10b981", borderRadius: 6 }
        ]
      },
      options: { scales: { y: gridOpt } }
    });

    const d = prep.deltas;
    $("cmpPrepKpis").innerHTML =
      kpi("k-blue", nf(a.rows), "Cleaned Instances") +
      kpi("k-green", `${d.missing_reduction_pct}%`, "Missing Reduced") +
      kpi("k-teal", nf(d.duplicates_removed), "Duplicates Cleaned") +
      kpi("k-purple", `${a.cleanliness_score}%`, "Data Hygiene Score");
  } else {
    noData("chCmpPrep", "Preprocessing stats not initialized");
  }

  // 4. Comparison Tables

  // Classification Table
  if (clfList.length) {
    $("cmpClfTable").innerHTML = "<tr><th>Rank</th><th>Algorithm</th><th>Accuracy</th><th>Macro Precision</th><th>Macro Recall</th><th>Macro F1-Score</th><th>Weighted F1</th><th>Training Time</th><th>Strengths &amp; Trade-offs</th></tr>" +
      clfList.map((r, i) => {
        const isWinner = (i === 0);
        const tier = isWinner ? '<span class="tag t-hi">🏆 Leader</span>' : r.accuracy >= 0.7 ? '<span class="tag t-mid">Solid</span>' : '<span class="tag t-lo">Baseline</span>';
        const tradeOff = r.name.includes("Forest") ? "Ensemble variance reduction; higher computational cost" :
          r.name.includes("Tree") ? "Fast non-linear splits; zero feature scaling required" :
          r.name.includes("KNN") ? "Instance-based local neighborhoods; sensitive to scaling & distance" :
          r.name.includes("Logistic") ? "Linear log-odds probability; simple and interpretable" :
          r.name.includes("Naive") ? "Fast probabilistic baseline; assumes feature independence" :
          "Max-margin boundary; effective in high-dimensional space";
        return `<tr>
          <td>${i + 1} ${isWinner ? '⭐' : ''}</td>
          <td><b>${esc(r.name)}</b></td>
          <td><b>${pc(r.accuracy)}</b></td>
          <td>${pc(r.precision_macro)}</td>
          <td>${pc(r.recall_macro)}</td>
          <td><span class="tag ${r.f1_macro >= 0.7 ? 't-hi' : 't-mid'}"><b>${pc(r.f1_macro)}</b></span></td>
          <td>${pc(r.f1_weighted)}</td>
          <td>${r.training_time_s}s</td>
          <td>${tier} <small class="muted">(${tradeOff})</small></td>
        </tr>`;
      }).join("");
  } else {
    $("cmpClfTable").innerHTML = "<tr><td colspan='9'>No classification algorithms selected.</td></tr>";
  }

  // Regression Table
  if (regList.length) {
    $("cmpRegTable").innerHTML = "<tr><th>Rank</th><th>Algorithm</th><th>R² Score (Explained Var)</th><th>MAE (Mean Error)</th><th>RMSE (Outlier Penalized)</th><th>MSE</th><th>Fit Time</th><th>Fit Assessment &amp; Trade-offs</th></tr>" +
      regList.map((r, i) => {
        const isWinner = (i === 0);
        const tier = isWinner ? '<span class="tag t-hi">🏆 Best Fit</span>' : r.r2 >= 0.8 ? '<span class="tag t-mid">High Fit</span>' : '<span class="tag t-lo">Moderate</span>';
        const tradeOff = r.name.includes("Polynomial") ? "Captures non-linear Price × Qty multiplicative revenue curve" :
          r.name.includes("Forest") ? "Aggregates partitioned subtrees; robust to extreme residuals" :
          r.name.includes("Tree") ? "Piecewise constant step predictions; fast and non-linear" :
          "Standard linear hyperplane; baseline assumption of constant marginal effects";
        return `<tr>
          <td>${i + 1} ${isWinner ? '⭐' : ''}</td>
          <td><b>${esc(r.name)}</b></td>
          <td><b>${r.r2.toFixed(4)}</b> <small class="muted">(${Math.max(0, r.r2 * 100).toFixed(1)}% var)</small></td>
          <td>$${r.mae.toFixed(2)}</td>
          <td>$${r.rmse.toFixed(2)}</td>
          <td>${r.mse.toFixed(2)}</td>
          <td>${r.training_time_s}s</td>
          <td>${tier} <small class="muted">(${tradeOff})</small></td>
        </tr>`;
      }).join("");
  } else {
    $("cmpRegTable").innerHTML = "<tr><td colspan='8'>No regression algorithms selected.</td></tr>";
  }

  // Clustering Table
  if (cluList.length) {
    $("cmpCluTable").innerHTML = "<tr><th>Rank</th><th>Algorithm</th><th>Identified Clusters</th><th>Silhouette Score</th><th>Separation Assessment</th><th>Inertia (WCSS)</th><th>Outliers / Noise</th><th>Execution Time</th><th>Topology &amp; Characteristics</th></tr>" +
      cluList.map((r, i) => {
        const isWinner = (i === 0);
        const sil = r.silhouette_score;
        const tier = isWinner ? '<span class="tag t-hi">🏆 Optimal</span>' : (sil && sil > 0.4) ? '<span class="tag t-mid">Moderate</span>' : '<span class="tag t-lo">Baseline</span>';
        const assessment = (sil === null) ? "N/A" : sil >= 0.5 ? "Strong separation" : sil >= 0.25 ? "Reasonable structure" : "Weak / Overlapping";
        const tradeOff = r.name.includes("K-Means") ? "Assumes spherical convex clusters; minimizes inertia" :
          r.name.includes("Agglomerative") ? "Hierarchical ward linkage; deterministic tree structure" :
          "Density-based; detects arbitrary cluster shapes and isolates noise";
        return `<tr>
          <td>${i + 1} ${isWinner ? '⭐' : ''}</td>
          <td><b>${esc(r.name)}</b></td>
          <td><b>${r.n_clusters} clusters</b></td>
          <td>${sil !== null ? `<b>${sil.toFixed(4)}</b>` : '–'}</td>
          <td><span class="tag ${sil >= 0.5 ? 't-hi' : 't-mid'}">${assessment}</span></td>
          <td>${r.inertia ? nf(r.inertia) : '–'}</td>
          <td>${r.n_noise > 0 ? `<span class="tag t-lo">${nf(r.n_noise)} pts</span>` : '0'}</td>
          <td>${r.training_time_s}s</td>
          <td>${tier} <small class="muted">(${tradeOff})</small></td>
        </tr>`;
      }).join("");
  } else {
    $("cmpCluTable").innerHTML = "<tr><td colspan='9'>No clustering algorithms selected.</td></tr>";
  }

  // Association Rules Table
  if (rules && rules.available && rules.sample_rules && rules.sample_rules.length) {
    $("cmpRulesTable").innerHTML = "<tr><th>#</th><th>Antecedent (If Bought)</th><th>Consequent (Then Bought)</th><th>Support</th><th>Confidence</th><th>Lift Ratio</th><th>Leverage</th><th>Retail Strategy</th></tr>" +
      rules.sample_rules.slice(0, 15).map((r, i) => {
        const a = r.antecedents.join(" + ");
        const c = r.consequents.join(" + ");
        const strat = r.lift >= 3.0 ? "Immediate combo bundle & checkout recommendation" :
          r.lift >= 2.0 ? "Co-locate items on adjacent supermarket shelves" :
          "Cross-category promotional discount coupon";
        return `<tr>
          <td>${i + 1}</td>
          <td><b>${esc(a)}</b></td>
          <td><b>${esc(c)}</b></td>
          <td>${pc(r.support, 2)}</td>
          <td>${pc(r.confidence)}</td>
          <td>${liftTag(r.lift)}</td>
          <td>${r.leverage !== undefined ? r.leverage.toFixed(4) : '–'}</td>
          <td><small class="muted">${strat}</small></td>
        </tr>`;
      }).join("");
  } else {
    $("cmpRulesTable").innerHTML = "<tr><td colspan='8'>No association rules generated.</td></tr>";
  }

  // Preprocessing Table
  if (prep && prep.available) {
    const b = prep.stats_before;
    const a = prep.stats_after;
    const d = prep.deltas;
    $("cmpPrepTable").innerHTML = "<tr><th>Data Dimension / Quality Aspect</th><th>Baseline (Before)</th><th>Cleaned (After)</th><th>Net Change / Delta</th><th>Downstream ML Impact</th></tr>" +
      [
        { dim: "Total Instance Rows", before: nf(b.rows), after: nf(a.rows), delta: `${(a.rows - b.rows >= 0 ? '+' : '')}${nf(a.rows - b.rows)} (${d.rows_retained_pct}% kept)`, impact: "Preserves training sample volume while removing invalid records" },
        { dim: "Total Feature Columns", before: b.columns, after: a.columns, delta: `${a.columns - b.columns >= 0 ? '+' : ''}${a.columns - b.columns} cols`, impact: "Provides structured numerical and categorical dimensions for models" },
        { dim: "Missing / Null Cells", before: nf(b.missing_values), after: nf(a.missing_values), delta: `-${nf(b.missing_values - a.missing_values)} (${d.missing_reduction_pct}% reduction)`, impact: "Prevents calculation errors and estimator bias during model training" },
        { dim: "Duplicate Transactions", before: nf(b.duplicate_rows), after: nf(a.duplicate_rows), delta: `-${nf(d.duplicates_removed)} removed`, impact: "Eliminates artificial frequency inflation in rule mining and clustering" },
        { dim: "Overall Data Hygiene Score", before: `${b.cleanliness_score}%`, after: `${a.cleanliness_score}%`, delta: `+${(a.cleanliness_score - b.cleanliness_score).toFixed(1)}%`, impact: "Ensures garbage-in garbage-out protection across all 5 techniques" }
      ].map(r => `<tr>
        <td><b>${r.dim}</b></td>
        <td>${r.before}</td>
        <td><b>${r.after}</b></td>
        <td><span class="tag t-hi">${r.delta}</span></td>
        <td><small class="muted">${r.impact}</small></td>
      </tr>`).join("");
  } else {
    $("cmpPrepTable").innerHTML = "<tr><td colspan='5'>Preprocessing profile not available.</td></tr>";
  }

  // Holistic Cross-Technique Matrix Table
  if (holistic && holistic.matrix) {
    $("cmpHolisticTable").innerHTML = "<tr><th>Technique Category</th><th>Learning Paradigm</th><th>Primary Objective</th><th>Appropriate Primary Metric</th><th>Best Model / Observed Score</th><th>Runtime Speed</th><th>Supermarket Retail Value</th></tr>" +
      holistic.matrix.map(m => `<tr>
        <td><b>${esc(m.technique)}</b></td>
        <td><span class="metric-cat-pill ${m.technique.includes('Class') ? 'cat-clf' : m.technique.includes('Reg') ? 'cat-reg' : m.technique.includes('Clust') ? 'cat-clu' : m.technique.includes('Rule') ? 'cat-rules' : 'cat-prep'}">${esc(m.category)}</span></td>
        <td class="wrap">${esc(m.objective)}</td>
        <td><b>${esc(m.primary_metric_name)}</b></td>
        <td><span class="tag t-hi">${esc(m.best_model)}</span><br><small><b>${esc(m.score_display)}</b></small></td>
        <td><code>${esc(m.speed)}</code></td>
        <td class="wrap"><small>${esc(m.business_impact)}</small></td>
      </tr>`).join("");
    $("cmpHolisticSummary").innerHTML = holistic.summary;
  }
}

/* ---------- Application Boot ---------- */
(async function () {
  try {
    const c = await api("/api/config");
    $("statusPill").textContent = "Backend connected · " + c.engine;
    $("statusPill").className = "pill ok";
    $("pSup").value = c.defaults.min_support;
    $("pConf").value = c.defaults.min_confidence;
    $("pLift").value = c.defaults.min_lift;
  } catch (e) {
    $("statusPill").textContent = "Backend not reachable";
    $("statusPill").className = "pill bad";
  }

  api("/api/dataset").then(d => {
    $("srcName").textContent = d.source;
  }).catch(() => { });

  loaded.dashboard = true;
  loaded.overview = true;
  loadOverview().catch(e => {
    loaded.dashboard = false;
    toast(e.message, true);
  });
})();
