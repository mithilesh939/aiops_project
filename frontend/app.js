/* QuizPulse UI - talks to the backend ONLY through REST calls to CFG.API_BASE_URL. */
"use strict";
const CFG = window.QUIZPULSE_CONFIG;
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
let currentStudent = null, lastPrediction = null;

/* ---------- helpers ---------- */
async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(CFG.API_BASE_URL + path, { headers: { "Content-Type": "application/json" }, ...opts });
  } catch (e) { setConn(false); throw new Error("Cannot reach the QuizPulse API. Is the backend running?"); }
  setConn(true);
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}
function toast(msg, isErr) {
  const t = $("toast"); t.textContent = msg; t.className = "toast" + (isErr ? " err" : "");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), 4500);
}
function setConn(up) {
  const c = $("conn"); c.className = "conn " + (up ? "up" : "down");
  c.querySelector("span").textContent = up ? "API connected" : "API unreachable";
}
const pct = (x) => (x == null ? "-" : Math.round(x * 100) + "%");
const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
function barColor(m) { return m < 0.5 ? "var(--weak-bar)" : m < 0.65 ? "var(--mid-bar)" : "var(--good-bar)"; }
async function busy(btn, fn) {
  const old = btn.textContent; btn.disabled = true; btn.textContent = "Working...";
  try { return await fn(); } catch (e) { toast(e.message, true); } finally { btn.disabled = false; btn.textContent = old; }
}

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === b.dataset.tab));
  if (b.dataset.tab === "console") refreshConsole();
}));

/* ---------- student view ---------- */
async function loadStudents(q = "") {
  try {
    const list = await api("/students?limit=15&q=" + encodeURIComponent(q));
    const dl = $("student-list"); dl.replaceChildren();
    list.forEach((s) => { const o = el("option"); o.value = s.student_id; o.label = s.attempts + " attempts"; dl.appendChild(o); });
    if (!$("student-id").value && list.length) $("student-id").placeholder = "Student ID, e.g. " + list[0].student_id;
  } catch (_) { /* connection badge already shows the problem */ }
}
$("student-id").addEventListener("input", (e) => loadStudents(e.target.value.trim()));

$("student-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const id = $("student-id").value.trim(); if (!id) return;
  const btn = e.target.querySelector("button");
  await busy(btn, async () => {
    const d = await api(`/students/${encodeURIComponent(id)}/mastery`);
    currentStudent = id; renderMastery(d);
    $("learn-empty").classList.add("hidden"); $("learn-body").classList.remove("hidden");
    $("next-list").replaceChildren(); $("predict-result").classList.add("hidden");
    loadSkills();
  });
});

function renderMastery(d) {
  $("mastery-version").textContent = "model v" + d.model_version;
  const ul = $("mastery-list"); ul.replaceChildren();
  d.topics.forEach((t) => {
    const li = el("li");
    const name = el("span", "name", t.skill); name.title = t.attempts ? `${t.attempts} past answers, ${pct(t.observed_accuracy)} correct` : "no answers yet";
    const bar = el("div", "bar"); const fill = el("span"); fill.style.width = Math.round(t.mastery * 100) + "%"; fill.style.background = barColor(t.mastery); bar.appendChild(fill);
    const val = el("span", "val", pct(t.mastery));
    if (t.weak) { const tag = el("div", "tag", "WEAK"); val.appendChild(tag); }
    li.append(name, bar, val); ul.appendChild(li);
  });
}

$("btn-next").addEventListener("click", (e) => busy(e.target, async () => {
  const d = await api("/next-questions", { method: "POST", body: JSON.stringify({ student_id: currentStudent, n: 5 }) });
  const ol = $("next-list"); ol.replaceChildren();
  if (!d.questions.length) { ol.appendChild(el("li", "", "No suggestions yet.")); return; }
  d.questions.forEach((q) => { const li = el("li"); li.append(el("b", "", `${q.skill} - ${q.question_id}`)); li.appendChild(el("small", "", `${pct(q.p_correct)} chance of success`)); ol.appendChild(li); });
}));

async function loadSkills() {
  const skills = await api("/skills"); const s = $("sel-skill"); s.replaceChildren();
  skills.forEach((k) => { const o = el("option", "", k); o.value = k; s.appendChild(o); });
  await loadQuestions();
}
async function loadQuestions() {
  const qs = await api("/questions?skill=" + encodeURIComponent($("sel-skill").value)); const s = $("sel-question"); s.replaceChildren();
  qs.forEach((k) => { const o = el("option", "", k); o.value = k; s.appendChild(o); });
}
$("sel-skill").addEventListener("change", () => loadQuestions().catch((e) => toast(e.message, true)));

$("btn-predict").addEventListener("click", (e) => busy(e.target, async () => {
  const body = { student_id: currentStudent, skill: $("sel-skill").value, question_id: $("sel-question").value };
  const d = await api("/predict", { method: "POST", body: JSON.stringify(body) });
  lastPrediction = d;
  $("predict-result").classList.remove("hidden");
  $("gauge-fill").style.width = Math.round(d.p_correct * 100) + "%"; $("gauge-fill").style.background = barColor(d.p_correct);
  $("predict-text").textContent = `About ${pct(d.p_correct)} chance that ${d.student_id} answers ${d.question_id} correctly.`;
}));
document.querySelectorAll("#predict-result [data-correct]").forEach((b) => b.addEventListener("click", () => busy(b, async () => {
  if (!lastPrediction) return;
  await api("/outcomes", { method: "POST", body: JSON.stringify({ prediction_id: lastPrediction.prediction_id, correct: Number(b.dataset.correct) }) });
  toast("Thanks - the answer was recorded."); lastPrediction = null; $("predict-result").classList.add("hidden");
  const d = await api(`/students/${encodeURIComponent(currentStudent)}/mastery`); renderMastery(d);
})));

/* ---------- pipeline console ---------- */
function stageClass(st) { return !st ? "idle" : (st.failed ? "fail" : st.success ? "ok" : "idle"); }
function renderFlow(status) {
  const f = $("flow"); f.replaceChildren();
  const stages = [["ingest", "Data ingest", "validate + load history"], ["train", "Train", "3 models in MLflow"], ["register", "Register", "production alias"],
    ["serve", "Serve", "FastAPI"], ["drift", "Drift check", "PSI + Evidently"], ["retrain", "Retrain", "gate + promote"]];
  stages.forEach(([k, title, sub], i) => {
    const st = status[k]; const cls = k === "serve" ? "ok" : k === "train" || k === "register" ? "ok" : stageClass(st);
    const s = el("div", "step " + cls); s.append(el("b", "", title), el("small", "", sub));
    const n = st ? Object.entries(st).map(([a, v]) => `${v.count} ${a}`).join(", ") : (k === "train" || k === "register" ? "via MLflow" : "no runs yet");
    s.appendChild(el("small", "", n)); f.appendChild(s); if (i < stages.length - 1) f.appendChild(el("span", "arrow", "->"));
  });
}
function renderModels(info) {
  const tb = document.querySelector("#tbl-models tbody"); tb.replaceChildren();
  info.versions.forEach((v) => {
    const tr = el("tr"); const alias = v.aliases.length ? v.aliases.join(", ") : "-";
    [String(v.version), alias, v.stage || "-", v.model_kind || "-", v.valid_auc ? v.valid_auc.toFixed(4) : "-", v.git_commit || "-"].forEach((x) => tr.appendChild(el("td", "", x)));
    if (String(v.version) === String(info.serving_version)) tr.style.fontWeight = "700";
    tb.appendChild(tr);
  });
}
function renderStatus(status) {
  const tb = document.querySelector("#tbl-status tbody"); tb.replaceChildren();
  Object.entries(status).forEach(([stage, m]) => {
    const tr = el("tr"); const g = (k) => (m[k] ? m[k].count : 0);
    const avg = Object.values(m).reduce((a, v) => a + v.avg_duration_s, 0) / Object.keys(m).length;
    [stage, g("success"), g("failed"), g("skipped"), avg.toFixed(1) + " s"].forEach((x) => tr.appendChild(el("td", "", String(x)))); tb.appendChild(tr);
  });
}
function describeRun(r) {
  const d = r.details || {};
  if (r.kind === "drift") return d.status === "skipped" ? d.reason : (d.drift_detected ? "DRIFT: " + (d.reasons || []).join("; ") : "no drift");
  if (r.kind === "retrain" || r.kind === "promotion") return d.status ? `${d.status}${d.new_version ? " -> v" + d.new_version : ""}${d.challenger_holdout ? ` (AUC ${d.challenger_holdout.auc} vs ${d.production_holdout.auc})` : ""}${d.reason && d.status === "skipped" ? " - " + d.reason : ""}` : "-";
  if (r.kind === "ingest") return `${r.rows} rows loaded`;
  return "";
}
function renderRuns(runs) {
  const tb = document.querySelector("#tbl-runs tbody"); tb.replaceChildren();
  if (!runs.length) { const tr = el("tr"); const td = el("td", "muted", "No runs yet."); td.colSpan = 7; tr.appendChild(td); tb.appendChild(tr); return; }
  runs.forEach((r) => {
    const tr = el("tr"); const badge = el("span", "badge " + r.status, r.status);
    const st = el("td"); st.appendChild(badge);
    const d = r.details || {}; if (d.status === "promoted" || d.status === "rejected") { st.replaceChildren(el("span", "badge " + d.status, d.status)); }
    tr.append(el("td", "", String(r.id)), el("td", "", fmtTime(r.started)), el("td", "", r.kind), st, el("td", "", (r.duration_s ?? 0).toFixed(1) + " s"), el("td", "", String(r.rows ?? "")), el("td", "", describeRun(r)));
    tb.appendChild(tr);
  });
}
function renderThroughput(t) {
  const box = $("throughput"); box.replaceChildren();
  const add = (label, value) => { const d = el("div", "pill", `${label}: ${value}`); box.appendChild(d); };
  if (t.data_quality) { add("data quality", t.data_quality.passed ? "passed" : "FAILED"); add("rows ingested", t.data_quality.rows); add("ingest speed", t.data_quality.throughput_rows_per_s + " rows/s"); }
  if (t.feature_stats) { add("feature engine", t.feature_stats.engine); add("features speed", t.feature_stats.throughput_rows_per_s + " rows/s"); }
  if (!box.children.length) box.textContent = "no pipeline report found yet (run dvc repro)";
}
async function refreshConsole() {
  try {
    const [sum, status, runs, info] = await Promise.all([api("/monitoring/summary"), api("/pipeline/status"),
      api("/pipeline/runs?limit=30" + ($("run-filter").value ? "&kind=" + $("run-filter").value : "")), api("/model/info").catch(() => null)]);
    api("/pipeline/throughput").then(renderThroughput).catch(() => {});
    $("k-version").textContent = sum.serving_version ? "v" + sum.serving_version : "none";
    const cur = info && info.versions.find((v) => String(v.version) === String(sum.serving_version));
    $("k-kind").textContent = cur ? `${cur.model_kind || ""} (${cur.stage || ""})` : "";
    $("k-window").textContent = (sum.rolling && sum.rolling.n) || 0;
    const auc = sum.rolling && sum.rolling.auc; const k = $("k-auc"); k.textContent = auc == null ? "n/a" : auc.toFixed(3);
    k.className = auc == null ? "" : auc >= 0.72 ? "good" : "badc";
    $("k-auc-sub").textContent = auc == null ? "needs 50+ labelled answers" : `log-loss ${sum.rolling.logloss}`;
    const ld = sum.last_drift && sum.last_drift.details; const dk = $("k-drift");
    if (!ld || ld.status !== "ok") { dk.textContent = "unknown"; dk.className = "warnc"; $("k-drift-sub").textContent = ld ? ld.reason : "no check yet"; }
    else { dk.textContent = ld.drift_detected ? "DRIFT" : "stable"; dk.className = ld.drift_detected ? "badc" : "good"; $("k-drift-sub").textContent = ld.drift_detected ? (ld.reasons || [])[0] : `${ld.drifted_features.length} drifted features`; }
    $("k-counts").textContent = `${sum.counts.predictions} / ${sum.counts.outcomes}`; $("k-counts-sub").textContent = `${sum.counts.attempts} attempts stored`;
    renderFlow(status); renderStatus(status); renderRuns(runs); if (info) renderModels(info);
    $("live").textContent = "live " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch (e) { $("live").textContent = "offline"; }
}
$("run-filter").addEventListener("change", refreshConsole);
$("act-drift").addEventListener("click", (e) => busy(e.target, async () => {
  const r = await api("/admin/drift-check", { method: "POST" });
  $("act-msg").textContent = r.status === "skipped" ? "Skipped: " + r.reason : (r.drift_detected ? "Drift detected: " + r.reasons.join("; ") : "No drift detected.");
  refreshConsole();
}));
$("act-retrain").addEventListener("click", (e) => busy(e.target, async () => {
  $("act-msg").textContent = "Training a challenger and comparing it with production...";
  const r = await api("/admin/retrain", { method: "POST" });
  $("act-msg").textContent = r.status === "promoted" ? `New model v${r.new_version} is now in use (AUC ${r.challenger_holdout.auc} vs ${r.production_holdout.auc}).`
    : `Challenger v${r.new_version} was NOT better, v${r.previous_version} stays in use.`;
  refreshConsole();
}));
$("act-reload").addEventListener("click", (e) => busy(e.target, async () => {
  const r = await api("/admin/reload", { method: "POST" }); toast(r.reloaded ? "Loaded version " + r.serving_version : "Already on the latest version " + r.serving_version); refreshConsole();
}));
$("lnk-report").href = CFG.API_BASE_URL + "/reports/drift"; $("lnk-mlflow").href = CFG.MLFLOW_URL; $("lnk-grafana").href = CFG.GRAFANA_URL; $("lnk-prom").href = CFG.PROMETHEUS_URL;

/* ---------- start (supports deep links: #console, #help, ?student=S0001) ---------- */
loadStudents();
const hash = location.hash.replace("#", "");
if (["console", "help"].includes(hash)) document.querySelector(`.tab[data-tab="${hash}"]`).click();
const qs = new URLSearchParams(location.search).get("student");
if (qs) { $("student-id").value = qs; $("student-form").requestSubmit(); }
setInterval(() => { if ($("console").classList.contains("active") && !document.hidden) refreshConsole(); }, CFG.REFRESH_SECONDS * 1000);
