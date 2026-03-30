// ── AegeanBench Dashboard — app.js (part 1: data + charts) ──

const PALETTE = {
  accent:  '#00d4ff',
  accent2: '#7c3aed',
  accent3: '#10b981',
  warn:    '#f59e0b',
  danger:  '#ef4444',
  dim:     '#6b7fa3',
};

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: true,
  plugins: {
    legend: {
      labels: { color: '#6b7fa3', font: { family: 'JetBrains Mono', size: 11 } }
    },
    tooltip: {
      backgroundColor: '#0e1520',
      borderColor: '#1e2d42',
      borderWidth: 1,
      titleColor: '#e2eaf4',
      bodyColor: '#6b7fa3',
      titleFont: { family: 'Syne', weight: '700' },
      bodyFont:  { family: 'JetBrains Mono', size: 11 },
    }
  },
  scales: {
    x: { ticks: { color: '#6b7fa3', font: { family: 'JetBrains Mono', size: 10 } },
         grid:  { color: 'rgba(30,45,66,.5)' } },
    y: { ticks: { color: '#6b7fa3', font: { family: 'JetBrains Mono', size: 10 } },
         grid:  { color: 'rgba(30,45,66,.5)' }, beginAtZero: true },
  }
};

let charts = {};

function destroyCharts() {
  Object.values(charts).forEach(c => c && c.destroy());
  charts = {};
}

// ── Demo data generator ──────────────────────────────────────
function makeDemoData() {
  const categories   = ['consensus','consensus','consensus','collaboration','collaboration','risk','risk','hybrid'];
  const difficulties = ['easy','medium','hard'];
  const outcomes     = ['converged','early_stop','diverged'];

  const results = Array.from({ length: 44 }, (_, i) => {
    const cat  = categories[i % categories.length];
    const diff = difficulties[i % 3];
    const correct = Math.random() > (diff === 'hard' ? 0.25 : diff === 'medium' ? 0.15 : 0.05);
    const tok_p = cat === 'risk' ? (Math.random() < 0.25 ? 0 : 400 * (1 + Math.random())) : 200 * (1 + Math.random() * 3);
    const tok_c = tok_p * 0.25;
    const tok_s = Math.random() < 0.4 ? tok_p * 0.3 : 0;
    return {
      case_id:   `CASE-${String(i+1).padStart(3,'0')}`,
      case_name: `${cat.charAt(0).toUpperCase()+cat.slice(1)} Case ${i+1}`,
      category:  cat,
      difficulty: diff,
      outcome:   correct ? (Math.random() < 0.5 ? 'early_stop' : 'converged') : 'diverged',
      correct,
      latency_s: 0.03 + Math.random() * 0.2,
      tokens_prompt:     Math.round(tok_p),
      tokens_completion: Math.round(tok_c),
      tokens_saved:      Math.round(tok_s),
      consensus_metrics: cat === 'consensus' ? {
        quorum_reached: correct,
        consensus_confidence: correct ? 0.7 + Math.random() * 0.29 : 0.3 + Math.random() * 0.3,
        disagreement_score:   correct ? Math.random() * 0.3 : 0.4 + Math.random() * 0.5,
        early_stop_triggered: Math.random() < 0.5,
        quorum_efficiency:    0.3 + Math.random() * 0.7,
        outlier_detected:     Math.random() < 0.7,
        rounds_used: Math.ceil(Math.random() * 4),
        tokens_prompt_total: Math.round(tok_p),
        tokens_completion_total: Math.round(tok_c),
      } : null,
      risk_metrics: cat === 'risk' ? {
        decision_correct: correct,
        risk_level_correct: correct,
        pre_screen_triggered: Math.random() < 0.25,
        validator_agreement: 0.6 + Math.random() * 0.4,
        tokens_prompt_total: Math.round(tok_p),
        tokens_completion_total: Math.round(tok_c),
        tokens_saved_by_prescreen: Math.round(tok_s),
      } : null,
    };
  });

  const total   = results.length;
  const passed  = results.filter(r => r.correct).length;
  const con_res = results.filter(r => r.consensus_metrics);
  const risk_res= results.filter(r => r.risk_metrics);
  const all_tok = results.reduce((s,r) => s + r.tokens_prompt + r.tokens_completion, 0);
  const all_saved = results.reduce((s,r) => s + r.tokens_saved, 0);

  return {
    suite_name: 'AegeanBench Demo Run',
    run_id: 'demo-run-001',
    total_cases: total,
    passed, failed: total-passed, errored: 0,
    accuracy: passed/total,
    mean_latency_s: 0.08,
    p50_latency_s:  0.07,
    p95_latency_s:  0.19,
    total_tokens_prompt:     results.reduce((s,r)=>s+r.tokens_prompt,0),
    total_tokens_completion: results.reduce((s,r)=>s+r.tokens_completion,0),
    total_tokens: all_tok,
    total_tokens_saved: all_saved,
    token_efficiency_rate: all_saved / (all_tok + all_saved),
    mean_tokens_per_case: all_tok / total,
    p50_tokens_per_case: 600,
    p95_tokens_per_case: 1400,
    mean_tokens_per_correct: all_tok / passed,
    consensus_rate:          con_res.filter(r=>r.consensus_metrics.quorum_reached).length / (con_res.length||1),
    mean_consensus_conf:     con_res.reduce((s,r)=>s+r.consensus_metrics.consensus_confidence,0)/(con_res.length||1),
    mean_disagreement:       con_res.reduce((s,r)=>s+r.consensus_metrics.disagreement_score,0)/(con_res.length||1),
    early_stop_rate:         con_res.filter(r=>r.consensus_metrics.early_stop_triggered).length/(con_res.length||1),
    mean_quorum_efficiency:  con_res.reduce((s,r)=>s+r.consensus_metrics.quorum_efficiency,0)/(con_res.length||1),
    outlier_detection_rate:  0.85,
    risk_accuracy:           risk_res.filter(r=>r.risk_metrics.decision_correct).length/(risk_res.length||1),
    risk_level_accuracy:     risk_res.filter(r=>r.risk_metrics.risk_level_correct).length/(risk_res.length||1),
    risk_f1_approve: 0.88, risk_f1_reject: 0.82,
    mean_validator_agreement: risk_res.reduce((s,r)=>s+r.risk_metrics.validator_agreement,0)/(risk_res.length||1),
    pre_screen_rate: risk_res.filter(r=>r.risk_metrics.pre_screen_triggered).length/(risk_res.length||1),
    mean_subtask_completion: 0.93, mean_milestone_hit: 0.91, mean_coordination: 0.87,
    results,
  };
}

// ── Chart builders ────────────────────────────────────────────
function buildCategoryChart(data) {
  const cats = {};
  data.results.forEach(r => {
    if (!cats[r.category]) cats[r.category] = { total:0, correct:0 };
    cats[r.category].total++;
    if (r.correct) cats[r.category].correct++;
  });
  const labels = Object.keys(cats);
  const values = labels.map(l => ((cats[l].correct/cats[l].total)*100).toFixed(1));
  const colors = labels.map(l => ({
    consensus:'rgba(0,212,255,.7)', collaboration:'rgba(124,58,237,.7)',
    hybrid:'rgba(16,185,129,.7)', risk:'rgba(245,158,11,.7)',
  }[l] || PALETTE.dim));

  const ctx = document.getElementById('chart-category').getContext('2d');
  charts.category = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Accuracy %', data: values,
      backgroundColor: colors, borderRadius: 6, borderSkipped: false }] },
    options: { ...CHART_DEFAULTS, plugins: { ...CHART_DEFAULTS.plugins,
      legend: { display: false } },
      scales: { ...CHART_DEFAULTS.scales, y: { ...CHART_DEFAULTS.scales.y, max: 100 } } },
  });
}

function buildDifficultyChart(data) {
  const diffs = {};
  data.results.forEach(r => {
    if (!diffs[r.difficulty]) diffs[r.difficulty] = { total:0, correct:0 };
    diffs[r.difficulty].total++;
    if (r.correct) diffs[r.difficulty].correct++;
  });
  const order = ['easy','medium','hard'];
  const labels = order.filter(d => diffs[d]);
  const values = labels.map(l => ((diffs[l].correct/diffs[l].total)*100).toFixed(1));
  const colors = labels.map(l => ({
    easy: 'rgba(16,185,129,.7)', medium: 'rgba(245,158,11,.7)', hard: 'rgba(239,68,68,.7)'
  }[l]));

  const ctx = document.getElementById('chart-difficulty').getContext('2d');
  charts.difficulty = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Accuracy %', data: values,
      backgroundColor: colors, borderRadius: 6, borderSkipped: false }] },
    options: { ...CHART_DEFAULTS, plugins: { ...CHART_DEFAULTS.plugins,
      legend: { display: false } },
      scales: { ...CHART_DEFAULTS.scales, y: { ...CHART_DEFAULTS.scales.y, max: 100 } } },
  });
}

function buildConsensusChart(data) {
  const labels = ['Consensus Rate','Mean Confidence','Early Stop Rate',
                  'Quorum Efficiency','Outlier Detection','1 - Disagreement'];
  const values = [
    (data.consensus_rate*100).toFixed(1),
    (data.mean_consensus_conf*100).toFixed(1),
    (data.early_stop_rate*100).toFixed(1),
    (data.mean_quorum_efficiency*100).toFixed(1),
    (data.outlier_detection_rate*100).toFixed(1),
    ((1-data.mean_disagreement)*100).toFixed(1),
  ];
  const ctx = document.getElementById('chart-consensus').getContext('2d');
  charts.consensus = new Chart(ctx, {
    type: 'radar',
    data: { labels, datasets: [{
      label: 'Consensus Metrics (%)', data: values,
      backgroundColor: 'rgba(0,212,255,.08)',
      borderColor: PALETTE.accent, pointBackgroundColor: PALETTE.accent,
      pointBorderColor: '#0e1520', pointRadius: 4, borderWidth: 2,
    }]},
    options: { responsive: true, maintainAspectRatio: true,
      plugins: { legend: { labels: { color: '#6b7fa3', font: { family:'JetBrains Mono', size:10 } } },
        tooltip: CHART_DEFAULTS.plugins.tooltip },
      scales: { r: { min:0, max:100, ticks: { stepSize:20, color:'#6b7fa3',
          font:{ family:'JetBrains Mono',size:9 }, backdropColor:'transparent' },
        grid:{ color:'rgba(30,45,66,.6)' }, pointLabels:{ color:'#6b7fa3',
          font:{ family:'JetBrains Mono',size:10 } } } } },
  });
}

function buildTokenChart(data) {
  const labels = ['Prompt','Completion','Saved'];
  const values = [data.total_tokens_prompt, data.total_tokens_completion, data.total_tokens_saved];
  const ctx = document.getElementById('chart-tokens').getContext('2d');
  charts.tokens = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data: values,
      backgroundColor: ['rgba(0,212,255,.7)','rgba(124,58,237,.7)','rgba(16,185,129,.7)'],
      borderColor: '#0e1520', borderWidth: 2, hoverOffset: 8 }]},
    options: { responsive: true, maintainAspectRatio: true, cutout: '62%',
      plugins: { legend: { position:'bottom', labels:{ color:'#6b7fa3',
          font:{family:'JetBrains Mono',size:10}, padding:12 } },
        tooltip: CHART_DEFAULTS.plugins.tooltip } },
  });
}

function buildRiskChart(data) {
  const labels = ['Decision Acc.','Risk Level Acc.','F1 Approve','F1 Reject','Validator Agree','Pre-screen Rate'];
  const values = [
    (data.risk_accuracy*100).toFixed(1),
    (data.risk_level_accuracy*100).toFixed(1),
    (data.risk_f1_approve*100).toFixed(1),
    (data.risk_f1_reject*100).toFixed(1),
    (data.mean_validator_agreement*100).toFixed(1),
    (data.pre_screen_rate*100).toFixed(1),
  ];
  const ctx = document.getElementById('chart-risk').getContext('2d');
  charts.risk = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Risk Metrics (%)', data: values,
      backgroundColor: 'rgba(245,158,11,.6)', borderColor: PALETTE.warn,
      borderWidth: 1, borderRadius: 5, borderSkipped: false }]},
    options: { indexAxis: 'y', ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins, legend: { display: false } },
      scales: { ...CHART_DEFAULTS.scales, x: { ...CHART_DEFAULTS.scales.x, max:100 } } },
  });
}

function buildLatencyChart(data) {
  const lats = data.results.map(r => (r.latency_s * 1000).toFixed(0));
  const labels = data.results.map(r => r.case_id);
  const colors = data.results.map(r => r.correct ? 'rgba(16,185,129,.6)' : 'rgba(239,68,68,.5)');
  const ctx = document.getElementById('chart-latency').getContext('2d');
  charts.latency = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Latency (ms)', data: lats,
      backgroundColor: colors, borderRadius: 3, borderSkipped: false }]},
    options: { ...CHART_DEFAULTS, plugins: { ...CHART_DEFAULTS.plugins,
      legend: { display: false } } },
  });
}

// ── AegeanBench Dashboard — app.js (part 2: KPIs + table + events) ──

function fmt(n, decimals=1) {
  if (n === undefined || n === null || isNaN(n)) return '—';
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: decimals });
}
function pct(n) { return n !== undefined ? (n*100).toFixed(1)+'%' : '—'; }
function fmtTok(n) {
  if (!n && n !== 0) return '—';
  return n >= 1000 ? (n/1000).toFixed(1)+'k' : String(n);
}

function updateKPIs(data) {
  const set = (id, val, sub) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.querySelector('.kpi-value').textContent = val;
    if (sub) el.querySelector('.kpi-sub').textContent = sub;
  };
  set('kpi-accuracy',  pct(data.accuracy),
      `${data.passed} / ${data.total_cases} cases`);
  set('kpi-consensus', pct(data.consensus_rate),
      `early-stop ${pct(data.early_stop_rate)}`);
  set('kpi-tokens',    fmtTok(data.total_tokens),
      `prompt ${fmtTok(data.total_tokens_prompt)} + compl ${fmtTok(data.total_tokens_completion)}`);
  set('kpi-saved',     fmtTok(data.total_tokens_saved),
      `efficiency ${pct(data.token_efficiency_rate)}`);
  set('kpi-latency',   fmt(data.p95_latency_s, 3)+'s',
      `p50 ${fmt(data.p50_latency_s, 3)}s  mean ${fmt(data.mean_latency_s, 3)}s`);
}

// ── Table rendering ──────────────────────────────────────────
let allResults = [];

function badge(cls, text) {
  return `<span class="badge badge-${cls}">${text}</span>`;
}

function renderTable(results) {
  const tbody = document.getElementById('results-body');
  if (!results.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="8">No cases match filters.</td></tr>';
    return;
  }
  tbody.innerHTML = results.map(r => {
    const catBadge  = badge(r.category, r.category);
    const diffBadge = badge(r.difficulty, r.difficulty);
    const passBadge = badge(r.correct ? 'pass' : 'fail', r.correct ? '✓ pass' : '✗ fail');
    const tokens    = r.tokens_prompt + r.tokens_completion;
    const saved     = r.tokens_saved ? `<span style="color:var(--accent3)"> +${fmtTok(r.tokens_saved)} saved</span>` : '';
    return `<tr>
      <td style="font-family:var(--mono);color:var(--text-dim)">${r.case_id}</td>
      <td style="color:var(--text);max-width:220px;overflow:hidden;text-overflow:ellipsis">${r.case_name}</td>
      <td>${catBadge}</td>
      <td>${diffBadge}</td>
      <td style="color:var(--text-dim)">${r.outcome || '—'}</td>
      <td>${fmtTok(tokens)} ${saved}</td>
      <td>${(r.latency_s*1000).toFixed(0)}ms</td>
      <td>${passBadge}</td>
    </tr>`;
  }).join('');
}

function populateFilters(results) {
  const cats  = [...new Set(results.map(r => r.category))];
  const diffs = [...new Set(results.map(r => r.difficulty))];
  const selCat  = document.getElementById('filter-category');
  const selDiff = document.getElementById('filter-difficulty');
  selCat.innerHTML  = '<option value="">All Categories</option>' +
    cats.map(c => `<option value="${c}">${c}</option>`).join('');
  selDiff.innerHTML = '<option value="">All Difficulties</option>' +
    diffs.map(d => `<option value="${d}">${d}</option>`).join('');
}

function applyFilters() {
  const cat     = document.getElementById('filter-category').value;
  const diff    = document.getElementById('filter-difficulty').value;
  const correct = document.getElementById('filter-correct').value;
  let filtered  = [...allResults];
  if (cat)     filtered = filtered.filter(r => r.category   === cat);
  if (diff)    filtered = filtered.filter(r => r.difficulty  === diff);
  if (correct) filtered = filtered.filter(r => String(r.correct) === correct);
  renderTable(filtered);
}

// ── Main load function ────────────────────────────────────────
function loadData(data) {
  destroyCharts();
  allResults = data.results || [];
  updateKPIs(data);
  buildCategoryChart(data);
  buildDifficultyChart(data);
  buildConsensusChart(data);
  buildTokenChart(data);
  buildRiskChart(data);
  buildLatencyChart(data);
  populateFilters(allResults);
  renderTable(allResults);
}

// ── Event wiring ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Demo button
  document.getElementById('btn-demo').addEventListener('click', () => {
    loadData(makeDemoData());
  });

  // File load button
  document.getElementById('btn-load').addEventListener('click', () => {
    document.getElementById('file-input').click();
  });

  document.getElementById('file-input').addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = evt => {
      try {
        const data = JSON.parse(evt.target.result);
        loadData(data);
      } catch (err) {
        alert('Invalid JSON file: ' + err.message);
      }
    };
    reader.readAsText(file);
    // Reset so same file can be reloaded
    e.target.value = '';
  });

  // Filter dropdowns
  ['filter-category','filter-difficulty','filter-correct'].forEach(id => {
    document.getElementById(id).addEventListener('change', applyFilters);
  });

  // Auto-load demo on first visit
  loadData(makeDemoData());
});

