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

