#!/usr/bin/env python3
"""
run_benchmark_arena.py — run the REAL benchmark arena.

LokaWorld (live multi-agent simulation, over HTTP) vs raw frontier models
(gpt-5.4 / claude-opus-4-6 / grok-4.3-fast, single-shot via the OpenAI-
compatible endpoint), each producing a prediction + report for real historical
cases, scored against the RECORDED real-world outcome. This is what turns the
frontend leaderboard from MOCK_DEMO into real numbers.

Nothing is mocked. A competitor whose call fails (no key / Loka down / bad
JSON) is scored as a MISS for that case with an explicit error note — the run
still completes and the leaderboard is honest about it.

────────────────────────────────────────────────────────────────────────────
USAGE
    # 1) Loka backend running with the report path (default port 5003):
    #    docker compose up -d   (in lokaworld)
    # 2) environment:
    export LLM_API_KEY=sk-...                       # praka key (raw models)
    export LLM_BASE_URL=https://praka.ai/v1         # OpenAI-compatible base
    export LOKA_URL=http://localhost:5003           # live Loka
    export BENCHMARK_MODELS=gpt-5.4,claude-opus-4-6,grok-4.3-fast
    python scripts/run_benchmark_arena.py [--emit-js]

COST: real LLM calls — 4 cases × (1 Loka sim + N raw models). With the default
3 models that's 4 Loka runs + 12 single-shot calls. Run it when you mean to.

OUTPUT (under results/):
    arena_live.json          full per-case, per-competitor predictions + reports + verdicts + leaderboard
    benchmark.generated.js   (with --emit-js) drop-in replacement for lokaworld/src/data/benchmark.js
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aegeanbench.core.models import EventGroundTruth, EventPrediction  # noqa: E402
from aegeanbench.scoring import (calibrate_confidence, extract_prediction,  # noqa: E402
                                 fit_confidence_table, score_prediction)

LOKA_URL = os.environ.get("LOKA_URL", "http://localhost:5003").rstrip("/")
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://praka.ai/v1").rstrip("/")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
MODELS = [m.strip().split(":")[0] for m in
          os.environ.get("BENCHMARK_MODELS", "gpt-5.4,claude-opus-4-6,grok-4.3-fast").split(",")
          if m.strip()]
RESULTS = Path(__file__).resolve().parent.parent / "results"
# Competitor id for the Loka entry — override for A/B runs so the history and
# leaderboard keep the variants apart (e.g. lokaworld-deep, lokaworld-deep-struct).
LOKA_ID = os.environ.get("BENCHMARK_LOKA_ID", "lokaworld")


# ── the 4 real cross-angle cases (mirror the frontend page) ─────────────────
# `pos`/`neg` are the two canonical direction tokens; `gt_dir` is the one that
# actually happened. `kw_pos`/`kw_neg` map free-text (zh/en) onto them.
CASES = [
    {
        "id": "WC-QATAR-2022", "entity": "Qatar · FIFA World Cup 2022",
        "tag": "Mega-event · government", "angle": "government",
        "asOf": "2022-06-01", "horizon": "~6 mo", "type": "range", "unit": "k",
        "question": "As of the analysis date 2022-06-01 (before kickoff), forecast how many "
                    "international visitors Qatar will draw during the 2022 World Cup window "
                    "(Nov 20 – Dec 18), in thousands, and the net national impact.",
        "gt": {"direction_label": "up", "actual_value": 1400, "unit": "k",
               "headline": "≈ 1.4M international visitors",
               "narrative": ">1.4M international visitors (Nov 20 – Dec 18); >US$200B on a decade of "
                            "infrastructure; a soft-power lift alongside migrant-labour scrutiny.",
               "source": "Qatar GCO / Supreme Committee", "confidence": "approx"},
        "pos": "up", "neg": "down", "kw_pos": ["正", "增", "涌入", "up", "surge", "positive"],
        "kw_neg": ["负", "下降", "down", "negative"],
    },
    {
        "id": "CLIM-BC-CTAX-2008", "entity": "British Columbia · Carbon Tax (2008)",
        "tag": "Climate policy · government", "angle": "government",
        "asOf": "2008-07-01", "horizon": "~5 yr", "type": "range", "unit": "%",
        "question": "As of 2008-07-01 (carbon tax just enacted), forecast the change ~5 years later "
                    "in British Columbia's per-capita fuel use relative to the rest of Canada "
                    "(percent; a fall is negative), and whether GDP is hurt.",
        "gt": {"direction_label": "down", "actual_value": -16, "unit": "%",
               "headline": "≈ −16% per-capita fuel use (GDP unhurt)",
               "narrative": "Per-capita petroleum fuel use fell ~16% (2008–2013) vs the rest of Canada "
                            "while GDP kept pace; revenue-neutral.",
               "source": "Elgie & McClay (2013)", "confidence": "approx"},
        "pos": "down", "neg": "up",  # 'good' outcome = fuel use down; sign of number decides
        "kw_pos": ["下降", "减少", "负", "decline", "fall", "down"],
        "kw_neg": ["上升", "增加", "up", "rise"],
        "numeric_sign": True,   # derive direction from the sign of point_estimate
    },
    {
        "id": "INS-CA-CLIMATE-2023", "entity": "California · Home-Insurance Market",
        "tag": "Insurance · market reaction", "angle": "company",
        "asOf": "2022-06-01", "horizon": "~24 mo", "type": "match",
        "options": ["Insurers retreat", "Market stays stable"],
        "question": "As of 2022-06-01, as wildfire and climate losses mount, will major home insurers "
                    "retreat from California? Choose one: Insurers retreat / Market stays stable.",
        "gt": {"direction_label": "retreat", "actual_value": None, "unit": None,
               "headline": "Insurers retreated",
               "narrative": "State Farm and Allstate stopped writing new CA home policies (2022–23); "
                            "the FAIR Plan swelled and premiums jumped.",
               "source": "CA Dept. of Insurance / press", "confidence": "high"},
        "pos": "retreat", "neg": "stable",
        "kw_pos": ["retreat", "退出", "撤", "stop", "exit", "pull"],
        "kw_neg": ["stable", "稳定", "stay", "remain"],
    },
    {
        "id": "AD-INV-004", "entity": "ADNOC Distribution IPO",
        # market angle: the outcome is decided by investor behaviour (IC / buy-side
        # research / macro / risk / trading / retail), not by an NGO-style org chart.
        "tag": "Sovereign fund · IPO", "angle": "market",
        "asOf": "2017-12-13", "horizon": "3–12 mo", "type": "match",
        "options": ["Up", "Down"],
        "question": "As of 2017-12-13, the state fuel-distribution subsidiary IPOs at AED 2.50 — up or "
                    "down over the first 3–12 months versus the offer price? Choose one: Up / Down.",
        "gt": {"direction_label": "up", "actual_value": None, "unit": None,
               "headline": "Up · above offer",
               "narrative": "Traded above the AED 2.50 offer price within the first year, supported by a "
                            "high, stable dividend.",
               "source": "Abu Dhabi Securities Exchange (ADX)", "confidence": "high"},
        "pos": "up", "neg": "down",
        "kw_pos": ["up", "涨", "上涨", "above", "rise"],
        "kw_neg": ["down", "跌", "下跌", "below", "fall"],
    },
]


def _post_retry(url, *, attempts=4, **kw):
    """POST with retry/backoff on transient 5xx / timeout (praka returns 503/524)."""
    import time
    import requests
    last = None
    for i in range(attempts):
        try:
            r = requests.post(url, **kw)
            if r.status_code in (429, 500, 502, 503, 504, 524):
                last = requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
                time.sleep(2 * (i + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last if last else RuntimeError("request failed")


def _norm_dir(case, text, point):
    """Map a competitor's free-text direction (+ numeric) onto the case's canonical token."""
    t = str(text or "").lower()
    if case.get("numeric_sign") and point is not None:
        return case["pos"] if point < 0 else case["neg"]  # BC: negative % = fuel down = 'down'(pos side)
    for kw in case["kw_pos"]:
        if kw.lower() in t:
            return case["pos"]
    for kw in case["kw_neg"]:
        if kw.lower() in t:
            return case["neg"]
    # Unparseable direction = no prediction. Do NOT default to case["pos"]:
    # pos happens to equal the recorded outcome for every case, so defaulting
    # there silently awarded ambiguous answers a correct verdict.
    return None


def _gt(case) -> EventGroundTruth:
    g = case["gt"]
    return EventGroundTruth(direction_label=g["direction_label"], actual_value=g["actual_value"],
                            unit=g["unit"], narrative=g["narrative"], source=g["source"],
                            confidence=g["confidence"])


# ── competitor 1: live LokaWorld over HTTP ──────────────────────────────────
def predict_lokaworld(case):
    r = _post_retry(f"{LOKA_URL}/api/panel/run",
                    json={"topic": case["question"], "angle": case["angle"], "lang": "en",
                          # shared fact base + one cross-department hearing round —
                          # the full simulation, not just the parallel ensemble.
                          "facts": True, "hearing": True}, timeout=600)
    data = (r.json() or {}).get("data") or {}
    pred = data.get("prediction") or {}
    head = data.get("headline") or {}
    point = pred.get("point_estimate")
    direction = _norm_dir(case, pred.get("direction") or head.get("net_assessment"), point)
    ep = EventPrediction(direction=direction, point_estimate=point, unit=pred.get("unit"),
                         ci_80=pred.get("ci_80"), confidence=float(pred.get("confidence") or 0),
                         rationale=pred.get("rationale") or head.get("summary") or "")
    dims = data.get("dimensions") or []
    report = {
        "method": f"Live multi-agent simulation · {len(dims)} departments → synthesis",
        # executive_summary is the professional multi-paragraph synthesis; the
        # one-line headline stays available separately for compact UI chips.
        "summary": data.get("executive_summary") or head.get("summary") or pred.get("rationale") or "",
        "headline": head.get("summary") or "",
        "net": head.get("net_assessment"),
        "keyFindings": data.get("key_findings") or [],
        "crossCutting": data.get("cross_cutting") or "",
        "rationale": pred.get("rationale") or "",
        # the shared fact sheet every department received — kept for the
        # process view AND for post-mortems: when a run misses, the first
        # suspect is a bad anchor in here.
        "factSheet": data.get("fact_sheet") or "",
        # full per-department structure — the frontend renders these both as the
        # report's angle sections and as the "simulation process" replay.
        "angles": [{
            "name": d.get("department", ""),
            "mandate": d.get("mandate") or "",
            "perspective": d.get("perspective") or "",
            "text": d.get("assessment") or d.get("perspective") or "",
            "impacts": d.get("impacts") or [],
            "risks": d.get("risks") or [],
            "confidence": d.get("confidence"),
            # cross-department hearing outcome (when the run had conflicts)
            "revised": bool(d.get("revised")),
            "revisionNote": d.get("revision_note") or "",
        } for d in dims],
    }
    return ep, report


def predict_lokaworld_deep(case, confirm_structure=False):
    """Score the REAL product: the full OASIS deep pipeline via the workflow
    API. The Loka server MUST run with LOKA_EMIT_PREDICTION=true (adds the
    machine-readable block to the report) and SHOULD run with
    LOKA_BENCHMARK_NO_LEAK=true (blocks live web search) for honest backtests.
    Slow: 10-30 min per case. `confirm_structure=True` first fetches the
    step-0 classification + proposed hierarchy/axes and passes them into the
    run — the A/B lever for "does structure confirmation improve accuracy?".
    Tunables: BENCHMARK_DEEP_AGENTS (150), BENCHMARK_DEEP_ROUNDS (10),
    BENCHMARK_DEEP_TIMEOUT (3600s)."""
    import time as _t
    import requests
    topic_analysis = None
    if confirm_structure:
        d = ((_post_retry(f"{LOKA_URL}/api/workflow/analyze-topic",
                          json={"question": case["question"]}, timeout=300)
              .json() or {}).get("data")) or {}
        if d.get("structure") or d.get("axes"):
            topic_analysis = {"angle": d.get("angle"), "suggested_angle": d.get("angle"),
                              "summary": d.get("summary"), "marginals": d.get("marginals"),
                              "structure": d.get("structure"), "axes": d.get("axes")}
    body = {"question": case["question"],
            "agent_count": int(os.environ.get("BENCHMARK_DEEP_AGENTS", "150")),
            "max_rounds": int(os.environ.get("BENCHMARK_DEEP_ROUNDS", "10"))}
    if topic_analysis:
        body["topic_analysis"] = topic_analysis
    dag = ((_post_retry(f"{LOKA_URL}/api/workflow/plan", json=body, timeout=600)
            .json() or {}).get("data")) or {}
    run = ((_post_retry(f"{LOKA_URL}/api/workflow/run",
                        json={"workflow_id": dag.get("workflow_id"), "dag": dag},
                        timeout=60).json() or {}).get("data")) or {}
    run_id = run.get("run_id")
    if not run_id:
        raise RuntimeError(f"deep run did not start: {run}")
    deadline = _t.time() + float(os.environ.get("BENCHMARK_DEEP_TIMEOUT", "3600"))
    print(f"    [deep] run {run_id} started — a full OASIS simulation takes "
          f"10-30 min; progress prints below (Ctrl-C only kills THIS script, "
          f"the Loka run keeps going server-side)", flush=True)
    project_id = None
    _last = ""
    while _t.time() < deadline:
        st = ((requests.get(f"{LOKA_URL}/api/workflow/run/{run_id}/status", timeout=30)
               .json() or {}).get("data")) or {}
        status = st.get("status")
        line = (f"[deep] {status} {st.get('progress', 0)}% · "
                f"{st.get('current_node_id') or ''} — "
                f"{(st.get('message') or '')[:90]}")
        if line != _last:
            print(f"    {line}", flush=True)
            _last = line
        if status == "completed":
            project_id = st.get("project_id")
            break
        if status in ("failed", "cancelled", "interrupted"):
            raise RuntimeError(f"deep run {run_id} {status}: {st.get('error')}")
        _t.sleep(15)
    if not project_id:
        raise TimeoutError(f"deep run {run_id} timed out")
    resp = requests.get(f"{LOKA_URL}/api/project/{project_id}/report_md", timeout=60)
    md = resp.text
    try:
        j = resp.json()
        md = ((j.get("data") or {}).get("markdown")
              or (j.get("data") or {}).get("report_md")
              or j.get("report_md") or md)
    except ValueError:
        pass
    ep0 = extract_prediction(md)
    if ep0 is None:
        raise RuntimeError("no AEGEANBENCH prediction block in the deep report — "
                           "start Loka with LOKA_EMIT_PREDICTION=true")
    direction = _norm_dir(case, ep0.direction, ep0.point_estimate)
    ep = EventPrediction(direction=direction, point_estimate=ep0.point_estimate,
                         unit=ep0.unit, ci_80=ep0.ci_80,
                         confidence=ep0.confidence, rationale=ep0.rationale)
    report = {"method": (f"Deep OASIS simulation ({body['agent_count']} agents x "
                         f"{body['max_rounds']} rounds) -> consulting report"
                         + (" · confirmed structure" if topic_analysis else "")),
              "summary": (ep0.rationale or md[:600]),
              "angles": []}
    return ep, report


# ── competitor 2: raw frontier model, single-shot ───────────────────────────
_RAW_SYS = ("You are a rigorous forecasting analyst. Use ONLY information available on or before the "
            "analysis date; do not use facts that happened afterward. Respond with strict JSON only, "
            "every value in English.")


def predict_raw_model(case, model_id):
    if not LLM_KEY:
        raise RuntimeError("LLM_API_KEY not set")
    # Ask for report depth comparable to LokaWorld's synthesized panel report,
    # so the arena grades forecasting skill — not output-length differences.
    report_ask = ('"summary":"120-180 word executive summary: verdict first, then the 2-3 decisive '
                  'drivers with numbers, the main risk to the call, and what would change your mind",'
                  '"key_findings":["3-5 crisp quantified findings"],'
                  '"rationale":"the drivers behind your number/choice",'
                  '"angles":[{"name":"analytical lens","text":"60-120 word analysis with concrete '
                  'figures where possible"}] (3-5 angles),'
                  '"risks":["main downside/upside risks to this forecast"]')
    if case["type"] == "match":
        ask = '{"choice":"one of the two options","confidence":0.0,' + report_ask + '}'
        extra = f"Pick exactly one of these two options: {case['options']}."
    else:
        ask = ('{"direction":"up/down","point_estimate":0,'
               f'"unit":"{case["unit"]}","ci_80":[0,0],"confidence":0.0,' + report_ask + '}')
        extra = f"point_estimate in unit {case['unit']}; ci_80 is the 80% confidence interval [low, high]."
    user = f"Question: {case['question']}\n{extra}\nOutput JSON: {ask}"
    r = _post_retry(f"{LLM_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
                    json={"model": model_id, "temperature": 0.3,
                          "messages": [{"role": "system", "content": _RAW_SYS},
                                       {"role": "user", "content": user}]}, timeout=180)
    content = (((r.json() or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    m = re.search(r"\{.*\}", content, re.DOTALL)
    obj = json.loads(m.group(0)) if m else {}
    if case["type"] == "match":
        direction = _norm_dir(case, obj.get("choice"), None)
        point, ci, unit = None, None, None
    else:
        point = obj.get("point_estimate")
        point = float(point) if isinstance(point, (int, float)) else None
        ci = obj.get("ci_80") if isinstance(obj.get("ci_80"), list) else None
        unit = obj.get("unit") or case["unit"]
        direction = _norm_dir(case, obj.get("direction"), point)
    ep = EventPrediction(direction=direction, point_estimate=point, unit=unit, ci_80=ci,
                         confidence=float(obj.get("confidence") or 0), rationale=obj.get("rationale") or "")
    report = {"method": "Single-shot forecast",
              "summary": obj.get("summary") or "",
              "keyFindings": obj.get("key_findings") if isinstance(obj.get("key_findings"), list) else [],
              "rationale": obj.get("rationale") or "",
              "risks": obj.get("risks") if isinstance(obj.get("risks"), list) else [],
              "angles": obj.get("angles") if isinstance(obj.get("angles"), list) else []}
    return ep, report


def _median(xs):
    s = sorted(x for x in xs if x is not None)
    if not s:
        return None
    n = len(s)
    return s[n // 2] if n % 2 else round((s[n // 2 - 1] + s[n // 2]) / 2, 2)


def _dry_predict(case, comp):
    """Deterministic stub (no LLM) — used by --dry to verify the wiring/plumbing."""
    good = comp == "lokaworld"
    worst = bool(MODELS) and comp == MODELS[-1]        # last model plays the loser
    if case["type"] == "match":
        direction = case["neg"] if worst else case["pos"]
        ep = EventPrediction(direction=direction, confidence=0.8 if good else 0.6)
    else:
        actual = case["gt"]["actual_value"] or 0
        point = actual * (1.01 if good else 1.2)
        span = abs(actual) * 0.15
        ep = EventPrediction(direction=case["pos"], point_estimate=round(point, 2),
                             unit=case["unit"], ci_80=[round(point - span, 2), round(point + span, 2)],
                             confidence=0.82 if good else 0.7)
    report = {"method": "DRY", "summary": f"{comp} dry summary", "angles": [{"name": "A", "text": "x"}]}
    return ep, report


def main():
    emit_js = "--emit-js" in sys.argv
    dry = "--dry" in sys.argv
    # --deep: score the REAL deep pipeline instead of the panel harness.
    # --confirm-structure: (deep only) pass the step-0 hierarchy/axes into the
    # run — run the suite once with and once without (different
    # BENCHMARK_LOKA_ID) for the structure A/B.
    deep = "--deep" in sys.argv
    confirm_structure = "--confirm-structure" in sys.argv
    competitors = [LOKA_ID] + MODELS
    RESULTS.mkdir(exist_ok=True)
    per_case, scored = [], {c: [] for c in competitors}

    for case in CASES:
        gt = _gt(case)
        print(f"\n=== {case['id']} — {case['entity']} ===")
        row = {"case_id": case["id"], "entity": case["entity"], "tag": case["tag"],
               "asOf": case["asOf"], "horizon": case["horizon"], "type": case["type"],
               "unit": case.get("unit"), "options": case.get("options"),
               "question": case["question"], "actual": case["gt"], "entries": {}}
        for comp in competitors:
            try:
                if dry:
                    ep, report = _dry_predict(case, comp)
                elif comp == LOKA_ID:
                    ep, report = (predict_lokaworld_deep(case, confirm_structure)
                                  if deep else predict_lokaworld(case))
                else:
                    ep, report = predict_raw_model(case, comp)
                err = None
            except Exception as e:  # noqa: BLE001
                ep, report, err = None, {"method": "", "summary": f"(failed: {e})", "angles": []}, str(e)
            metrics = score_prediction(ep, gt)
            scored[comp].append(metrics)
            row["entries"][comp] = {
                "direction": metrics.predicted_direction,
                "pred": ep.point_estimate if ep else None,
                "ci": ep.ci_80 if ep else None,
                "confidence": ep.confidence if ep else None,
                "correct": metrics.direction_correct if case["type"] == "match" else
                           (metrics.value_error_pct is not None and metrics.value_error_pct <= 15),
                "errPct": None if metrics.value_error_pct is None else round(metrics.value_error_pct),
                "within": metrics.within_ci,
                "report": report, "error": err,
            }
            v = row["entries"][comp]
            print(f"  {comp:16} dir={metrics.predicted_direction} "
                  f"val={ep.point_estimate if ep else '-'} err={v['errPct']}% "
                  f"within={v['within']} correct={v['correct']}" + (f"  ERR:{err}" if err else ""))
        per_case.append(row)

    # ── confidence calibration ─────────────────────────────────────────────
    # Every real run appends (confidence, correct) pairs to a history file;
    # once a competitor has enough samples, its self-reported confidence is
    # annotated with the observed hit rate of that confidence bin (calConf).
    # Dry runs are excluded so stub data never pollutes the history.
    history_path = RESULTS / "arena_history.jsonl"
    if not dry:
        import datetime
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        with history_path.open("a", encoding="utf-8") as f:
            for row in per_case:
                for comp, e in row["entries"].items():
                    if e.get("confidence") is None:
                        continue
                    f.write(json.dumps({"ts": ts, "case_id": row["case_id"],
                                        "competitor": comp,
                                        "confidence": e["confidence"],
                                        "correct": bool(e["correct"])},
                                       ensure_ascii=False) + "\n")
    cal_history = []
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            try:
                cal_history.append(json.loads(line))
            except ValueError:
                continue
    cal_tables = fit_confidence_table(cal_history)
    for row in per_case:
        for comp, e in row["entries"].items():
            e["calConf"] = calibrate_confidence(cal_tables, comp, e.get("confidence"))
    if cal_tables:
        print("\n── calibration (observed hit rate per confidence bin) ──")
        for comp, bins in cal_tables.items():
            cells = "  ".join(f"[{b['lo']:.2f},{b['hi']:.2f}) n={b['n']} obs={b['observed']}"
                              for b in bins if b["n"])
            print(f"  {comp:16} {cells}")
    else:
        need = 8
        print(f"\n(confidence calibration pending — fewer than {need} samples per "
              f"competitor in {history_path.name}; keep running real cases)")

    leaderboard = []
    for comp in competitors:
        ms = scored[comp]
        n = len(ms) or 1
        correct = sum(1 for x in ms if (x.direction_correct if x.actual_value is None
                                        else (x.value_error_pct is not None and x.value_error_pct <= 15)))
        errs = [x.value_error_pct for x in ms if x.value_error_pct is not None]
        cis = [x for x in ms if x.within_ci is not None]
        ci_hit = sum(1 for x in cis if x.within_ci)
        leaderboard.append({"competitor": comp, "n": len(ms), "correct": correct,
                            "accuracy": round(correct / n * 100),
                            "median_error_pct": _median(errs),
                            "ci_hit": ci_hit, "ci_total": len(cis),
                            "ci_pct": round(ci_hit / len(cis) * 100) if cis else None})
    leaderboard.sort(key=lambda r: (-r["accuracy"],
                                    r["median_error_pct"] if r["median_error_pct"] is not None else 1e9))

    out = {"leaderboard": leaderboard, "per_case": per_case,
           "config": {"loka_url": LOKA_URL, "models": MODELS}}
    (RESULTS / "arena_live.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ LEADERBOARD ================")
    for i, r in enumerate(leaderboard, 1):
        me = "-" if r["median_error_pct"] is None else f"{r['median_error_pct']}%"
        ci = "-" if r["ci_pct"] is None else f"{r['ci_pct']}%"
        print(f" {i}. {r['competitor']:16} acc={r['accuracy']}% ({r['correct']}/{r['n']})  medErr={me}  CI={ci}")
    print(f"\n→ results/arena_live.json")

    if emit_js:
        _emit_frontend_js(per_case)


def _emit_frontend_js(per_case):
    """Write a drop-in for lokaworld/src/data/benchmark.js (real numbers + reports)."""
    def js(o):
        return json.dumps(o, ensure_ascii=False)
    cases_js = []
    for row in per_case:
        entries = {}
        opts = row.get("options") or []

        def _opt_label(tok):
            if not tok:
                return None
            return next((o for o in opts if str(tok).lower() in str(o).lower()), tok)

        for comp, e in row["entries"].items():
            if row["type"] == "match":
                entries[comp] = {"choice": _opt_label(e["direction"]), "confidence": e["confidence"],
                                 "calConf": e.get("calConf"),
                                 "correct": e["correct"], "report": e["report"]}
            else:
                entries[comp] = {"pred": e["pred"], "ci": e["ci"], "confidence": e["confidence"],
                                 "calConf": e.get("calConf"),
                                 "errPct": e["errPct"], "within": e["within"], "correct": e["correct"],
                                 "report": e["report"]}
        g = row["actual"]   # raw ground-truth dict
        actual = {"value": g.get("actual_value"), "headline": g.get("headline") or g.get("direction_label", ""),
                  "detail": g.get("narrative", ""), "source": g.get("source", "")}
        if row["type"] == "match":
            actual["choice"] = _opt_label(g.get("direction_label"))
        cases_js.append(f"""  {{
    id: {js(row['case_id'])}, entity: {js(row['entity'])}, tag: {js(row['tag'])},
    horizon: {js(row['horizon'])}, asOf: {js(row['asOf'])}, type: {js(row['type'])},
    unit: {js(row.get('unit'))}, {'options: ' + js(row['options']) + ',' if row.get('options') else ''}
    question: {js(row['question'])},
    actual: {js(actual)},
    entries: {js(entries)},
  }}""")
    # computed exports the frontend needs (landing.js reads BENCHMARK_SUMMARY,
    # benchmark.js screen reads LEADERBOARD) — mirror src/data/benchmark.js so
    # this file is a COMPLETE drop-in.
    tail = """
export const LEADERBOARD = COMPETITORS.map(m => {
  const rows = BENCHMARK.map(c => c.entries[m.id]).filter(Boolean);
  const n = rows.length || 1;
  const correct = rows.filter(r => r.correct).length;
  const errs = rows.map(r => r.errPct).filter(e => e !== null && e !== undefined).sort((a, b) => a - b);
  const medianErr = errs.length ? (errs.length % 2 ? errs[(errs.length - 1) / 2]
    : Math.round((errs[errs.length / 2 - 1] + errs[errs.length / 2]) / 2)) : null;
  const ciRows = rows.filter(r => r.within !== null && r.within !== undefined);
  const ciHit = ciRows.filter(r => r.within).length;
  return { id: m.id, name: m.name, kind: m.kind, tagline: m.tagline,
    n, correct, accuracy: Math.round((correct / n) * 100), medianErr,
    ciHit, ciTotal: ciRows.length, ciPct: ciRows.length ? Math.round((ciHit / ciRows.length) * 100) : null };
}).sort((a, b) => b.accuracy - a.accuracy || (a.medianErr ?? 999) - (b.medianErr ?? 999));

export const BENCHMARK_SUMMARY = (() => {
  const top = LEADERBOARD[0] || { accuracy: 0, correct: 0, ciPct: 0, ciHit: 0, ciTotal: 0, medianErr: null, name: '—' };
  return { total: BENCHMARK.length, dirPct: top.accuracy, dirCorrect: top.correct,
    medianErr: top.medianErr, ciPct: top.ciPct || 0, ciHit: top.ciHit, ciTotal: top.ciTotal, topModel: top.name };
})();
"""
    name_map = {"gpt-5.4": "GPT-5.4", "claude-opus-4-6": "Claude Opus 4.6", "grok-4.3-fast": "Grok 4.3 Fast"}
    content = ("// AUTO-GENERATED by scripts/run_benchmark_arena.py — real arena results.\n"
               "// Complete drop-in for lokaworld/src/data/benchmark.js.\n"
               "export const COMPETITORS = [\n"
               f"  {{ id: {json.dumps(LOKA_ID)}, name: 'LokaWorld', kind: 'agent', tagline: 'Multi-agent simulation' }},\n"
               + "".join(f"  {{ id: {json.dumps(m)}, name: {json.dumps(name_map.get(m, m))}, kind: 'model' }},\n" for m in MODELS)
               + "];\n\nexport const BENCHMARK = [\n" + ",\n".join(cases_js) + "\n];\n" + tail)
    (RESULTS / "benchmark.generated.js").write_text(content, encoding="utf-8")
    print(f"→ results/benchmark.generated.js  (drop-in for lokaworld/src/data/benchmark.js)")


if __name__ == "__main__":
    main()
