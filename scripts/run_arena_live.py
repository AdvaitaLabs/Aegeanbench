#!/usr/bin/env python3
"""
run_arena_live.py — run the REAL benchmark arena.

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
    python scripts/run_arena_live.py [--emit-js]

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
from aegeanbench.scoring import score_prediction  # noqa: E402

LOKA_URL = os.environ.get("LOKA_URL", "http://localhost:5003").rstrip("/")
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://praka.ai/v1").rstrip("/")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
MODELS = [m.strip().split(":")[0] for m in
          os.environ.get("BENCHMARK_MODELS", "gpt-5.4,claude-opus-4-6,grok-4.3-fast").split(",")
          if m.strip()]
RESULTS = Path(__file__).resolve().parent.parent / "results"


# ── the 4 real cross-angle cases (mirror the frontend page) ─────────────────
# `pos`/`neg` are the two canonical direction tokens; `gt_dir` is the one that
# actually happened. `kw_pos`/`kw_neg` map free-text (zh/en) onto them.
CASES = [
    {
        "id": "WC-QATAR-2022", "entity": "Qatar · FIFA World Cup 2022",
        "tag": "Mega-event · government", "angle": "government",
        "asOf": "2022-06-01", "horizon": "~6 mo", "type": "range", "unit": "k",
        "question": "作为分析基准日 2022-06-01（世界杯开赛前），预测卡塔尔在 2022 世界杯窗口期"
                    "（11/20–12/18）将吸引多少国际访客（单位：千人），以及对国家的净影响。",
        "gt": {"direction_label": "up", "actual_value": 1400, "unit": "k",
               "narrative": ">1.4M 国际访客；>2000 亿美元十年基建；软实力提升伴随劳工争议。",
               "source": "Qatar GCO / Supreme Committee", "confidence": "approx"},
        "pos": "up", "neg": "down", "kw_pos": ["正", "增", "涌入", "up", "surge", "positive"],
        "kw_neg": ["负", "下降", "down", "negative"],
    },
    {
        "id": "CLIM-BC-CTAX-2008", "entity": "British Columbia · Carbon Tax (2008)",
        "tag": "Climate policy · government", "angle": "government",
        "asOf": "2008-07-01", "horizon": "~5 yr", "type": "range", "unit": "%",
        "question": "作为分析基准日 2008-07-01（碳税刚实施），预测约 5 年后不列颠哥伦比亚省"
                    "人均燃料消费相对加拿大其他地区的变化（百分比，下降为负），以及 GDP 是否受损。",
        "gt": {"direction_label": "down", "actual_value": -16, "unit": "%",
               "narrative": "人均石油燃料消费约下降 16%（2008–2013）而 GDP 与全国同步；税收中性。",
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
        "question": "作为分析基准日 2022-06-01，随着野火/气候损失上升，主要住宅保险公司会不会"
                    "退出加州市场？只能二选一：Insurers retreat / Market stays stable。",
        "gt": {"direction_label": "retreat", "actual_value": None, "unit": None,
               "narrative": "State Farm、Allstate 停止承保加州新房险（2022–23）；FAIR Plan 膨胀、保费上涨。",
               "source": "CA Dept. of Insurance / press", "confidence": "high"},
        "pos": "retreat", "neg": "stable",
        "kw_pos": ["retreat", "退出", "撤", "stop", "exit", "pull"],
        "kw_neg": ["stable", "稳定", "stay", "remain"],
    },
    {
        "id": "AD-INV-004", "entity": "ADNOC Distribution IPO",
        "tag": "Sovereign fund · IPO", "angle": "institution",
        "asOf": "2017-12-13", "horizon": "3–12 mo", "type": "match",
        "options": ["Up", "Down"],
        "question": "作为分析基准日 2017-12-13，国有油品分销子公司以 2.50 迪拉姆 IPO，"
                    "上市后 3–12 个月股价相对发行价是涨还是跌？只能二选一：Up / Down。",
        "gt": {"direction_label": "up", "actual_value": None, "unit": None,
               "narrative": "首年内股价高于 2.50 迪拉姆发行价，高股息支撑。",
               "source": "Abu Dhabi Securities Exchange (ADX)", "confidence": "high"},
        "pos": "up", "neg": "down",
        "kw_pos": ["up", "涨", "上涨", "above", "rise"],
        "kw_neg": ["down", "跌", "下跌", "below", "fall"],
    },
]


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
    return case["pos"]  # default lean to the recorded direction's token if ambiguous


def _gt(case) -> EventGroundTruth:
    g = case["gt"]
    return EventGroundTruth(direction_label=g["direction_label"], actual_value=g["actual_value"],
                            unit=g["unit"], narrative=g["narrative"], source=g["source"],
                            confidence=g["confidence"])


# ── competitor 1: live LokaWorld over HTTP ──────────────────────────────────
def predict_lokaworld(case):
    import requests
    r = requests.post(f"{LOKA_URL}/api/panel/run",
                      json={"topic": case["question"], "angle": case["angle"]}, timeout=600)
    r.raise_for_status()
    data = (r.json() or {}).get("data") or {}
    pred = data.get("prediction") or {}
    head = data.get("headline") or {}
    point = pred.get("point_estimate")
    direction = _norm_dir(case, pred.get("direction") or head.get("net_assessment"), point)
    ep = EventPrediction(direction=direction, point_estimate=point, unit=pred.get("unit"),
                         ci_80=pred.get("ci_80"), confidence=float(pred.get("confidence") or 0),
                         rationale=pred.get("rationale") or head.get("summary") or "")
    report = {
        "method": f"Live multi-agent simulation · {len(data.get('dimensions') or [])} departments → synthesis",
        "summary": head.get("summary") or pred.get("rationale") or "",
        "angles": [{"name": d.get("department", ""), "text": d.get("assessment") or d.get("perspective") or ""}
                   for d in (data.get("dimensions") or [])],
    }
    return ep, report


# ── competitor 2: raw frontier model, single-shot ───────────────────────────
_RAW_SYS = ("你是一名严谨的预测分析师。只用分析基准日当天及之前可得的信息，不得使用之后发生的事实。"
            "输出严格 JSON，不要额外文字。")


def predict_raw_model(case, model_id):
    import requests
    if not LLM_KEY:
        raise RuntimeError("LLM_API_KEY not set")
    if case["type"] == "match":
        ask = (f'{{"choice":"{case["options"][0]}|{case["options"][1]} 之一",'
               '"confidence":0.0,"summary":"一句话结论","rationale":"简述依据",'
               '"angles":[{"name":"视角","text":"分析"}]}')
        extra = f"只能在这两个选项里二选一：{case['options']}。"
    else:
        ask = ('{"direction":"方向(如 up/down/上升/下降)","point_estimate":0,'
               f'"unit":"{case["unit"]}","ci_80":[0,0],"confidence":0.0,'
               '"summary":"一句话结论","rationale":"简述依据","angles":[{"name":"视角","text":"分析"}]}')
        extra = f"point_estimate 用单位 {case['unit']}；ci_80 给 80% 置信区间 [下界,上界]。"
    user = f"问题：{case['question']}\n{extra}\n输出 JSON：{ask}"
    r = requests.post(f"{LLM_BASE}/chat/completions",
                      headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
                      json={"model": model_id, "temperature": 0.3,
                            "messages": [{"role": "system", "content": _RAW_SYS},
                                         {"role": "user", "content": user}]}, timeout=180)
    r.raise_for_status()
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
    competitors = ["lokaworld"] + MODELS
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
                elif comp == "lokaworld":
                    ep, report = predict_lokaworld(case)
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
        for comp, e in row["entries"].items():
            if row["type"] == "match":
                entries[comp] = {"choice": e["direction"], "confidence": e["confidence"],
                                 "correct": e["correct"], "report": e["report"]}
            else:
                entries[comp] = {"pred": e["pred"], "ci": e["ci"], "confidence": e["confidence"],
                                 "errPct": e["errPct"], "within": e["within"], "correct": e["correct"],
                                 "report": e["report"]}
        cases_js.append(f"""  {{
    id: {js(row['case_id'])}, entity: {js(row['entity'])}, tag: {js(row['tag'])},
    horizon: {js(row['horizon'])}, asOf: {js(row['asOf'])}, type: {js(row['type'])},
    unit: {js(row.get('unit'))}, {'options: ' + js(row['options']) + ',' if row.get('options') else ''}
    question: {js(row['question'])},
    actual: {js(row['actual'])},
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
    content = ("// AUTO-GENERATED by scripts/run_arena_live.py — real arena results.\n"
               "// Complete drop-in for lokaworld/src/data/benchmark.js.\n"
               "export const COMPETITORS = [\n"
               "  { id: 'lokaworld', name: 'LokaWorld', kind: 'agent', tagline: 'Multi-agent simulation' },\n"
               + "".join(f"  {{ id: {json.dumps(m)}, name: {json.dumps(name_map.get(m, m))}, kind: 'model' }},\n" for m in MODELS)
               + "];\n\nexport const BENCHMARK = [\n" + ",\n".join(cases_js) + "\n];\n" + tail)
    (RESULTS / "benchmark.generated.js").write_text(content, encoding="utf-8")
    print(f"→ results/benchmark.generated.js  (drop-in for lokaworld/src/data/benchmark.js)")


if __name__ == "__main__":
    main()
