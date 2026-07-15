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
from aegeanbench.scoring import (calibrate_confidence, convert_to_unit,  # noqa: E402
                                 extract_prediction, fit_confidence_table,
                                 score_prediction)


def _rescale_to_case_unit(ep, case):
    """Convert a prediction's point estimate + CI into the CASE's unit so the
    printed value, the stored `pred`, and scoring all agree. A report that
    answers "1.4 million" for a case measured in thousands (k) otherwise stores
    1_400_000 and scores as ~99900% error against actual=1400. No-op when the
    prediction is already in the case unit or the unit isn't a magnitude scale."""
    if ep is None:
        return ep
    to_u = case.get("unit")
    if to_u is None:
        return ep
    from_u = ep.unit
    ep.point_estimate = convert_to_unit(ep.point_estimate, from_u, to_u)
    if ep.ci_80:
        ep.ci_80 = [convert_to_unit(ep.ci_80[0], from_u, to_u),
                    convert_to_unit(ep.ci_80[1], from_u, to_u)]
    ep.unit = to_u
    return ep

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
        "anon_question": "As of 6 months before kickoff: a small, wealthy Gulf monarchy (population ~2.9M, ~30k hotel rooms plus cruise-ship and fan-village capacity, ~$200B spent on a decade of infrastructure) hosts the world's biggest football tournament — 64 matches over 4 weeks in Nov-Dec, ~3M tickets issued, strong shuttle-flight links to neighboring states. Forecast international visitors during the tournament window, in thousands, and the net national impact.",
        "question": "As of the analysis date 2022-06-01 (before kickoff), forecast how many "
                    "international visitors Qatar will draw during the 2022 World Cup window "
                    "(Nov 20 – Dec 18, 2022), in thousands, and the net national impact.",
        "gt": {"direction_label": "up", "actual_value": 1400, "unit": "k",
               "headline": "≈ 1.4M international visitors",
               "narrative": ">1.4M international visitors (Nov 20 – Dec 18, 2022); >US$200B on a decade of "
                            "infrastructure; a soft-power lift alongside migrant-labour scrutiny.",
               "source": "Qatar GCO / Supreme Committee", "confidence": "approx"},
        "pos": "up", "neg": "down", "kw_pos": ["正", "增", "涌入", "up", "surge", "positive"],
        "kw_neg": ["负", "下降", "down", "negative"],
    },
    {
        "id": "CLIM-BC-CTAX-2008", "entity": "British Columbia · Carbon Tax (2008)",
        "tag": "Climate policy · government", "angle": "government",
        "asOf": "2008-07-01", "horizon": "~5 yr", "type": "range", "unit": "%",
        "anon_question": "A western province of a large developed federation enacts the continent's first revenue-neutral carbon tax in mid-2008: starting ~$10/tonne, rising $5/yr to $30/tonne, all revenue returned via income-tax cuts. The province is highly urbanized with decent transit alternatives. Forecast the change ~5 years later in the province's per-capita fuel use relative to the rest of the federation (percent; a fall is negative), and whether provincial GDP is hurt.",
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
        "anon_question": 'As of mid-2022: a large, wealthy coastal state (a top-5 economy if it were a country) faces years of record wildfire losses; its insurance regulator must approve every rate increase and forbids forward-looking catastrophe models in pricing, so premiums lag true risk. Reinsurance costs are surging. Within ~24 months, will major home insurers retreat from the state, or does the market stay stable? Choose one: Insurers retreat / Market stays stable.',
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
        "anon_question": "December 2017: a Gulf state's national oil company lists its fuel-distribution retail subsidiary — the dominant nationwide station network — at a fixed offer price. The IPO is ~22x oversubscribed, the company pledges a high stable dividend payout, oil prices are recovering, and quality listings are scarce on the local exchange. Up or down versus the offer price over the first 3-12 months? Choose one: Up / Down.",
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


# ── multi-dimensional outcome scorecards (per case) ─────────────────────────
# A real outcome is more than one number. These dimensions — direct, second-order,
# distributional — are what a whole-ecosystem simulation can cover but a single-shot
# model usually can't. Actuals + sources are web-verified; weight>1 marks the
# second-order/distributional dims (the ones you can't get right by recall alone).
DIMENSIONS = {
    "WC-QATAR-2022": [
        {"id": "D1_visitors", "dim": "direct", "weight": 1.0, "q": "赛事窗口国际访客数", "actual": {"value": "1.4M", "confidence": "high", "source": "Qatar Supreme Committee", "url": "https://thepeninsulaqatar.com/article/19/12/2022/qatar-hosts-more-than-14-million-visitors-during-fifa-world-cup"}},
        {"id": "D2_gdp_share", "dim": "direct", "weight": 1.0, "q": "占全年 GDP 的直接贡献", "actual": {"value": "0.7–1.0%", "confidence": "medium", "source": "IMF (2024)", "url": "https://www.imf.org/-/media/Files/Publications/Selected-Issues-Papers/2024/English/SIPEA2024011.ashx", "note": "全年占比，非单季 ppt"}},
        {"id": "D3_hotel_peak", "dim": "distributional", "weight": 1.5, "q": "酒店单日峰值入住率", "actual": {"value": "90.5%", "confidence": "medium", "source": "STR / CoStar", "url": "https://www.mylighthouse.com/resources/blog/hotel-pricing-qatar-world-cup", "note": "月度综合仅~56%"}},
        {"id": "D4_shuttle_share", "dim": "distributional", "weight": 1.5, "q": "住邻国、乘穿梭航班往返的访客占比", "actual": {"value": "~30%", "confidence": "medium", "source": "IMF (2024)", "url": "https://www.imf.org/-/media/Files/Publications/Selected-Issues-Papers/2024/English/SIPEA2024011.ashx"}},
        {"id": "D5_air_surge", "dim": "second_order", "weight": 1.5, "q": "Hamad 机场 11–12 月客流同比增幅", "actual": {"value": "+61.7%", "confidence": "medium", "source": "Hamad Int'l", "url": "https://www.arabianbusiness.com/industries/travel-hospitality/passenger-numbers-go-through-the-roof-during-fifa-world-cup-in-qatar"}},
        {"id": "D6_legacy_tourism", "dim": "second_order", "weight": 1.5, "q": "赛后(2023)入境游客 vs 2019", "actual": {"value": "+89% ↑", "confidence": "high", "source": "Qatar Tourism / IMF", "url": "https://www.qatartourism.com/en/news-and-media/press-releases/qatar-welcomes-4-million-visitors"}},
    ],
    "CLIM-BC-CTAX-2008": [
        {"id": "D1_fuel_demand", "dim": "direct", "weight": 1.0, "q": "5年后人均受税燃料消费(相对全国)变化", "actual": {"value": "-17~-19%", "confidence": "high", "source": "Elgie & McClay, Canadian Public Policy (2013)", "url": "https://utppublishing.com/doi/10.3138/CPP.39.Supplement2.S1"}},
        {"id": "D2_emissions", "dim": "direct", "weight": 1.0, "q": "人均温室气体排放变化", "actual": {"value": "约 -9% (整体 -5~-15%)", "confidence": "medium", "source": "Murray & Rivers, Energy Policy 86 (2015)", "url": "https://www.sciencedirect.com/science/article/abs/pii/S0301421515300550"}},
        {"id": "D3_gdp_effect", "dim": "second_order", "weight": 1.5, "q": "BC 经济增长 vs 全国其余地区", "actual": {"value": "持平/略优 ↑ (无增长损失)", "confidence": "high", "source": "Elgie & McClay (2013); Yamazaki (2017)", "url": "https://utppublishing.com/doi/10.3138/CPP.39.Supplement2.S1"}},
        {"id": "D4_revenue_neutral", "dim": "structural", "weight": 1.0, "q": "是否收入中性(全额退税抵扣)", "actual": {"value": "是", "confidence": "high", "source": "BC Finance / Murray & Rivers (2015)", "url": "https://en.wikipedia.org/wiki/British_Columbia_carbon_tax"}},
        {"id": "D5_leakage", "dim": "distributional", "weight": 1.5, "q": "碳泄漏/贸易竞争力损害", "actual": {"value": "无显著证据", "confidence": "medium", "source": "Rivers & Schaufele, CJAE 63 (2015)", "url": "https://onlinelibrary.wiley.com/doi/abs/10.1111/cjag.12048"}},
        {"id": "D6_durability", "dim": "second_order", "weight": 1.5, "q": "政治存续与公众支持演变", "actual": {"value": "存续 · 支持升至 ~55-64%", "confidence": "high", "source": "Harrison; Murray & Rivers (2015)", "url": "https://www.sciencedirect.com/science/article/abs/pii/S0301421515300550"}},
    ],
    "INS-CA-CLIMATE-2023": [
        {"id": "D1_carrier_exit", "dim": "direct", "weight": 1.0, "q": "暂停/收缩新住房险的前 12 大承保商数量", "actual": {"value": "7 家", "confidence": "high", "source": "CA DOI / CalMatters", "url": "https://calmatters.org/politics/2023/09/california-insurance-crisis/"}},
        {"id": "D2_fair_plan", "dim": "distributional", "weight": 1.5, "q": "FAIR Plan(兜底险)保单量增幅", "actual": {"value": "+74%", "confidence": "high", "source": "California FAIR Plan", "url": "https://www.cfpnet.com/key-statistics-data/"}},
        {"id": "D3_premium", "dim": "direct", "weight": 1.0, "q": "全州住房险均保费涨幅", "actual": {"value": "+10% (2024)", "confidence": "medium", "source": "S&P Global", "url": "https://www.insurance.com/home-and-renters-insurance/home-insurance-rate-increases", "note": "窗口累计约 25–35%"}},
        {"id": "D4_nonrenewal", "dim": "distributional", "weight": 1.5, "q": "高火险县(最高)非续保率", "actual": {"value": "17.8%", "confidence": "medium", "source": "CA DOI", "url": "https://www.moneygeek.com/insurance/homeowners/california-wildfire-fair-plan-insurer-retreat/"}},
        {"id": "D5_regulatory", "dim": "second_order", "weight": 1.5, "q": "是否允许巨灾模型+再保成本计入费率", "actual": {"value": "是 (2024/12)", "confidence": "high", "source": "CA DOI · Sustainable Insurance Strategy", "url": "https://www.insurance.ca.gov/0400-news/0100-press-releases/2024/release065-2024.cfm"}},
        {"id": "D6_coverage_gap", "dim": "distributional", "weight": 1.5, "q": "火险区由兜底险承保比例(可及性缺口)", "actual": {"value": "~33%", "confidence": "low", "source": "FAIR Plan / CDI", "url": "https://www.moneygeek.com/insurance/homeowners/california-wildfire-fair-plan-insurer-retreat/", "note": "代理指标"}},
    ],
    "AD-INV-004": [
        {"id": "D1_return_12m", "dim": "direct", "weight": 1.0, "q": "上市~12 个月相对发行价回报", "actual": {"value": "-7.2% ↓ (跌破发行价)", "confidence": "medium", "source": "ADNOC FY2018 / companiesmarketcap", "url": "https://www.adnocdistribution.ae/-/media/adnoc-distribution/ir/quarter-results/2019-data-import/adnoc-distribution-q4-fy-2018-results-announcement-english-14-feb-2019.pdf"}},
        {"id": "D2_alpha_vs_adx", "dim": "second_order", "weight": 1.5, "q": "首年相对 ADX 指数超额收益", "actual": {"value": "约 -19pp", "confidence": "low", "source": "推导 vs ADX 2018", "url": "https://tradingeconomics.com/adsmi:ind"}},
        {"id": "D3_first_day", "dim": "timing", "weight": 1.0, "q": "上市首日相对发行价涨幅", "actual": {"value": "+6% (盘中 +16%)", "confidence": "high", "source": "The National / Bloomberg", "url": "https://www.thenationalnews.com/business/markets/adnoc-distribution-shares-surge-on-adx-debut-1.684026"}},
        {"id": "D4_dividend", "dim": "direct", "weight": 1.0, "q": "首年股息率 vs 招股指引", "actual": {"value": "~4.7%，达最低指引", "confidence": "medium", "source": "ADNOC Distribution", "url": "https://www.adnoc.ae/en/news-and-media/press-releases/2025/adnoc-listed-companies-target-record-aed-158-billion-43-billion-in-dividends"}},
        {"id": "D5_index_inclusion", "dim": "structural", "weight": 1.5, "q": "12 个月内是否纳入 MSCI/FTSE", "actual": {"value": "否 (2021 才纳入)", "confidence": "high", "source": "ADNOC Distribution 公告", "url": "https://www.adnocdistribution.ae/en/media/press-releases/2021/adnoc-distribution-included-in-msci-emerging-markets-index-effective-27-may-2021uae-listed-companies-to-be-part-of-msci-em-index/"}},
        {"id": "D6_volatility", "dim": "risk", "weight": 1.5, "q": "首 12 个月价格区间", "actual": {"value": "AED 2.17–2.74", "confidence": "medium", "source": "ADNOC FY2018", "url": "https://www.adnocdistribution.ae/-/media/adnoc-distribution/ir/quarter-results/2019-data-import/adnoc-distribution-q4-fy-2018-results-announcement-english-14-feb-2019.pdf"}},
    ],
}


# ── human-authored gold labels (DRAFT — review & correct) ──────────────────
# Two per case, the basis for two rubric sub-metrics that can't be auto-derived:
#   expected_stakeholders — the stakeholder TYPES any competent analysis must
#                           cover → Stakeholder Coverage = found / expected.
#   gold_risks            — the real top risks, ranked → the model's risk
#                           ordering is scored (NDCG/Recall@K) against this.
# Drafted from the same web research behind DIMENSIONS; treat as provisional
# until a domain reviewer signs off.
GOLD = {
    "WC-QATAR-2022": {
        "expected_stakeholders": ["组委会/最高委员会", "国家旅游局", "民航局/机场", "航空公司",
                                  "酒店/住宿", "邻国穿梭航班运营方", "球迷/游客", "本地居民",
                                  "媒体", "安保", "移民劳工", "餐饮/零售商家"],
        "gold_risks": ["住宿容量压力(酒店/球迷村/邮轮)", "机场与交通拥堵", "移民劳工权益争议",
                       "高价挤出普通游客", "赛后设施闲置(白象效应)"],
    },
    "CLIM-BC-CTAX-2008": {
        "expected_stakeholders": ["省政府/财政部", "家庭/通勤者", "燃油零售", "贸易暴露行业(水泥/农业)",
                                  "运输/商业企业", "环保团体", "反对党/纳税人团体",
                                  "经济学家/研究机构", "联邦/邻省政府"],
        "gold_risks": ["贸易暴露行业竞争力/碳泄漏担忧", "农村/长途通勤者负担不均",
                       "政治反弹/税负公平争议", "收入中性可信度", "城乡区域影响差异"],
    },
    "INS-CA-CLIMATE-2023": {
        "expected_stakeholders": ["州保险监管(CDI)", "大型承保商(State Farm/Allstate…)", "再保险公司",
                                  "FAIR Plan兜底险", "高火险区房主", "抵押贷款机构/银行",
                                  "房地产市场", "消费者团体", "州立法/政府"],
        "gold_risks": ["主要承保商退出/停新单", "FAIR Plan风险过度集中", "保费飙升/可负担性",
                       "高火险区无法投保(覆盖缺口)", "抵押贷款与房产交易受阻"],
    },
    "AD-INV-004": {
        "expected_stakeholders": ["发行人(母公司国油)", "承销投行", "主权/机构投资者", "散户投资者",
                                  "交易所(ADX)", "指数编制方(MSCI/FTSE)", "卖方分析师",
                                  "外资持股/监管"],
        "gold_risks": ["上市12个月跌破发行价", "相对ADX指数跑输", "油价/宏观波动",
                       "自由流通量小/流动性不足", "股息可持续性"],
    },
}


def _flatten_report_text(report):
    """Everything a competitor actually said, as one blob for the coverage judge."""
    parts = [report.get("summary", ""), report.get("rationale", ""), report.get("crossCutting", "")]
    for a in (report.get("angles") or []):
        parts.append(f"{a.get('name', '')}: {a.get('text', '') or a.get('perspective', '')}")
    parts += [str(x) for x in (report.get("keyFindings") or [])]
    parts += [str(x) for x in (report.get("risks") or [])]
    return "\n".join(str(p) for p in parts if p)[:6000]


def score_coverage(report, dims):
    """How many of the outcome scorecard's dimensions did this competitor's report
    actually get right? An impartial LLM judge reads the report the competitor
    ALREADY produced (uniform across LokaWorld and single-shot models — no prompt
    changes, no fabrication) and marks each dimension covered/not. Returns
    {coverage: weighted %, coveredDims: [ids]} or None if unavailable."""
    if not dims or not LLM_KEY:
        return None
    answer = _flatten_report_text(report or {})
    if not answer.strip():
        return {"coverage": 0, "coveredDims": []}
    judge_model = os.environ.get("COVERAGE_JUDGE_MODEL") or (MODELS[0] if MODELS else "gpt-4o")
    dims_txt = "\n".join(
        f"- {d['id']} [{d['dim']}]: {d['q']} — ACTUAL OUTCOME: {d['actual']['value']}"
        for d in dims)
    sys = ("You grade whether a forecast report correctly ADDRESSED and got each dimension of an "
           "outcome scorecard about right. For each dimension id output true ONLY if the report "
           "makes a claim matching the ACTUAL outcome (right direction and rough magnitude); output "
           "false if the report ignores that dimension or is materially wrong. Be strict: a report "
           "that gives only the headline number scores false on the second-order and distributional "
           "dimensions it never discusses. Output STRICT JSON with one boolean per dimension id and "
           "nothing else.")
    user = f"SCORECARD:\n{dims_txt}\n\nREPORT:\n{answer}\n\nJSON (one boolean per dimension id):"
    try:
        r = _post_retry(f"{LLM_BASE}/chat/completions",
                        headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
                        json={"model": judge_model, "temperature": 0,
                              "messages": [{"role": "system", "content": sys},
                                           {"role": "user", "content": user}]}, timeout=120)
        content = (((r.json() or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        m = re.search(r"\{.*\}", content, re.DOTALL)
        verdicts = json.loads(m.group(0)) if m else {}
    except Exception as e:  # noqa: BLE001
        print(f"    (coverage judge failed: {e})")
        return None
    covered = [d["id"] for d in dims if verdicts.get(d["id"]) is True]
    tot = sum(d["weight"] for d in dims) or 1
    hit = sum(d["weight"] for d in dims if d["id"] in covered)
    return {"coverage": round(hit / tot * 100), "coveredDims": covered}


def score_reasoning(report, case):
    """LLM judge for two rubric axes, 0-100 each, from the report a competitor
    produced:
      • reasoning       — evidence, explicit causal chains, internal consistency
                          (does the number follow from the stated drivers?)
      • explainability  — does it trace the conclusion to specific drivers /
                          actors / mechanisms a reader could audit?
    None if unavailable. Uniform across all competitors — no prompt changes."""
    if not LLM_KEY:
        return None
    answer = _flatten_report_text(report or {})
    if not answer.strip():
        return {"reasoning": 0, "explainability": 0}
    judge_model = os.environ.get("COVERAGE_JUDGE_MODEL") or (MODELS[0] if MODELS else "gpt-4o")
    sys = ("You grade a forecast report on two axes, 0-100 each. "
           "reasoning: quality of evidence, explicit causal chains, and internal "
           "consistency — does the forecast follow from the stated drivers? "
           "explainability: does it trace the conclusion to specific drivers, actors "
           "and mechanisms a reader could audit, versus a bare assertion? "
           "Reward concrete figures, named mechanisms and 'because X, therefore Y' logic; "
           "penalize vague or unsupported claims. "
           "Output STRICT JSON only: {\"reasoning\": <0-100>, \"explainability\": <0-100>}.")
    user = f"QUESTION: {case.get('question', '')}\n\nREPORT:\n{answer}\n\nJSON:"
    try:
        r = _post_retry(f"{LLM_BASE}/chat/completions",
                        headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
                        json={"model": judge_model, "temperature": 0,
                              "messages": [{"role": "system", "content": sys},
                                           {"role": "user", "content": user}]}, timeout=120)
        content = (((r.json() or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        m = re.search(r"\{.*\}", content, re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}
        clamp = lambda x: int(max(0, min(100, x))) if isinstance(x, (int, float)) else 0
        return {"reasoning": clamp(obj.get("reasoning")), "explainability": clamp(obj.get("explainability"))}
    except Exception as e:  # noqa: BLE001
        print(f"    (reasoning judge failed: {e})")
        return None


def _fidelity_score(sim):
    """0-100 Simulation Fidelity from the captured run internals. 0 when there is
    no simulation (single-shot models) — the 40% they structurally can't reach."""
    if not sim:
        return 0
    clamp = lambda x: max(0.0, min(100.0, x))
    persona = clamp((sim.get("persona") or {}).get("diversity", 0) * 100)
    inter = clamp((sim.get("interaction") or {}).get("cross_group_ratio", 0) * 100)
    opinion = clamp((sim.get("opinion") or {}).get("swing", 0) / 40 * 100)
    return round(0.40 * persona + 0.35 * inter + 0.25 * opinion)


def _prediction_score(metrics, case_type):
    """0-100 Prediction Accuracy: numerical closeness + direction. (Risk-ranking
    NDCG needs a human gold ranking — added once those labels exist.)"""
    if case_type == "match":
        return 100 if metrics.direction_correct else 0
    err = metrics.value_error_pct if metrics.value_error_pct is not None else 100
    numerical = max(0.0, 100.0 - min(100.0, err))
    direction = 100 if metrics.direction_correct else 0
    return round(0.6 * numerical + 0.4 * direction)


def capture_sim_fidelity(project_id, run_id, agent_count):
    """Capture the run's internals so we can score Simulation Fidelity (the 40%
    dimension single-shot models can't touch). Reads the project analytics + the
    run timeline and computes:
      • persona diversity  — normalized entropy over profession / country
      • interaction        — total edges, density, and the CROSS-GROUP ratio
                             (core↔ecosystem edges: the decision-core-vs-world
                             cooperation loop, quantified)
      • opinion evolution  — sentiment swing across rounds (proxy)
    Also keeps the analytics payload so the report modal can render real charts.
    Returns {} on failure — Fidelity just scores 0 then, which is honest."""
    import math
    import collections
    import requests
    cap = {}
    analytics = {}
    try:
        pdata = ((requests.get(f"{LOKA_URL}/api/project/{project_id}/data",
                               headers={"Accept-Language": "en"}, timeout=60)
                  .json() or {}).get("data")) or {}
        analytics = pdata.get("analytics") or {}
        cap["analytics"] = analytics
        cap["profile_count"] = pdata.get("profileCount") or 0
    except Exception as e:  # noqa: BLE001
        print(f"    (fidelity: project data fetch failed: {e})")

    professions, countries, depts = [], [], []
    agent_dept, agent_node, edges = {}, {}, []
    core_nodes = set()
    try:
        tl = ((requests.get(f"{LOKA_URL}/api/workflow/run/{run_id}/timeline",
                            params={"cursor": 0}, timeout=120).json() or {}).get("data")) or {}
        for ev in (tl.get("events") or []):
            d = ev.get("data") or {}
            t = ev.get("type")
            if t == "world_built":
                # New Topic→World pipeline: the decision core = authority≥4 nodes.
                # Gives us a PRECISE core-set for the cross-group (core↔ecosystem)
                # metric instead of the old "has a department" proxy.
                for nd in (d.get("nodes") or []):
                    if isinstance(nd, dict) and (nd.get("authority") or 0) >= 4 and nd.get("id"):
                        core_nodes.add(str(nd["id"]))
            elif t == "agent_spawned":
                if d.get("profession"): professions.append(str(d["profession"]))
                if d.get("country"): countries.append(str(d["country"]))
                if d.get("department"): depts.append(str(d["department"]))
                nm = str(d.get("name") or "")
                if nm:
                    agent_dept[nm] = str(d.get("department") or "")
                    if d.get("node_id"): agent_node[nm] = str(d["node_id"])
            elif t == "interaction_edges":
                for e in (d.get("edges") or []):
                    edges.append((e.get("from_name"), e.get("to_name"),
                                  e.get("from_node"), e.get("to_node")))
            elif t == "round_action" and d.get("target_name"):
                edges.append((d.get("agent_name"), d.get("target_name"),
                              d.get("node_id"), None))
    except Exception as e:  # noqa: BLE001
        print(f"    (fidelity: timeline fetch failed: {e})")

    def _entropy(items):
        if not items:
            return 0.0
        c = collections.Counter(items)
        n = len(items)
        h = -sum((v / n) * math.log(v / n, 2) for v in c.values())
        hmax = math.log(len(c), 2) if len(c) > 1 else 1.0
        return round(h / hmax, 3) if hmax else 0.0

    # Persona diversity: average entropy over the axes that actually VARY. The
    # world pipeline gives roles (profession) + institutions (department) but a
    # constant country; the old pipeline gave profession + country. Averaging only
    # the axes with ≥2 distinct values makes this robust to both (a constant axis
    # no longer silently halves the score).
    _axes = [ax for ax in (professions, countries, depts) if len(set(ax)) >= 2]
    diversity = round(sum(_entropy(ax) for ax in _axes) / len(_axes), 3) if _axes else 0.0
    cap["persona"] = {
        "professions": len(set(professions)), "countries": len(set(countries)),
        "departments": len(set(depts)), "diversity": diversity,
    }
    total = len(edges)
    cross = 0
    for a, b, na, nb in edges:
        # Prefer the node_id the event now carries directly; else map the name
        # (world-name agents only); else fall back to the department heuristic.
        na = na or agent_node.get(str(a))
        nb = nb or agent_node.get(str(b))
        if na and nb and core_nodes:
            # Precise: cross-group iff exactly one endpoint is in the decision
            # core (the core↔ecosystem cooperation loop).
            if (na in core_nodes) != (nb in core_nodes):
                cross += 1
        else:
            da, db = (agent_dept.get(str(a)) or ""), (agent_dept.get(str(b)) or "")
            if bool(da.strip()) != bool(db.strip()):
                cross += 1
    n = cap.get("profile_count") or agent_count or 0
    cap["interaction"] = {
        "edges": total,
        "cross_group_ratio": round(cross / total, 3) if total else 0.0,
        "density": round(total / (n * (n - 1) / 2), 5) if n > 1 else 0.0,
    }
    stl = ((analytics.get("sentiment") or {}).get("timeline")) or []
    scores = [p.get("score") for p in stl if isinstance(p, dict) and p.get("score") is not None]
    cap["opinion"] = {
        "rounds": len(stl),
        "swing": round(max(scores) - min(scores), 1) if len(scores) >= 2 else 0.0,
    }
    return cap


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
                          json={"question": case["question"]},
                          headers={"Accept-Language": "en"}, timeout=300)
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
    dag = ((_post_retry(f"{LOKA_URL}/api/workflow/plan", json=body,
                        headers={"Accept-Language": "en"}, timeout=600)
            .json() or {}).get("data")) or {}

    # Start the run — self-healing against the one-active-run-per-owner gate:
    # a 429 means a previous run (often an orphan from a Ctrl-C'd script) is
    # still blocking this owner. Cancel it, wait for it to die, retry once.
    def _start_run():
        # Accept-Language: en → the whole run (status messages, report
        # sections, prediction-block rationale) comes out in English, so the
        # benchmark page isn't a zh/en patchwork.
        return requests.post(f"{LOKA_URL}/api/workflow/run",
                             json={"workflow_id": dag.get("workflow_id"), "dag": dag},
                             headers={"Accept-Language": "en"}, timeout=60)
    r = _start_run()
    if r.status_code == 429:
        try:
            blocker = (r.json() or {}).get("active_run_id")
        except ValueError:
            blocker = None
        print(f"    [deep] owner busy (active run {blocker or '?'}) — "
              f"cancelling it and retrying", flush=True)
        if blocker:
            try:
                requests.post(f"{LOKA_URL}/api/workflow/run/{blocker}/cancel", timeout=30)
            except Exception:
                pass
            _end = _t.time() + 240
            _live = ("running", "queued", "planning",
                     "awaiting_decision", "awaiting_community")
            while _t.time() < _end:
                try:
                    stb = ((requests.get(f"{LOKA_URL}/api/workflow/run/{blocker}/status",
                                         timeout=30).json() or {}).get("data")) or {}
                except Exception:
                    break
                if stb.get("status") not in _live:
                    break
                _t.sleep(10)
        r = _start_run()
    r.raise_for_status()
    run = (r.json() or {}).get("data") or {}
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
    if direction is None:
        # zh reports sometimes phrase the binary verdict outside the direction
        # token — scan the block's own rationale before scoring a miss
        direction = _norm_dir(case, f"{ep0.direction or ''} {ep0.rationale or ''}",
                              ep0.point_estimate)
    ep = EventPrediction(direction=direction, point_estimate=ep0.point_estimate,
                         unit=ep0.unit, ci_80=ep0.ci_80,
                         confidence=ep0.confidence, rationale=ep0.rationale)
    # Carve the FULL consulting report into sections for the frontend modal —
    # earlier this threw the 6-section report away and shipped one sentence,
    # which made a 20-minute simulation look like a fortune cookie.
    import re as _re
    sections = []
    for sec in _re.split(r"\n(?=##\s)", md):
        m2 = _re.match(r"##\s*(.+)", sec)
        if not m2:
            continue
        title = m2.group(1).strip()[:90]
        bodytxt = sec[m2.end():]
        bodytxt = _re.sub(r"<!--.*?-->", "", bodytxt, flags=_re.DOTALL)
        bodytxt = _re.sub(r"[#*`>|]+", "", bodytxt)      # light de-markdown
        bodytxt = _re.sub(r"\n{2,}", "\n\n", bodytxt).strip()
        if bodytxt:
            sections.append({"name": title, "text": bodytxt[:2200]})
    exec_txt = next((a["text"] for a in sections
                     if "summary" in a["name"].lower() or "执行摘要" in a["name"]), "")
    report = {"method": (f"Deep OASIS simulation ({body['agent_count']} agents x "
                         f"{body['max_rounds']} rounds) -> consulting report"
                         + (" · confirmed structure" if topic_analysis else "")),
              "summary": exec_txt or ep0.rationale or md[:600],
              "rationale": ep0.rationale or "",
              "angles": sections[:8],
              # Simulation-fidelity capture (Step 2): agent diversity, the
              # core↔ecosystem interaction loop, opinion swing, + analytics for
              # charts. This is the data single-shot models structurally can't
              # produce — the 40% of the score they can't reach.
              "sim": capture_sim_fidelity(project_id, run_id, body.get("agent_count"))}
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


def _anon_case(case):
    """The case with its memorization-proof twin question (names scrubbed,
    every number and constraint preserved). A competitor that aces the real
    case but flunks the twin was reciting, not reasoning."""
    c2 = dict(case)
    c2["question"] = case["anon_question"]
    return c2


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
    # DEFAULT: every competitor answers the ANONYMIZED question only —
    # identifying names scrubbed, all numbers/constraints preserved — so a
    # famous historical outcome cannot simply be recited from training data.
    #   --real   ask the original named questions instead (legacy mode)
    #   --twins  ask BOTH and compute the memorization gap (2x cost;
    #            methodology demos)
    twins = "--twins" in sys.argv
    use_real = "--real" in sys.argv
    # --cases=ID1,ID2 — rerun only these cases (e.g. the ones a flaky upstream
    # killed); results for the untouched cases are merged in from the previous
    # results/arena_live.json so --emit-js still produces the COMPLETE drop-in.
    case_filter = None
    for _a in sys.argv:
        if _a.startswith("--cases="):
            case_filter = {c.strip() for c in _a.split("=", 1)[1].split(",") if c.strip()}
    run_cases = [c for c in CASES if not case_filter or c["id"] in case_filter]
    competitors = [LOKA_ID] + MODELS
    RESULTS.mkdir(exist_ok=True)
    per_case, scored = [], {c: [] for c in competitors}

    for case in run_cases:
        gt = _gt(case)
        # the question competitors actually receive
        qcase = case if (use_real or twins) else _anon_case(case)
        anonymized = not (use_real or twins)
        print(f"\n=== {case['id']} — {case['entity']}"
              f"{' [ANONYMIZED PROMPT]' if anonymized else ''} ===")
        row = {"case_id": case["id"], "entity": case["entity"], "tag": case["tag"],
               "asOf": case["asOf"], "horizon": case["horizon"], "type": case["type"],
               "unit": case.get("unit"), "options": case.get("options"),
               "anonymized": anonymized,
               "question": qcase["question"], "actual": case["gt"], "entries": {}}
        for comp in competitors:
            try:
                if dry:
                    ep, report = _dry_predict(qcase, comp)
                elif comp == LOKA_ID:
                    ep, report = (predict_lokaworld_deep(qcase, confirm_structure)
                                  if deep else predict_lokaworld(qcase))
                else:
                    ep, report = predict_raw_model(qcase, comp)
                err = None
            except Exception as e:  # noqa: BLE001
                ep, report, err = None, {"method": "", "summary": f"(failed: {e})", "angles": []}, str(e)
            _rescale_to_case_unit(ep, case)
            metrics = score_prediction(ep, gt)

            # ── memorization check: same competitor, anonymized twin ──
            anon_entry = None
            if twins and err is None:
                try:
                    ac = _anon_case(case)
                    if dry:
                        ep_a, _ = _dry_predict(ac, comp)
                    elif comp == LOKA_ID:
                        ep_a, _ = (predict_lokaworld_deep(ac, confirm_structure)
                                   if deep else predict_lokaworld(ac))
                    else:
                        ep_a, _ = predict_raw_model(ac, comp)
                    _rescale_to_case_unit(ep_a, case)
                    m_a = score_prediction(ep_a, gt, is_anonymized=True)
                    from aegeanbench.scoring import apply_memorization_gap
                    apply_memorization_gap(metrics, m_a)
                    anon_entry = {
                        "pred": ep_a.point_estimate if ep_a else None,
                        "ci": ep_a.ci_80 if ep_a else None,
                        "direction": m_a.predicted_direction,
                        "errPct": None if m_a.value_error_pct is None else round(m_a.value_error_pct),
                        "within": m_a.within_ci,
                        "correct": m_a.direction_correct if case["type"] == "match" else
                                   (m_a.value_error_pct is not None and m_a.value_error_pct <= 15),
                    }
                except Exception as ae:  # noqa: BLE001
                    print(f"    (anon twin failed for {comp}: {ae})")

            # Scorecard coverage: an impartial judge reads the report this
            # competitor just produced and marks which outcome dimensions it
            # actually got right (weighted). Skipped for dry runs.
            cov = None if (dry or err) else score_coverage(report, DIMENSIONS.get(case["id"]))

            # ── Four-dimension rubric (Simulation Fidelity 40 / Prediction 30 /
            # Reasoning 20 / Explainability 10). Fidelity comes from the captured
            # run internals (0 for single-shot models — the part they can't
            # reach); reasoning/explainability from an impartial judge. This is
            # what turns the benchmark from "who guessed the number" into "who
            # reconstructed the decision process". ──
            _rj = None if (dry or err) else score_reasoning(report, case)
            fidelity = _fidelity_score((report or {}).get("sim")) if not err else 0
            prediction = 0 if err else _prediction_score(metrics, case["type"])
            reasoning = (_rj or {}).get("reasoning", 0)
            explainability = (_rj or {}).get("explainability", 0)
            composite = round(0.40 * fidelity + 0.30 * prediction
                              + 0.20 * reasoning + 0.10 * explainability)

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
                "anon": anon_entry,
                "memGap": metrics.memorization_gap,
                "gapLevel": metrics.gap_level,
                "coverage": (cov or {}).get("coverage"),
                "coveredDims": (cov or {}).get("coveredDims"),
                "fidelity": None if err else fidelity,
                "prediction": None if err else prediction,
                "reasoning": None if err else reasoning,
                "explainability": None if err else explainability,
                "composite": None if err else composite,
                "report": report, "error": err,
            }
            v = row["entries"][comp]
            gap_s = ("" if v["memGap"] is None
                     else f"  memGap={v['memGap']} ({v['gapLevel']})")
            print(f"  {comp:16} dir={metrics.predicted_direction} "
                  f"val={ep.point_estimate if ep else '-'} err={v['errPct']}% "
                  f"within={v['within']} correct={v['correct']}{gap_s}"
                  + (f"  ERR:{err}" if err else ""))
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
    # merge untouched cases from the previous run so partial reruns still
    # yield a complete result set (order follows CASES)
    if case_filter and (RESULTS / "arena_live.json").exists():
        try:
            prev_rows = {r["case_id"]: r for r in
                         json.loads((RESULTS / "arena_live.json").read_text(encoding="utf-8"))
                         .get("per_case", [])}
            fresh = {r["case_id"]: r for r in per_case}
            per_case = [fresh.get(c["id"]) or prev_rows.get(c["id"])
                        for c in CASES]
            per_case = [r for r in per_case if r]
            print(f"(merged {len(per_case) - len(fresh)} case(s) from previous results)")
        except Exception as me:  # noqa: BLE001
            print(f"(merge with previous results failed: {me})")

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
        rows = [r["entries"].get(comp) for r in per_case if r["entries"].get(comp)]
        n = len(rows) or 1
        correct = sum(1 for e in rows if e.get("correct"))
        errs = [e["errPct"] for e in rows if e.get("errPct") is not None]
        cis = [e for e in rows if e.get("within") is not None]
        ci_hit = sum(1 for e in cis if e.get("within"))
        leaderboard.append({"competitor": comp, "n": len(rows), "correct": correct,
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
            _cov = {"coverage": e.get("coverage"), "coveredDims": e.get("coveredDims")}
            _rub = {"fidelity": e.get("fidelity"), "prediction": e.get("prediction"),
                    "reasoning": e.get("reasoning"), "explainability": e.get("explainability"),
                    "composite": e.get("composite")}
            if row["type"] == "match":
                entries[comp] = {"choice": _opt_label(e["direction"]), "confidence": e["confidence"],
                                 "calConf": e.get("calConf"),
                                 "anon": e.get("anon"), "memGap": e.get("memGap"),
                                 "gapLevel": e.get("gapLevel"), **_cov, **_rub,
                                 "correct": e["correct"], "report": e["report"]}
            else:
                entries[comp] = {"pred": e["pred"], "ci": e["ci"], "confidence": e["confidence"],
                                 "calConf": e.get("calConf"),
                                 "anon": e.get("anon"), "memGap": e.get("memGap"),
                                 "gapLevel": e.get("gapLevel"), **_cov, **_rub,
                                 "errPct": e["errPct"], "within": e["within"], "correct": e["correct"],
                                 "report": e["report"]}
        g = row["actual"]   # raw ground-truth dict
        actual = {"value": g.get("actual_value"), "headline": g.get("headline") or g.get("direction_label", ""),
                  "detail": g.get("narrative", ""), "source": g.get("source", "")}
        if row["type"] == "match":
            actual["choice"] = _opt_label(g.get("direction_label"))
        cases_js.append(f"""  {{
    id: {js(row['case_id'])}, entity: {js(row['entity'])}, tag: {js(row['tag'])},
    anonymized: {js(bool(row.get('anonymized')))},
    horizon: {js(row['horizon'])}, asOf: {js(row['asOf'])}, type: {js(row['type'])},
    unit: {js(row.get('unit'))}, {'options: ' + js(row['options']) + ',' if row.get('options') else ''}
    question: {js(row['question'])},
    actual: {js(actual)},
    dimensions: {js(DIMENSIONS.get(row['case_id'], []))},
    entries: {js(entries)},
  }}""")

    # Per-competitor scorecard coverage (weighted % + which dims each got right),
    # judged from the report each competitor actually produced — this is the
    # real, computed version of the coverage the page shows.
    sc_cov = {}
    for row in per_case:
        cm = {}
        for comp, e in row["entries"].items():
            if e.get("coverage") is not None:
                cm[comp] = {"coverage": e["coverage"], "coveredDims": e.get("coveredDims") or []}
        if cm:
            sc_cov[row["case_id"]] = cm
    # computed exports the frontend needs (landing.js reads BENCHMARK_SUMMARY,
    # benchmark.js screen reads LEADERBOARD) — mirror src/data/benchmark.js so
    # this file is a COMPLETE drop-in.
    cov_js = "export const SCORECARD_COVERAGE = " + js(sc_cov) + ";\n\n"
    tail = cov_js + """export const LEADERBOARD = COMPETITORS.map(m => {
  const rows = BENCHMARK.map(c => c.entries[m.id]).filter(Boolean);
  const covRows = BENCHMARK.map(c => (SCORECARD_COVERAGE[c.id] || {})[m.id]).filter(Boolean);
  const coverage = covRows.length
    ? Math.round(covRows.reduce((a, b) => a + (b.coverage || 0), 0) / covRows.length) : null;
  const n = rows.length || 1;
  const correct = rows.filter(r => r.correct).length;
  const errs = rows.map(r => r.errPct).filter(e => e !== null && e !== undefined).sort((a, b) => a - b);
  const medianErr = errs.length ? (errs.length % 2 ? errs[(errs.length - 1) / 2]
    : Math.round((errs[errs.length / 2 - 1] + errs[errs.length / 2]) / 2)) : null;
  const ciRows = rows.filter(r => r.within !== null && r.within !== undefined);
  const ciHit = ciRows.filter(r => r.within).length;
  const gaps = rows.map(r => r.memGap).filter(g => g !== null && g !== undefined);
  const memGap = gaps.length ? Math.round(gaps.reduce((a, b) => a + b, 0) / gaps.length * 100) / 100 : null;
  const accuracy = Math.round((correct / n) * 100);
  // Memorization-adjusted ranking: (1 - memGap)^2 discounts correctness that only
  // survives with the real names, so recall can't top genuine forward reasoning.
  const trust = Math.pow(1 - Math.min(1, memGap || 0), 2);
  const genScore = Math.round(accuracy * trust);
  // Four-dimension rubric (Simulation Fidelity 40 / Prediction 30 / Reasoning 20 /
  // Explainability 10), averaged over cases. Present once a run has scored them;
  // null → the leaderboard falls back to the memorization-adjusted sort.
  const _avg = (f) => { const xs = rows.map(f).filter(x => x != null && !isNaN(x)); return xs.length ? Math.round(xs.reduce((a, b) => a + b, 0) / xs.length) : null; };
  const fidelity = _avg(r => r.fidelity);
  const prediction = _avg(r => r.prediction);
  const reasoning = _avg(r => r.reasoning);
  const explainability = _avg(r => r.explainability);
  const composite = _avg(r => r.composite);
  return { id: m.id, name: m.name, kind: m.kind, tagline: m.tagline,
    n, correct, accuracy, medianErr, memGap, genScore, coverage,
    fidelity, prediction, reasoning, explainability, composite,
    ciHit, ciTotal: ciRows.length, ciPct: ciRows.length ? Math.round((ciHit / ciRows.length) * 100) : null };
}).sort((a, b) => (b.composite ?? -1) - (a.composite ?? -1) || b.genScore - a.genScore || b.accuracy - a.accuracy);

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
