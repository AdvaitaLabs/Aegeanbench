"""
Event / Scenario backtest cases — Clinton Foundation & Abu Dhabi government.

Each scenario is a real historical case, scored against its realized outcome.
Every scenario expands into a TWIN PAIR of BenchmarkCases:

    * real — real entity names + real analysis date
    * anon — identity masked + date shifted, SAME facts / numbers

linked by ``twin_case_id`` and sharing one ``EventGroundTruth``. Running both
and comparing scores yields the memorization gap (small gap = genuine
reasoning, not recall of famous-entity news).

Ground-truth ``confidence``:
    "high"   — verified public fact, demo-safe as-is.
    "approx" — direction / magnitude correct, but the exact figure is from
               memory; verify against ``source`` before a client-facing demo.
"""

from __future__ import annotations

from typing import Any, Dict, List

from aegeanbench.core.models import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkSuite,
    Difficulty,
    EventGroundTruth,
)


# Each entry expands into a real + anon twin. `internal_only` marks a
# controversy case that must be filtered out of any client-facing view.
SCENARIOS: List[Dict[str, Any]] = [
    # ───────────────────────── Clinton Foundation ─────────────────────────
    {
        "id": "CF-FUND-001", "entity_tag": "clinton_foundation",
        "name": "Clinton Foundation post-2016-election fundraising",
        "scenario_type": "fundraising", "difficulty": "medium", "horizon": "24m",
        "question": "创始人配偶败选后，基金会未来两年捐赠收入会怎么走？",
        "ground_truth": {
            "direction_label": "decline", "actual_value": -58, "unit": "percent",
            "narrative": "2017 年捐赠约 2,660 万美元，较 2016 年约 6,290 万美元下降约 58%",
            "source": "IRS Form 990 / AP 报道", "confidence": "approx",
        },
        "real": {"analysis_date": "2016-11-09", "entity_shown": "Clinton Foundation",
                 "public_facts": [
                     "基金会主要收入来自个人/企业/外国政府捐赠",
                     "Bill Clinton 曾公开承诺若希拉里当选将停止外国与企业捐赠",
                     "CGI 承诺规模累计声称约 1850 亿美元"]},
        "anon": {"analysis_date": "2091-11-09",
                 "entity_shown": "某前国家元首夫妇创立的大型慈善基金会",
                 "public_facts": [
                     "基金会收入主要来自个人/企业/外国政府捐赠",
                     "创始人曾公开承诺若配偶当选将停止外国与企业捐赠",
                     "旗下旗舰倡议累计承诺规模声称约 1850 亿美元"]},
    },
    {
        "id": "CF-CGI-002", "entity_tag": "clinton_foundation",
        "name": "Clinton Global Initiative continuity after the election",
        "scenario_type": "program_continuity", "difficulty": "medium", "horizon": "12m",
        "question": "大选后该旗舰全球倡议会停办还是继续？",
        "ground_truth": {
            "direction_label": "suspended", "actual_value": None, "unit": None,
            "narrative": "2016-09 办完最后一届年会后停办、裁员约 22 人，2022-09 才重启",
            "source": "公开新闻时间线 / WARN 裁员通知", "confidence": "high",
        },
        "real": {"analysis_date": "2016-09-01", "entity_shown": "Clinton Global Initiative",
                 "public_facts": [
                     "Bill Clinton 已公开暗示大选后将重组基金会治理",
                     "CGI 依赖企业与外国捐赠者的年度承诺模式"]},
        "anon": {"analysis_date": "2091-09-01",
                 "entity_shown": "某基金会旗下的旗舰年度全球倡议峰会",
                 "public_facts": [
                     "创始人已公开暗示大选后将重组基金会治理",
                     "该倡议依赖企业与外国捐赠者的年度承诺模式"]},
    },
    {
        "id": "CF-CHAI-003", "entity_tag": "clinton_foundation",
        "name": "CHAI HIV drug-price negotiation outcome",
        "scenario_type": "program_outcome", "difficulty": "hard", "horizon": "36m",
        "question": "启动药价谈判后，数年内能否把发展中国家抗艾滋治疗成本大幅压低？",
        "ground_truth": {
            "direction_label": "target_achieved", "actual_value": 140, "unit": "usd_per_year",
            "narrative": "一线 ARV 年治疗成本从约 1 万美元/人降至约 140 美元/人，覆盖扩大至数百万人",
            "source": "CHAI / WHO 公开成效报告", "confidence": "high",
        },
        "real": {"analysis_date": "2003-06-01", "entity_shown": "Clinton Health Access Initiative",
                 "public_facts": [
                     "当时发展中国家一线 ARV 治疗年费约 1 万美元/人",
                     "计划通过规模化采购与仿制药厂（印度等）谈判压价"]},
        "anon": {"analysis_date": "2077-06-01", "entity_shown": "某健康类慈善机构",
                 "public_facts": [
                     "当时发展中国家该病一线治疗年费约 1 万美元/人",
                     "计划通过规模化采购与仿制药厂谈判压价"]},
    },
    {
        "id": "CF-SENT-004", "entity_tag": "clinton_foundation", "internal_only": True,
        "name": "Clinton Foundation media sentiment (Uranium One / Clinton Cash)",
        "scenario_type": "public_sentiment", "difficulty": "medium", "horizon": "3m",
        "question": "调查报道发布前后 1–3 个月，围绕基金会的媒体舆情情绪会怎么走？",
        "ground_truth": {
            "direction_label": "negative_spike", "actual_value": None, "unit": None,
            "narrative": "NYT 2015-04-23 发 Uranium One 报道、《Clinton Cash》2015-05 出版，负面舆情显著飙升",
            "source": "公开新闻事件", "confidence": "high",
        },
        "real": {"analysis_date": "2015-04-20", "entity_shown": "Clinton Foundation",
                 "public_facts": [
                     "希拉里已宣布参选总统",
                     "基金会捐赠人含外国实体，媒体关注捐赠与政策的利益关联"]},
        "anon": {"analysis_date": "2088-04-20", "entity_shown": "某知名政治家族慈善基金会",
                 "public_facts": [
                     "该家族核心成员已宣布参选总统",
                     "基金会捐赠人含外国实体，媒体关注捐赠与政策的利益关联"]},
    },
    {
        "id": "CF-HAITI-005", "entity_tag": "clinton_foundation",
        "name": "Clinton Bush Haiti Fund fundraising + later scrutiny",
        "scenario_type": "fundraising_and_sentiment", "difficulty": "hard", "horizon": "36m",
        "question": "重大灾害后基金发起，短期能筹多少、后续会不会出现善款去向质疑？",
        "ground_truth": {
            "direction_label": "strong_raise", "actual_value": 54_400_000, "unit": "usd",
            "narrative": "累计筹得约 5,440 万美元、2012 年结束；数年后(约2015)出现善款效率与去向质疑",
            "source": "基金会募捐记录 + 事后媒体报道", "confidence": "approx",
        },
        "real": {"analysis_date": "2010-01-16", "entity_shown": "Clinton Bush Haiti Fund",
                 "public_facts": [
                     "2010-01-12 海地 7.0 级大地震，国际救援呼吁强烈",
                     "Bill Clinton 时任联合国海地事务特使"]},
        "anon": {"analysis_date": "2085-01-16", "entity_shown": "两位卸任国家元首联合发起的赈灾基金",
                 "public_facts": [
                     "某国发生重大自然灾害，国际救援呼吁强烈",
                     "其中一位发起人时任相关国际事务特使"]},
    },
    # ───────────────────────── Abu Dhabi government ────────────────────────
    {
        "id": "AD-PROJ-001", "entity_tag": "abu_dhabi",
        "name": "Louvre Abu Dhabi first-year attendance",
        "scenario_type": "megaproject_outcome", "difficulty": "medium", "horizon": "12m",
        "question": "引入世界顶级博物馆品牌、开馆后首年客流大概多少？",
        "ground_truth": {
            "direction_label": "high_turnout", "actual_value": 1_000_000, "unit": "visitors",
            "narrative": "开馆首年访客超过 100 万",
            "source": "Louvre Abu Dhabi / DCT Abu Dhabi 官方", "confidence": "approx",
        },
        "real": {"analysis_date": "2017-11-11", "entity_shown": "Louvre Abu Dhabi（阿布扎比政府）",
                 "public_facts": [
                     "与法国签署 30 年品牌授权+藏品借展协议",
                     "位于萨迪亚特文化区，定位区域文化旅游枢纽",
                     "阿布扎比推动旅游作为经济多元化支柱"]},
        "anon": {"analysis_date": "2091-11-11", "entity_shown": "某海湾产油国引入的世界顶级博物馆品牌分馆",
                 "public_facts": [
                     "与品牌母国签署 30 年授权+藏品借展协议",
                     "位于新建文化区，定位区域文化旅游枢纽",
                     "该国推动旅游作为经济多元化支柱"]},
    },
    {
        "id": "AD-PROJ-002", "entity_tag": "abu_dhabi",
        "name": "Barakah nuclear plant first-unit schedule",
        "scenario_type": "megaproject_schedule", "difficulty": "hard", "horizon": "60m",
        "question": "该国首座核电站首堆能否在约 2017 年如期商运？",
        "ground_truth": {
            "direction_label": "delayed", "actual_value": 4, "unit": "years_delay",
            "narrative": "首堆 2020-08 并网、2021-04 商运，较原目标约2017延期约4年",
            "source": "ENEC / KEPCO 公开时间线", "confidence": "high",
        },
        "real": {"analysis_date": "2012-07-01", "entity_shown": "Barakah 核电站（阿布扎比政府）",
                 "public_facts": [
                     "UAE 首座核电站，承包方为韩国 KEPCO",
                     "共 4 台机组，首堆原计划约 2017 年投运",
                     "核电监管审批与运营许可流程复杂"]},
        "anon": {"analysis_date": "2072-07-01", "entity_shown": "某海湾产油国首座核电站",
                 "public_facts": [
                     "该国首座核电站，承包方为一家东亚电力集团",
                     "共 4 台机组，首堆原计划约 5 年后投运",
                     "核电监管审批与运营许可流程复杂"]},
    },
    {
        "id": "AD-REG-003", "entity_tag": "abu_dhabi",
        "name": "UAE 100% foreign-ownership reform → FDI reaction",
        "scenario_type": "policy_market_reaction", "difficulty": "medium", "horizon": "24m",
        "question": "取消内地公司外资 49% 上限、允许 100% 外资持股后，外资流入会怎么走？",
        "ground_truth": {
            "direction_label": "fdi_increase", "actual_value": 21_000_000_000, "unit": "usd",
            "narrative": "UAE 年度 FDI 流入维持约 200 亿美元级并上升，稳居西亚/中东地区首位",
            "source": "UNCTAD World Investment Report", "confidence": "approx",
        },
        "real": {"analysis_date": "2020-11-24", "entity_shown": "UAE / 阿布扎比 100% 外资持股改革",
                 "public_facts": [
                     "此前内地公司外资持股上限 49%",
                     "改革于 2021-06 生效",
                     "区域引资竞争加剧（邻国推进 Vision 2030）"]},
        "anon": {"analysis_date": "2090-11-24", "entity_shown": "某海湾产油国外资持股改革",
                 "public_facts": [
                     "此前内地公司外资持股上限 49%",
                     "改革意在提升营商吸引力",
                     "区域引资竞争加剧"]},
    },
    {
        "id": "AD-INV-004", "entity_tag": "abu_dhabi",
        "name": "ADNOC Distribution IPO post-listing direction",
        "scenario_type": "ipo_performance", "difficulty": "medium", "horizon": "12m",
        "question": "国有石油下游分销子公司 IPO（发行价 2.50），上市后 3–12 个月股价方向？",
        "ground_truth": {
            "direction_label": "up", "actual_value": None, "unit": None,
            "narrative": "上市后一年内股价高于 2.50 AED 发行价（上行）；精确价格路径以 ADX 行情为准",
            "source": "Abu Dhabi Securities Exchange (ADX)", "confidence": "approx",
        },
        "real": {"analysis_date": "2017-12-13", "entity_shown": "ADNOC Distribution（ADX 上市）",
                 "public_facts": [
                     "IPO 发行价 2.50 AED，出售约 10% 股权，募资约 8.5 亿美元",
                     "母公司为阿布扎比国家石油公司 ADNOC",
                     "下游燃油分销业务，派息政策明确"]},
        "anon": {"analysis_date": "2091-12-13", "entity_shown": "某国有石油公司下游分销子公司 IPO",
                 "public_facts": [
                     "IPO 发行价 2.50 本币，出售约 10% 股权",
                     "母公司为该国国家石油公司",
                     "下游燃油分销业务，派息政策明确"]},
    },
    {
        "id": "AD-KPI-005", "entity_tag": "abu_dhabi",
        "name": "Abu Dhabi Economic Vision 2030 non-oil GDP trajectory",
        "scenario_type": "policy_kpi", "difficulty": "hard", "horizon": "long",
        "question": "发布 2030 经济愿景（目标大幅提升非油 GDP 占比）后，占比走向如何？",
        "ground_truth": {
            "direction_label": "rising", "actual_value": 53, "unit": "percent",
            "narrative": "非油 GDP 占比长期上升，至 2023 年前后达约 53% 历史高位，随油价周期波动",
            "source": "SCAD 阿布扎比统计中心", "confidence": "approx",
        },
        "real": {"analysis_date": "2008-11-01", "entity_shown": "Abu Dhabi Economic Vision 2030",
                 "public_facts": [
                     "愿景目标将非油 GDP 占比大幅提升（目标约 64%）",
                     "当时经济高度依赖石油收入",
                     "多元化举措涵盖旅游/金融/制造/文化"]},
        "anon": {"analysis_date": "2081-11-01", "entity_shown": "某依赖石油的酋长国 2030 经济愿景",
                 "public_facts": [
                     "愿景目标大幅提升非油 GDP 占比、降低石油依赖",
                     "当时经济高度依赖石油收入",
                     "多元化举措涵盖旅游/金融/制造/文化"]},
    },
]


def _twin_pair(s: Dict[str, Any]) -> List[BenchmarkCase]:
    """Expand one scenario dict into its real + anon BenchmarkCase twins."""
    gt = EventGroundTruth(**s["ground_truth"])
    internal = bool(s.get("internal_only"))
    cases: List[BenchmarkCase] = []
    for variant in ("real", "anon"):
        v = s[variant]
        suffix, twin_suffix = ("R", "A") if variant == "real" else ("A", "R")
        tags = ["event", s["entity_tag"], s["scenario_type"], variant]
        if internal:
            tags.append("internal_only")
        cases.append(BenchmarkCase(
            case_id=f"{s['id']}-{suffix}",
            name=f"{s['name']} ({variant})",
            description=s["question"],
            category=BenchmarkCategory.EVENT,
            difficulty=Difficulty(s.get("difficulty", "medium")),
            tags=tags,
            scenario_request={
                "question": s["question"],
                "analysis_date": v["analysis_date"],
                "horizon": s["horizon"],
                "entity_shown": v["entity_shown"],
                "public_facts": v["public_facts"],
                "scenario_type": s["scenario_type"],
            },
            event_ground_truth=gt,
            is_anonymized=(variant == "anon"),
            twin_case_id=f"{s['id']}-{twin_suffix}",
            metadata={"internal_only": internal, "scenario_id": s["id"]},
        ))
    return cases


def load_event_suite(*, include_internal: bool = True) -> BenchmarkSuite:
    """
    Load the Clinton Foundation + Abu Dhabi scenario suite (real + anon twins).

    Set ``include_internal=False`` to drop controversy cases tagged
    ``internal_only`` (use for any client-facing run).
    """
    cases: List[BenchmarkCase] = []
    for s in SCENARIOS:
        if not include_internal and s.get("internal_only"):
            continue
        cases.extend(_twin_pair(s))
    return BenchmarkSuite(
        name="Loka Event Backtest Suite (Clinton Foundation + Abu Dhabi)",
        description="Historical scenario backtests scored against realized outcomes, "
                    "each with a real/anonymized twin for memorization-gap.",
        version="0.1.0",
        cases=cases,
        metadata={"entities": ["clinton_foundation", "abu_dhabi"],
                  "twin_paired": True, "scoring": "event"},
    )
