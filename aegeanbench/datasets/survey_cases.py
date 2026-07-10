"""
Survey calibration dataset — historical events benchmarked via the synthetic
population/survey path (SurveyRunner).

Each case carries a point-in-time population spec (marginals, or a GeoScope +
per-unit marginals for electoral cases), a survey question, a `predict` spec
telling the runner how to turn results into a prediction, and the realized
EventGroundTruth.

⚠️ The `marginals` here are ILLUSTRATIVE (approximate demographic splits) so the
pipeline is runnable end-to-end; the OUTCOMES (ground truth) are real. Replace
marginals with actual census/exit-poll data before trusting accuracy numbers.
(Same convention as the event dataset: outcomes real, distributions to-verify.)
"""

from __future__ import annotations

from typing import List

from aegeanbench.core.models import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkSuite,
    Difficulty,
    EventGroundTruth,
)

# Approximate adult marginals reused across cases (illustrative).
_AGE = {"18-24": 0.12, "25-34": 0.18, "35-49": 0.26, "50-64": 0.26, "65+": 0.18}
_EDU = {"no_degree": 0.6, "degree": 0.4}
_GENDER = {"male": 0.49, "female": 0.51}


def _brexit() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="SV-BREXIT-2016", name="UK Brexit referendum 2016",
        description="公投结果：Leave 还是 Remain？",
        category=BenchmarkCategory.EVENT, difficulty=Difficulty.HARD,
        tags=["survey", "referendum", "uk"],
        scenario_request={
            "size": 2000,
            "marginals": {"age": _AGE, "education": _EDU, "gender": _GENDER},
            "questions": [{"qid": "vote", "type": "single_choice",
                           "text": "在脱欧公投中你会投？", "options": ["Leave", "Remain"]}],
            "predict": {"qid": "vote", "mode": "top_choice"},
        },
        event_ground_truth=EventGroundTruth(
            direction_label="Leave", actual_value=51.9, unit="percent",
            narrative="2016-06 英国脱欧公投 Leave 以 51.9% 胜出",
            source="UK Electoral Commission", confidence="high"),
        metadata={"marginals_source": "illustrative — replace with real 2016 data"},
    )


def _us_2016() -> BenchmarkCase:
    # Three states Trump flipped/held narrowly in 2016 → electoral roll-up.
    geo = {
        "root_level": "country", "rollup": "electoral",
        "units": [
            {"geo_id": "US", "level": "country", "name": "United States"},
            {"geo_id": "PA", "level": "state", "name": "Pennsylvania", "parent_id": "US", "electoral_weight": 20},
            {"geo_id": "MI", "level": "state", "name": "Michigan", "parent_id": "US", "electoral_weight": 16},
            {"geo_id": "WI", "level": "state", "name": "Wisconsin", "parent_id": "US", "electoral_weight": 10},
        ],
    }
    unit_marg = {s: {"age": _AGE, "education": _EDU, "gender": _GENDER}
                 for s in ("PA", "MI", "WI")}
    return BenchmarkCase(
        case_id="SV-USELEC-2016", name="US presidential election 2016 (rust-belt swing)",
        description="三个摇摆州谁赢 → 选举人票滚动",
        category=BenchmarkCategory.EVENT, difficulty=Difficulty.HARD,
        tags=["survey", "election", "us", "electoral"],
        scenario_request={
            "default_size": 800, "geo": geo, "unit_marginals": unit_marg,
            "questions": [{"qid": "vote", "type": "single_choice",
                           "text": "总统大选你会投给？", "options": ["Trump", "Clinton"]}],
            "predict": {"qid": "vote", "mode": "geo_leader"},
        },
        event_ground_truth=EventGroundTruth(
            direction_label="Trump", actual_value=None, unit=None,
            narrative="2016 Trump 拿下 PA/MI/WI，赢得选举人票入主白宫",
            source="US 2016 official results", confidence="high"),
        metadata={"marginals_source": "illustrative — replace with real state census/exit-poll"},
    )


def load_survey_suite() -> BenchmarkSuite:
    cases: List[BenchmarkCase] = [_brexit(), _us_2016()]
    return BenchmarkSuite(
        name="Survey Calibration Suite (historical events)",
        description="Population/survey backtests: real outcomes, illustrative marginals.",
        version="0.1.0", cases=cases,
        metadata={"path": "survey", "marginals": "illustrative"},
    )
