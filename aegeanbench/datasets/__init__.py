from aegeanbench.datasets.consensus_cases import load_consensus_suite
from aegeanbench.datasets.collaboration_cases import load_collaboration_suite
from aegeanbench.datasets.risk_cases import load_risk_suite
from aegeanbench.datasets.investment_cases import (
    load_investment_cases_from_file,
    load_investment_suite,
)
from aegeanbench.core.models import BenchmarkSuite, BenchmarkCategory


def load_full_suite() -> BenchmarkSuite:
    """Load all benchmark cases into a single combined suite."""
    consensus     = load_consensus_suite()
    collaboration = load_collaboration_suite()
    risk          = load_risk_suite()
    investment    = load_investment_suite()

    all_cases = consensus.cases + collaboration.cases + risk.cases + investment.cases
    return BenchmarkSuite(
        name="AegeanBench Full Suite",
        description=(
            "Complete AegeanBench: consensus, collaboration, risk "
            "assessment, and investment backtest cases for multi-agent LLM systems."
        ),
        version="0.1.0",
        cases=all_cases,
    )


__all__ = [
    "load_consensus_suite",
    "load_collaboration_suite",
    "load_risk_suite",
    "load_investment_suite",
    "load_investment_cases_from_file",
    "load_full_suite",
]

