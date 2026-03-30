from aegeanbench.datasets.consensus_cases import load_consensus_suite
from aegeanbench.datasets.collaboration_cases import load_collaboration_suite
from aegeanbench.datasets.risk_cases import load_risk_suite
from aegeanbench.core.models import BenchmarkSuite, BenchmarkCategory


def load_full_suite() -> BenchmarkSuite:
    """Load all benchmark cases into a single combined suite."""
    consensus     = load_consensus_suite()
    collaboration = load_collaboration_suite()
    risk          = load_risk_suite()

    all_cases = consensus.cases + collaboration.cases + risk.cases
    return BenchmarkSuite(
        name="AegeanBench Full Suite",
        description=(
            "Complete AegeanBench: consensus, collaboration, and risk "
            "assessment cases for multi-agent LLM systems."
        ),
        version="0.1.0",
        cases=all_cases,
    )


__all__ = [
    "load_consensus_suite",
    "load_collaboration_suite",
    "load_risk_suite",
    "load_full_suite",
]

