"""
Consensus Benchmark Cases.

Tests whether the Aegean consensus protocol correctly converges,
detects outliers, and terminates early when appropriate.

Case taxonomy:
  C-EASY-*  : All agents agree from round 1 (quorum trivially reached)
  C-MED-*   : Majority agrees after 1-2 rounds of refinement
  C-HARD-*  : Outlier agent(s), noisy signals, adversarial input
  C-EDGE-*  : Edge cases (tie, single agent, max-rounds exhausted)
"""

from aegeanbench.core.models import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkSuite,
    Difficulty,
)


def load_consensus_suite() -> BenchmarkSuite:
    cases = [

        # ── EASY: all agents agree immediately ──────────────────────────

        BenchmarkCase(
            case_id="C-EASY-001",
            name="Basic Arithmetic Consensus",
            description="Trivial arithmetic that all agents should answer identically.",
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.EASY,
            tags=["arithmetic", "easy", "full-agreement"],
            task="What is 7 multiplied by 8?",
            expected_answer="56",
            answer_variants=["56", "fifty-six"],
            num_agents=3,
            max_rounds=5,
            quorum_threshold=0.5,
        ),

        BenchmarkCase(
            case_id="C-EASY-002",
            name="Capital City Consensus",
            description="Factual geography that agents should agree on immediately.",
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.EASY,
            tags=["factual", "easy", "full-agreement"],
            task="What is the capital city of France?",
            expected_answer="Paris",
            answer_variants=["Paris", "paris"],
            num_agents=3,
            max_rounds=5,
            quorum_threshold=0.5,
        ),

        BenchmarkCase(
            case_id="C-EASY-003",
            name="Boolean Consensus",
            description="Simple true/false question — tests quorum on binary answers.",
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.EASY,
            tags=["boolean", "easy"],
            task="Is the Earth flat?",
            expected_answer="No",
            answer_variants=["No", "no", "False", "false"],
            num_agents=5,
            max_rounds=3,
            quorum_threshold=0.5,
        ),

        BenchmarkCase(
            case_id="C-EASY-004",
            name="Early Stop — Trivial Case",
            description=(
                "All 5 agents answer identically. Coordinator should "
                "trigger early termination after receiving quorum responses."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.EASY,
            tags=["early-stop", "easy"],
            task="What is the chemical symbol for water?",
            expected_answer="H2O",
            answer_variants=["H2O", "h2o"],
            num_agents=5,
            max_rounds=5,
            quorum_threshold=0.5,
        ),

        BenchmarkCase(
            case_id="C-EASY-005",
            name="Weighted Voting — High-Capability Agent Wins",
            description=(
                "Two low-weight agents say 'A', one high-weight agent says 'B'. "
                "Weighted engine should select 'B' due to superior capability weight."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.EASY,
            tags=["weighted-voting", "easy", "capability"],
            task="Which sorting algorithm has O(n log n) average complexity: BubbleSort or MergeSort?",
            expected_answer="MergeSort",
            answer_variants=["MergeSort", "Merge Sort", "mergesort"],
            num_agents=3,
            max_rounds=3,
            quorum_threshold=0.5,
            agent_capability_weights=[0.9, 0.2, 0.2],  # agent-0 is the expert
        ),

        # ── MEDIUM: majority after refinement ───────────────────────────

        BenchmarkCase(
            case_id="C-MED-001",
            name="Multi-step Arithmetic Consensus",
            description=(
                "Agents must perform multi-step reasoning. One agent may "
                "initially give a wrong intermediate answer but refine correctly."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.MEDIUM,
            tags=["reasoning", "medium", "refinement"],
            task=(
                "A store sells apples for $1.20 each and oranges for $0.80 each. "
                "Alice buys 3 apples and 5 oranges. How much does she pay in total?"
            ),
            expected_answer="7.60",
            answer_variants=["7.60", "$7.60", "7.6"],
            num_agents=3,
            max_rounds=5,
            quorum_threshold=0.5,
        ),

        BenchmarkCase(
            case_id="C-MED-002",
            name="Architecture Recommendation Consensus",
            description=(
                "Agents must converge on a best-practice recommendation "
                "even when initial opinions differ."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.MEDIUM,
            tags=["opinion", "medium", "refinement"],
            task=(
                "For a Python microservice that must handle 10k concurrent requests, "
                "which architecture pattern is most appropriate: monolith, microservices, "
                "or serverless? Answer with one word."
            ),
            expected_answer="microservices",
            answer_variants=["microservices", "Microservices", "micro-services"],
            num_agents=3,
            max_rounds=5,
            quorum_threshold=0.5,
        ),

        BenchmarkCase(
            case_id="C-MED-003",
            name="Stability Horizon Test",
            description=(
                "Consensus answer is correct but must remain stable for "
                "beta=2 consecutive rounds before termination."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.MEDIUM,
            tags=["stability-horizon", "medium"],
            task=(
                "A train travels 300 km at 60 km/h and then 200 km at 100 km/h. "
                "What is the total travel time in hours?"
            ),
            expected_answer="7",
            answer_variants=["7", "7.0", "7 hours"],
            num_agents=4,
            max_rounds=5,
            quorum_threshold=0.5,
        ),

        BenchmarkCase(
            case_id="C-MED-004",
            name="Confidence-Weighted Refinement",
            description=(
                "Agents provide different confidence levels. Lower-confidence agents "
                "should defer to higher-confidence agents during refinement."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.MEDIUM,
            tags=["confidence", "weighted", "medium"],
            task=(
                "What is the time complexity of searching in a balanced BST? "
                "Answer with Big-O notation only."
            ),
            expected_answer="O(log n)",
            answer_variants=["O(log n)", "O(log N)", "O(logn)"],
            num_agents=3,
            max_rounds=4,
            quorum_threshold=0.5,
        ),

        BenchmarkCase(
            case_id="C-MED-005",
            name="5-Agent Quorum Test",
            description=(
                "5 agents, quorum=3. Tests that the system correctly requires "
                "ceil(N/2)+1 agreements before proceeding."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.MEDIUM,
            tags=["quorum", "medium", "multi-agent"],
            task=(
                "In Python, which built-in data structure provides O(1) average "
                "lookup time: list, dict, or set? Answer with one word."
            ),
            expected_answer="dict",
            answer_variants=["dict", "dictionary", "set"],
            num_agents=5,
            max_rounds=5,
            quorum_threshold=0.6,
        ),

        # ── HARD: outlier agents, adversarial, noisy ─────────────────────

        BenchmarkCase(
            case_id="C-HARD-001",
            name="Outlier Agent Detection",
            description=(
                "One agent (agent-2) deliberately gives a wrong answer. "
                "The system must detect the outlier and still converge to the correct answer."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.HARD,
            tags=["outlier", "hard", "adversarial"],
            task="What is the square root of 144?",
            expected_answer="12",
            answer_variants=["12", "12.0", "twelve"],
            num_agents=5,
            max_rounds=5,
            quorum_threshold=0.6,
            has_outlier_agent=True,
            outlier_agent_idx=2,
        ),

        BenchmarkCase(
            case_id="C-HARD-002",
            name="Two Outlier Agents",
            description=(
                "Two of five agents give incorrect answers. System must still "
                "reach correct consensus with the 3-agent majority."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.HARD,
            tags=["outlier", "hard", "byzantine"],
            task="In the TCP/IP model, which layer is responsible for end-to-end communication? Answer with layer name only.",
            expected_answer="Transport",
            answer_variants=["Transport", "transport", "Transport Layer"],
            num_agents=5,
            max_rounds=5,
            quorum_threshold=0.6,
            has_outlier_agent=True,
            outlier_agent_idx=0,
            metadata={"num_outliers": 2, "outlier_indices": [0, 4]},
        ),

        BenchmarkCase(
            case_id="C-HARD-003",
            name="Noisy Responses — Paraphrase Consensus",
            description=(
                "Agents answer with semantically equivalent but lexically "
                "different strings. The system must normalize and detect consensus."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.HARD,
            tags=["noise", "hard", "normalization"],
            task="What does REST stand for in REST API?",
            expected_answer="Representational State Transfer",
            answer_variants=[
                "Representational State Transfer",
                "representational state transfer",
                "Representational-State-Transfer",
            ],
            num_agents=3,
            max_rounds=5,
            quorum_threshold=0.5,
            inject_noise=True,
        ),

        BenchmarkCase(
            case_id="C-HARD-004",
            name="Max Rounds Exhaustion",
            description=(
                "Agents are configured to diverge. The system must gracefully "
                "handle reaching max_rounds without consensus."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.HARD,
            tags=["max-rounds", "hard", "no-consensus"],
            task=(
                "Is object-oriented or functional programming better for large codebases? "
                "Answer with exactly one word: 'OOP' or 'FP'."
            ),
            expected_answer=None,  # No ground truth — measures graceful degradation
            num_agents=4,
            max_rounds=3,
            quorum_threshold=0.7,  # Very high threshold — hard to reach
            metadata={"expect_consensus": False},
        ),

        BenchmarkCase(
            case_id="C-HARD-005",
            name="Weighted Outlier Override",
            description=(
                "A low-weight majority (3 agents, weight=0.2) answers incorrectly. "
                "A high-weight minority (2 agents, weight=0.9) answers correctly. "
                "Weighted engine must select the minority's answer."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.HARD,
            tags=["weighted", "hard", "outlier", "capability"],
            task="What does CAP theorem stand for? Answer: Consistency, Availability, ___",
            expected_answer="Partition Tolerance",
            answer_variants=["Partition Tolerance", "Partition tolerance", "Partition"],
            num_agents=5,
            max_rounds=5,
            quorum_threshold=0.5,
            agent_capability_weights=[0.9, 0.9, 0.2, 0.2, 0.2],
            has_outlier_agent=True,
            metadata={"expert_agents": [0, 1], "non_expert_agents": [2, 3, 4]},
        ),

        BenchmarkCase(
            case_id="C-HARD-006",
            name="Consensus Confidence Under Uncertainty",
            description=(
                "All agents agree on the answer but report very low confidence. "
                "System must still reach consensus but flag low confidence."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.HARD,
            tags=["confidence", "hard", "uncertainty"],
            task=(
                "In 2031, which programming language will be most popular? "
                "Answer with one word."
            ),
            expected_answer=None,  # Future prediction — no ground truth
            num_agents=3,
            max_rounds=5,
            quorum_threshold=0.5,
            metadata={"expect_low_confidence": True},
        ),

        # ── EDGE: boundary conditions ────────────────────────────────────

        BenchmarkCase(
            case_id="C-EDGE-001",
            name="Single Agent Consensus",
            description="Only 1 agent — consensus should immediately resolve to that agent's answer.",
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.EASY,
            tags=["edge", "single-agent"],
            task="What is 2 + 2?",
            expected_answer="4",
            answer_variants=["4", "four"],
            num_agents=1,
            max_rounds=5,
            quorum_threshold=0.5,
        ),

        BenchmarkCase(
            case_id="C-EDGE-002",
            name="Perfect Tie — Two Agents, Two Answers",
            description=(
                "Two agents give different answers. No majority possible. "
                "System must handle gracefully without crashing."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.HARD,
            tags=["edge", "tie", "two-agents"],
            task="Is Python or JavaScript more popular for backend development? One word.",
            expected_answer=None,
            num_agents=2,
            max_rounds=3,
            quorum_threshold=0.5,
            metadata={"expect_consensus": False},
        ),

        BenchmarkCase(
            case_id="C-EDGE-003",
            name="High-Frequency Early Stop",
            description=(
                "7 agents, quorum=4. After 4 agents respond with same answer, "
                "remaining 3 should be cancelled immediately."
            ),
            category=BenchmarkCategory.CONSENSUS,
            difficulty=Difficulty.MEDIUM,
            tags=["early-stop", "edge", "large-group"],
            task="What is the default port for HTTPS?",
            expected_answer="443",
            answer_variants=["443", "port 443"],
            num_agents=7,
            max_rounds=5,
            quorum_threshold=0.5,
        ),

    ]

    return BenchmarkSuite(
        name="AegeanBench Consensus Suite",
        description=(
            "Evaluates multi-agent consensus convergence, quorum detection, "
            "early termination, outlier identification, and weighted voting."
        ),
        version="0.1.0",
        cases=cases,
        metadata={"total_cases": len(cases), "category": "consensus"},
    )
