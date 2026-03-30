"""
Collaboration Benchmark Cases.

Tests whether agents effectively divide and conquer complex tasks,
coordinate without redundancy, and hit defined milestones.

Case taxonomy:
  L-EASY-*  : Linear pipeline — agents work in fixed sequence
  L-MED-*   : Parallel subtasks with merge step
  L-HARD-*  : Complex dependency graph, dynamic routing
  L-HYB-*   : Hybrid mode — mix of consensus + collaboration
"""

from aegeanbench.core.models import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkSuite,
    Difficulty,
)


def load_collaboration_suite() -> BenchmarkSuite:
    cases = [

        # ── EASY: linear pipeline ───────────────────────────────────────

        BenchmarkCase(
            case_id="L-EASY-001",
            name="Report Writing Pipeline",
            description=(
                "Three agents work sequentially: researcher gathers facts, "
                "writer drafts the report, editor reviews and finalises."
            ),
            category=BenchmarkCategory.COLLABORATION,
            difficulty=Difficulty.EASY,
            tags=["pipeline", "easy", "writing"],
            task="Produce a 3-paragraph report on the benefits of renewable energy.",
            num_agents=3,
            subtasks=[
                "Gather 5 key facts about renewable energy benefits",
                "Write a 3-paragraph report using the gathered facts",
                "Review and edit the report for clarity and grammar",
            ],
            milestones=[
                "facts_gathered",
                "draft_written",
                "report_finalised",
            ],
            max_rounds=3,
        ),

        BenchmarkCase(
            case_id="L-EASY-002",
            name="Code Review Pipeline",
            description=(
                "Developer agent writes code, reviewer agent checks it, "
                "tester agent writes unit tests."
            ),
            category=BenchmarkCategory.COLLABORATION,
            difficulty=Difficulty.EASY,
            tags=["pipeline", "easy", "coding"],
            task="Implement a Python function that checks if a string is a palindrome, then review and test it.",
            num_agents=3,
            subtasks=[
                "Write a Python function `is_palindrome(s: str) -> bool`",
                "Review the function for correctness and edge cases",
                "Write 5 unit tests covering normal cases, empty string, and single character",
            ],
            milestones=[
                "function_implemented",
                "code_reviewed",
                "tests_written",
            ],
            max_rounds=3,
        ),

        BenchmarkCase(
            case_id="L-EASY-003",
            name="Data Analysis Pipeline",
            description=(
                "Analyst agent computes stats, visualiser agent describes the chart, "
                "presenter agent writes the summary."
            ),
            category=BenchmarkCategory.COLLABORATION,
            difficulty=Difficulty.EASY,
            tags=["pipeline", "easy", "data"],
            task=(
                "Given sales data [120, 135, 98, 167, 145, 110, 190]: "
                "compute statistics, describe a bar chart, and write an executive summary."
            ),
            num_agents=3,
            subtasks=[
                "Compute mean, median, min, max, and standard deviation",
                "Describe what a bar chart of this data would look like",
                "Write a 2-sentence executive summary of the sales trend",
            ],
            milestones=[
                "statistics_computed",
                "chart_described",
                "summary_written",
            ],
            max_rounds=3,
        ),

        # ── MEDIUM: parallel subtasks with merge ─────────────────────────

        BenchmarkCase(
            case_id="L-MED-001",
            name="Parallel Market Research",
            description=(
                "Four agents simultaneously research different market segments, "
                "then a coordinator merges findings into a single report."
            ),
            category=BenchmarkCategory.COLLABORATION,
            difficulty=Difficulty.MEDIUM,
            tags=["parallel", "medium", "research", "merge"],
            task="Conduct market research on the global AI software market across 4 segments, then produce a unified summary.",
            num_agents=4,
            subtasks=[
                "Research the enterprise AI software segment (market size, key players)",
                "Research the consumer AI applications segment (market size, key players)",
                "Research the AI infrastructure segment (cloud AI, GPUs)",
                "Merge all three research findings into a 1-page market overview",
            ],
            milestones=[
                "segment_1_researched",
                "segment_2_researched",
                "segment_3_researched",
                "report_merged",
            ],
            max_rounds=4,
        ),

        BenchmarkCase(
            case_id="L-MED-002",
            name="Software Design — Parallel Modules",
            description=(
                "Agents design independent modules of a system in parallel, "
                "then an architect reviews integration points."
            ),
            category=BenchmarkCategory.COLLABORATION,
            difficulty=Difficulty.MEDIUM,
            tags=["parallel", "medium", "software-design"],
            task="Design a ride-sharing application: auth module, matching module, payment module, and integration review.",
            num_agents=4,
            subtasks=[
                "Design the authentication and user management module (endpoints, data model)",
                "Design the driver-rider matching algorithm module",
                "Design the payment processing and billing module",
                "Review integration points and identify potential conflicts between modules",
            ],
            milestones=[
                "auth_module_designed",
                "matching_module_designed",
                "payment_module_designed",
                "integration_reviewed",
            ],
            max_rounds=5,
        ),

        BenchmarkCase(
            case_id="L-MED-003",
            name="Multilingual Translation Pipeline",
            description=(
                "Three translation agents work on different language pairs in parallel, "
                "then a quality-check agent validates all translations."
            ),
            category=BenchmarkCategory.COLLABORATION,
            difficulty=Difficulty.MEDIUM,
            tags=["parallel", "medium", "translation"],
            task="Translate 'The quick brown fox jumps over the lazy dog' into French, Spanish, and German. Then verify all translations.",
            num_agents=4,
            subtasks=[
                "Translate the sentence to French",
                "Translate the sentence to Spanish",
                "Translate the sentence to German",
                "Verify all three translations for accuracy",
            ],
            milestones=[
                "french_translated",
                "spanish_translated",
                "german_translated",
                "translations_verified",
            ],
            max_rounds=4,
        ),

        BenchmarkCase(
            case_id="L-MED-004",
            name="Coordination Efficiency Test",
            description=(
                "Measures how efficiently agents divide work without overlap. "
                "Redundancy rate should be < 20%."
            ),
            category=BenchmarkCategory.COLLABORATION,
            difficulty=Difficulty.MEDIUM,
            tags=["coordination", "medium", "efficiency"],
            task="List 15 unique Python libraries for data science, divided among 3 agents (5 each, no duplicates).",
            num_agents=3,
            subtasks=[
                "List 5 Python data science libraries for data manipulation (no ML)",
                "List 5 Python data science libraries for machine learning",
                "List 5 Python data science libraries for data visualisation",
            ],
            milestones=[
                "data_libs_listed",
                "ml_libs_listed",
                "viz_libs_listed",
            ],
            max_rounds=3,
            metadata={"max_allowed_redundancy": 0.2},
        ),

        # ── HARD: dependency graphs, dynamic routing ─────────────────────

        BenchmarkCase(
            case_id="L-HARD-001",
            name="Full Software Development Lifecycle",
            description=(
                "Agents simulate a full SDLC: requirements, design, implementation, "
                "testing, and deployment planning. Complex task dependencies."
            ),
            category=BenchmarkCategory.COLLABORATION,
            difficulty=Difficulty.HARD,
            tags=["hard", "sdlc", "dependency-graph"],
            task="Develop a plan for building a URL shortener service (like bit.ly).",
            num_agents=5,
            subtasks=[
                "Write functional and non-functional requirements",
                "Design the system architecture (components, database schema, API endpoints)",
                "Write pseudocode for the core URL shortening and redirect logic",
                "Define a test plan (unit, integration, load testing)",
                "Write a deployment checklist for production release",
            ],
            milestones=[
                "requirements_defined",
                "architecture_designed",
                "core_logic_written",
                "test_plan_defined",
                "deployment_planned",
            ],
            max_rounds=5,
        ),

        BenchmarkCase(
            case_id="L-HARD-002",
            name="Multi-Domain Research Synthesis",
            description=(
                "Agents research different domains and must synthesise findings "
                "that have cross-domain dependencies."
            ),
            category=BenchmarkCategory.COLLABORATION,
            difficulty=Difficulty.HARD,
            tags=["hard", "research", "synthesis"],
            task="Analyse the impact of AI on healthcare: research clinical AI, regulatory landscape, economic impact, and ethical concerns. Synthesise into recommendations.",
            num_agents=5,
            subtasks=[
                "Research clinical AI applications (diagnosis, treatment planning)",
                "Research AI healthcare regulations (FDA, EU AI Act)",
                "Analyse economic impact of AI on healthcare costs and employment",
                "Identify top 5 ethical concerns with AI in healthcare",
                "Synthesise all findings into 3 actionable policy recommendations",
            ],
            milestones=[
                "clinical_ai_researched",
                "regulations_researched",
                "economic_impact_analysed",
                "ethics_identified",
                "recommendations_synthesised",
            ],
            max_rounds=5,
        ),

        BenchmarkCase(
            case_id="L-HARD-003",
            name="Redundancy Stress Test",
            description=(
                "Agents are given overlapping task descriptions. System must detect "
                "and eliminate redundant work. Redundancy rate should stay < 30%."
            ),
            category=BenchmarkCategory.COLLABORATION,
            difficulty=Difficulty.HARD,
            tags=["hard", "redundancy", "coordination"],
            task="Each agent independently lists top Python web frameworks. System must deduplicate and produce a single ranked list.",
            num_agents=4,
            subtasks=[
                "List top 5 Python web frameworks with descriptions",
                "List top 5 Python web frameworks ranked by popularity",
                "List top 5 Python web frameworks for REST APIs",
                "Merge and deduplicate all framework lists into a single ranked list",
            ],
            milestones=[
                "agent_0_listed",
                "agent_1_listed",
                "agent_2_listed",
                "deduplicated_list_produced",
            ],
            max_rounds=4,
            metadata={"max_allowed_redundancy": 0.3},
        ),

        # ── HYBRID: consensus + collaboration ────────────────────────────

        BenchmarkCase(
            case_id="L-HYB-001",
            name="Hybrid: Consensus Decision + Collaborative Implementation",
            description=(
                "First, all agents reach consensus on which approach to use. "
                "Then they collaboratively implement it."
            ),
            category=BenchmarkCategory.HYBRID,
            difficulty=Difficulty.MEDIUM,
            tags=["hybrid", "medium", "consensus+collaboration"],
            task=(
                "First, reach consensus on whether to use REST or GraphQL for a new API. "
                "Then collaboratively: design schema, write sample queries, and document endpoints."
            ),
            num_agents=3,
            subtasks=[
                "[CONSENSUS] Vote: REST or GraphQL for a social media API?",
                "[COLLABORATION] Design the chosen API schema",
                "[COLLABORATION] Write 3 sample queries/requests",
                "[COLLABORATION] Write API documentation outline",
            ],
            milestones=[
                "approach_decided",
                "schema_designed",
                "samples_written",
                "docs_outlined",
            ],
            max_rounds=5,
            quorum_threshold=0.5,
        ),

        BenchmarkCase(
            case_id="L-HYB-002",
            name="Hybrid: Risk Consensus + Remediation Plan",
            description=(
                "Agents first reach consensus on the severity of a security finding, "
                "then collaboratively build a remediation plan."
            ),
            category=BenchmarkCategory.HYBRID,
            difficulty=Difficulty.HARD,
            tags=["hybrid", "hard", "security", "risk"],
            task=(
                "A SQL injection vulnerability was found in a production login endpoint. "
                "First reach consensus on severity (Critical/High/Medium). "
                "Then collaboratively produce: immediate fix, long-term hardening, and incident report."
            ),
            num_agents=4,
            subtasks=[
                "[CONSENSUS] Rate severity: Critical, High, or Medium?",
                "[COLLABORATION] Write an immediate hotfix patch description",
                "[COLLABORATION] Write a long-term security hardening plan",
                "[COLLABORATION] Draft an incident report summary",
            ],
            milestones=[
                "severity_agreed",
                "hotfix_described",
                "hardening_planned",
                "incident_reported",
            ],
            max_rounds=5,
            quorum_threshold=0.6,
            expected_answer="Critical",
        ),

    ]

    return BenchmarkSuite(
        name="AegeanBench Collaboration Suite",
        description=(
            "Evaluates multi-agent task division, coordination efficiency, "
            "milestone achievement, and hybrid consensus+collaboration workflows."
        ),
        version="0.1.0",
        cases=cases,
        metadata={"total_cases": len(cases), "category": "collaboration"},
    )
