# AegeanBench

**The first open benchmark for Multi-Agent Consensus systems.**

Built on top of [aegean-consensus](../aegean-consensus), AegeanBench evaluates three capabilities that existing benchmarks (AutoGenBench, MultiAgentBench, REALM-Bench) do not cover:

| Capability | Existing benchmarks | AegeanBench |
|---|---|---|
| Multi-agent task completion | ✅ | ✅ |
| Collaboration quality | ⚠️ partial | ✅ |
| **Consensus convergence** | ❌ | ✅ |
| **Early-stop / quorum efficiency** | ❌ | ✅ |
| **Outlier agent detection** | ❌ | ✅ |
| **VAN risk pipeline accuracy** | ❌ | ✅ |

---

## Quick Start

```bash
# Install
cd AegeanBench
pip install -e .

# ── Three ways to run benchmarks ──

# 1. Mock Runner (fastest, no LLM needed, no dependencies)
python -m aegeanbench.cli run --runner mock

# 2. Aegean Runner (Python import, requires: pip install -e ../aegean-consensus)
python -m aegeanbench.cli run --runner aegean

# 3. HTTP Runner (true end-to-end, requires aegean-consensus service running)
#    First, start the service in another terminal:
#    cd ../aegean-consensus && uvicorn aegean.api.app:create_app --factory --host 0.0.0.0 --port 8000
#    Then run:
python -m aegeanbench.cli run --runner http

# ── Filtering ──

# Run only consensus cases
python -m aegeanbench.cli run --runner mock --category consensus

# Run only hard cases
python -m aegeanbench.cli run --runner mock --difficulty hard

# Run a single case
python -m aegeanbench.cli run --runner mock --case-id C-HARD-001

# ── Utilities ──

# List all available cases
python -m aegeanbench.cli list

# Show details of a case
python -m aegeanbench.cli show C-HARD-001
```

---

## Benchmark Categories

### 1. Consensus Benchmark (`consensus`)

Tests whether the agent committee correctly converges, detects outliers, and terminates early.

**Novel metrics:**

| Metric | Description |
|---|---|
| `consensus_rate` | Fraction of cases where consensus was reached |
| `consensus_confidence` | Mean confidence of the winning answer |
| `disagreement_score` | Fraction of agents NOT in the majority (0 = perfect agreement) |
| `quorum_efficiency` | `1 - rounds_used/max_rounds` — higher means faster convergence |
| `early_stop_rate` | Fraction of cases where early termination fired |
| `outlier_detection_rate` | Fraction of outlier cases where majority correctly overrode |

**Case taxonomy:**

```
C-EASY-*   All agents agree from round 1
C-MED-*    Majority after 1-2 rounds of refinement
C-HARD-*   Outlier agents, high quorum threshold, adversarial
C-EDGE-*   Boundary conditions (single agent, perfect tie, max-rounds)
```

### 2. Collaboration Benchmark (`collaboration`)

Tests whether agents effectively divide and conquer complex tasks.

**Novel metrics:**

| Metric | Description |
|---|---|
| `subtask_completion_rate` | Fraction of subtasks completed |
| `milestone_hit_rate` | Fraction of defined milestones achieved |
| `coordination_score` | Quality of agent coordination (0–1) |
| `redundancy_rate` | Fraction of duplicated work across agents |

**Case taxonomy:**

```
L-EASY-*  Linear pipeline (sequential agents)
L-MED-*   Parallel subtasks with merge step
L-HARD-*  Complex dependency graph, multi-domain synthesis
L-HYB-*   Hybrid: consensus decision + collaborative implementation
```

### 3. Risk Assessment Benchmark (`risk`)

Tests the Aegean VAN (Verification Agent Network) pipeline.

**Novel metrics:**

| Metric | Description |
|---|---|
| `risk_accuracy` | Decision correctness (approve/reject/challenge/review) |
| `risk_level_accuracy` | Risk level classification (low/medium/high/critical) |
| `risk_f1_approve` | F1 for the APPROVE class |
| `risk_f1_reject` | F1 for the REJECT class |
| `pre_screen_rate` | Fraction resolved by deterministic pre-screen (no LLM) |
| `validator_agreement` | Mean fraction of validators aligned with final decision |

**Case taxonomy:**

```
R-LOW-*    Should be APPROVED  (low risk)
R-MED-*    Should be REVIEWED  (medium risk)
R-HIGH-*   Should be REJECTED  (high risk)
R-CRIT-*   REJECTED instantly via pre-screen (sanctions, injection, limits)
R-CHAL-*   Should be CHALLENGED (uncertain, needs evidence)
R-SEQ-*    Tests sequencer routing accuracy
```

---

## Architecture

```
AegeanBench/
├── aegeanbench/
│   ├── core/
│   │   └── models.py          # BenchmarkCase, BenchmarkResult, all metrics models
│   ├── datasets/
│   │   ├── consensus_cases.py  # 16 consensus test cases
│   │   ├── collaboration_cases.py # 11 collaboration test cases
│   │   └── risk_cases.py       # 17 risk assessment test cases
│   ├── metrics/
│   │   └── engine.py          # MetricsEngine — computes all metrics
│   ├── runners/
│   │   ├── mock_runner.py     # Deterministic mock (no LLM needed)
│   │   ├── aegean_runner.py   # Real aegean-consensus integration
│   │   └── _mock_llm_agent.py # Pluggable agent for AegeanRunner
│   └── cli.py                 # Command-line interface
├── configs/
│   └── default.yaml           # Default configuration
├── tests/
│   └── test_benchmark.py      # Full test suite
└── pyproject.toml
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Expected output (mock runner, all cases):

```
tests/test_benchmark.py::TestDatasetLoading::test_consensus_suite_loads       PASSED
tests/test_benchmark.py::TestDatasetLoading::test_collaboration_suite_loads   PASSED
tests/test_benchmark.py::TestDatasetLoading::test_risk_suite_loads            PASSED
tests/test_benchmark.py::TestDatasetLoading::test_full_suite_combines_all     PASSED
tests/test_benchmark.py::TestMockRunnerConsensus::test_easy_case_passes       PASSED
tests/test_benchmark.py::TestMockRunnerConsensus::test_outlier_case_still_converges PASSED
...
```

---

## Three Runner Modes Explained

AegeanBench supports three ways to run benchmarks, each with different tradeoffs:

### 1. MockRunner — Fastest, No Dependencies

```bash
python -m aegeanbench.cli run --runner mock
```

| Aspect | Details |
|---|---|
| **How it works** | Simulates agent behavior deterministically (no LLM calls) |
| **Speed** | ~50ms per case (milliseconds) |
| **Dependencies** | Only pydantic |
| **Use case** | Validate benchmark framework, CI/CD, quick iteration |
| **Accuracy** | Mock oracle — always returns correct answer (unrealistic) |
| **Token data** | Estimated via formula (not real) |

**Best for:** Testing the benchmark system itself, not the actual aegean-consensus performance.

---

### 2. AegeanRunner — Python Import (Direct Integration)

```bash
pip install -e ../aegean-consensus
python -m aegeanbench.cli run --runner aegean
```

| Aspect | Details |
|---|---|
| **How it works** | Imports aegean-consensus as Python package, calls classes directly |
| **Speed** | Depends on LLM latency (typically 1-10s per case) |
| **Dependencies** | `pip install -e ../aegean-consensus` |
| **Use case** | Development, testing with real LLM, CI/CD with LLM |
| **Accuracy** | Real system behavior (with real LLM) |
| **Token data** | Real from `ConsensusResult.tokens_used` |

**Best for:** Development and testing when you want to iterate quickly on the aegean-consensus code itself.

---

### 3. HttpRunner — REST API (True End-to-End)

```bash
# Terminal 1: Start aegean-consensus service
cd ../aegean-consensus
uvicorn aegean.api.app:create_app --factory --host 0.0.0.0 --port 8000

# Terminal 2: Run benchmark
python -m aegeanbench.cli run --runner http
```

| Aspect | Details |
|---|---|
| **How it works** | Calls aegean-consensus via HTTP REST API |
| **Speed** | Depends on LLM latency + network (typically 1-10s per case) |
| **Dependencies** | aegean-consensus service must be running |
| **Use case** | Production testing, deployment validation, true E2E |
| **Accuracy** | Real system behavior (with real LLM) |
| **Token data** | Real from API response |

**Best for:** Production validation, testing the system as it would be deployed, integration testing.

---

### Comparison Table

| Feature | Mock | Aegean | HTTP |
|---|---|---|---|
| Speed | ⚡⚡⚡ Fast | ⚡ Slow | ⚡ Slow |
| Real LLM | ❌ No | ✅ Yes | ✅ Yes |
| Real tokens | ❌ Estimated | ✅ Real | ✅ Real |
| Dependencies | Minimal | aegean-consensus | aegean-consensus service |
| E2E test | ❌ No | ⚠️ Partial | ✅ Yes |
| CI/CD friendly | ✅ Yes | ⚠️ Maybe | ❌ Requires service |
| Development | ✅ Best | ✅ Good | ⚠️ Extra setup |

---

### Recommended Workflow

```
Development:
  MockRunner (fast iteration)
    ↓
  AegeanRunner (test with real LLM)
    ↓
Production/Deployment:
  HttpRunner (true E2E test)
```

---

```bash
# Install aegean-consensus first
pip install -e ../aegean-consensus

# Run with real framework
python -m aegeanbench.cli run --runner aegean
```

Or in Python:

```python
from aegeanbench.datasets import load_full_suite
from aegeanbench.runners.aegean_runner import AegeanRunner
from aegeanbench.metrics.engine import MetricsEngine

runner = AegeanRunner(llm_client=your_openai_client)
suite  = load_full_suite()
results = runner.run_suite(suite.cases)

engine = MetricsEngine()
sr     = engine.aggregate(suite.suite_id, suite.name, results)

print(f"Accuracy:          {sr.accuracy:.1%}")
print(f"Consensus rate:    {sr.consensus_rate:.1%}")
print(f"Early stop rate:   {sr.early_stop_rate:.1%}")
print(f"Outlier detection: {sr.outlier_detection_rate:.1%}")
print(f"Risk accuracy:     {sr.risk_accuracy:.1%}")
print(f"Pre-screen rate:   {sr.pre_screen_rate:.1%}")
```

---

## Adding Custom Cases

```python
from aegeanbench.core.models import BenchmarkCase, BenchmarkCategory, Difficulty

my_case = BenchmarkCase(
    case_id="MY-001",
    name="My Custom Consensus Case",
    description="Tests consensus on a domain-specific question.",
    category=BenchmarkCategory.CONSENSUS,
    difficulty=Difficulty.MEDIUM,
    tags=["custom", "domain"],
    task="What is the recommended database for a write-heavy workload?",
    expected_answer="Cassandra",
    answer_variants=["Cassandra", "cassandra", "Apache Cassandra"],
    num_agents=5,
    max_rounds=5,
    quorum_threshold=0.6,
)
```

---

## Benchmark Results Format

Results are saved as JSON in `results/run_<timestamp>.json`:

```json
{
  "suite_id": "suite-abc123",
  "suite_name": "AegeanBench Full Suite",
  "total_cases": 44,
  "passed": 40,
  "accuracy": 0.909,
  "consensus_rate": 0.875,
  "mean_disagreement": 0.142,
  "early_stop_rate": 0.625,
  "outlier_detection_rate": 1.0,
  "risk_accuracy": 0.941,
  "pre_screen_rate": 0.25,
  "p50_latency_s": 0.051,
  "p95_latency_s": 0.078
}
```

---

## Citation

If you use AegeanBench in your research:

```bibtex
@software{aegeanbench2026,
  title   = {AegeanBench: The First Open Benchmark for Multi-Agent Consensus},
  year    = {2026},
  url     = {https://github.com/your-org/AegeanBench},
  note    = {Built on the Aegean consensus protocol (arXiv:2512.20184)}
}
```

---

## License

MIT

