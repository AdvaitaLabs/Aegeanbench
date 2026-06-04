# AegeanBench · Sports Module

World Cup 2026 prediction + betting + benchmark pipeline.

## What it does

For every World Cup fixture, run a panel of predictors (classical models,
single LLMs, multi-agent Aegean consensus) over the same input, build a
value-bet portfolio from the strongest model, persist everything to disk,
and serve six JSON endpoints to the product front-end.

```
   data sources ──► gateway ──► predictors ──► betting ──► persistence ──► reporter
   (mock + free + paid)        (Elo / DC / LLM / Aegean)    (run JSON)     (6 endpoints)
```

## Quick start

```bash
# 1. End-to-end demo (offline, mock data, no keys required)
python -m aegeanbench.sports.demo

# 2. Run pipeline with custom predictors
python -m aegeanbench.sports.cli predict \
    --runners=elo,dixon-coles,gpt-5,claude,aegean \
    --label=my-run

# 3. List persisted runs
python -m aegeanbench.sports.cli list-runs

# 4. Serve the live API
uvicorn aegeanbench.sports.reporter.server:app --port 8200
```

## Module layout

```
aegeanbench/sports/
├── models.py            Team / Player / Odds / Match / Prediction dataclasses
├── cache.py             Local JSON cache with TTL
├── gateway.py           SportsDataGateway facade
├── prompts.py           Unified prompt builder (same input for every LLM)
├── chat_mock.py         Sprint-mode crowd chat generator
├── cli.py               Command-line interface
├── demo.py              End-to-end one-command demo
├── sources/             Data adapters (football-data, soccersapi, fbref, clubelo, kaggle)
├── predictors/          Elo / Dixon-Coles / Monte Carlo / LLM / Aegean
├── betting/             Kelly + value bet + portfolio sizing
├── orchestrator/        WorldCupPipeline + evaluator + persistence
└── reporter/            6 product JSON endpoints + optional FastAPI server
```

## Predictors

| ID                  | Type        | Recommended LLM     | Cost / match |
|---------------------|-------------|---------------------|-------------:|
| `elo`               | classical   | n/a                 | $0.000 |
| `dixon_coles`       | classical   | n/a                 | $0.000 |
| `gpt-5`             | single LLM  | OpenAI GPT-5        | $0.04 |
| `claude-opus-4-7`   | single LLM  | Anthropic Claude    | $0.06 |
| `deepseek-v3`       | single LLM  | DeepSeek V3         | $0.001 |
| `aegean`            | consensus   | 6 agents, mixed     | $0.11 |
| `random`            | baseline    | n/a                 | $0.000 |

Each predictor implements the same interface: `predict(MatchContext) -> Prediction`.

## Data sources

| Source           | Type   | Free? | Status     | Used for |
|------------------|--------|-------|------------|----------|
| `clubelo.com`    | live   | ✅    | integrated | Elo seed ratings |
| `football-data.org` (free tier) | live | ✅ | integrated | World Cup fixtures, teams |
| Kaggle CSV       | offline| ✅    | integrated | Historical training data |
| `football-data.org` (TIER ONE) | live | 💰 | upgrade by setting key | Full fixtures + history |
| `soccersapi.com` | live   | 💰    | mock       | Odds, lineups, h2h |
| `fbref.com`      | scrape | ✅    | integrated (rate-limited, opt-in) | xG / PPDA / possession |

To enable live FBref scraping set `FBREF_LIVE=1`. Default is mock.

## Endpoints

The six contracts the product front-end consumes:

| Endpoint | Method | Body | Purpose |
|---|---|---|---|
| `/api/v1/runners` | GET | - | Predictor catalog |
| `/api/v1/tournaments` | GET | - | Available tournaments |
| `/api/v1/leaderboard` | GET | - | Ranked across all evaluated runs |
| `/api/v1/runs/{id}` | GET | - | Full run bundle |
| `/api/v1/runs/{id}/matches/{mid}` | GET | - | Match drill-down per predictor |
| `/api/v1/runners/{rid}/card` | GET | - | Per-predictor profile |
| `/api/v1/agents/{id}/answer` | POST | `{question, ...}` | V2 @-mention single-agent Q&A |
| `/ws/predictions` | WS | - | V2 live prediction push |
| `/ws/matches/{id}` | WS | - | V2 per-match live channel |
| `/api/v1/health` | GET | - | Health check |

## Persistence layout

```
~/.aegeanbench/worldcup_runs/run_<ts>_<hash>/
    manifest.json       run metadata + config
    predictions.json    all predictions across runners and matches
    portfolio.json      sized bets + risk caps applied
    evaluation.json     Brier / hit rate / ROI (after ground truth)
```

Read these directly with `cat` / `jq`, or via `persistence.load_run()`.

## Running the demo

```bash
$ python -m aegeanbench.sports.demo

STEP 1: Gateway + free data sources
  Loaded 8 teams; Elo seeded from clubelo fallback table.
    BRA (Brazil): Elo=1981
    ARG (Argentina): Elo=2114
    ...

STEP 3: Running predictions
  Predictors run: 6
    elo            match=WC2026-A1  H=0.458 D=0.271 A=0.271
    dixon_coles    match=WC2026-A1  H=0.470 D=0.267 A=0.263
    ...

STEP 4: Betting portfolio
  8 bets / 8 matches | stake $260.33 (26.0% exposure) | E[return] $51.29

STEP 5: Injecting ground truth + evaluating
  Predictor rankings (by mean Brier, lower = better):
    gpt-5            brier=0.5939  hit=50.00%
    claude-opus-4-7  brier=0.6029  hit=62.50%
    ...
  Betting realised: 4 won / 4 lost | net $+59.88  ROI +23.0%

STEP 6: Building product JSON endpoints
  Output: /tmp/aegean_demo_api
    runners: 1   tournaments: 1   leaderboard: 1
    runs: 2      match_details: 16   runner_cards: 7
```

## Tests

```bash
PYTHONPATH=. pytest tests/sports/ -q
# 157 passed
```

## Going from mock to real

See `docs/API_KEYS.md` (top-level) for the exact env vars and instructions.

Short version:
```bash
# Free immediately
export AEGEANBENCH_FOOTBALL_DATA_KEY=<your free football-data key>
export FBREF_LIVE=1

# Paid (once finance procures)
export AEGEANBENCH_SOCCERSAPI_KEY=<your soccersapi key>
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export DEEPSEEK_API_KEY=...
```

After setting, re-run the same CLI / demo and adapters switch to live
data automatically (no code changes).
