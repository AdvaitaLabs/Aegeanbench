# API Keys Configuration

All keys are read from environment variables at startup. Setting / unsetting
an env var is the only switch between mock and live for that source.

## TL;DR

```bash
# Minimal launch (free sources only - works today)
export AEGEANBENCH_FOOTBALL_DATA_KEY=<free key from football-data.org>
export FBREF_LIVE=1

# Full launch (after finance procures paid keys)
export AEGEANBENCH_SOCCERSAPI_KEY=...
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export DEEPSEEK_API_KEY=...

# Optional (multi-agent consensus over HTTP, otherwise mock-aegean)
export AEGEAN_CONSENSUS_URL=http://localhost:8000

# Optional (chat service polling)
export CHAT_SERVICE_URL=http://chat-service:9100
```

## Key inventory

| Env Var                            | Purpose                            | Cost / month  | Status |
|-----------------------------------|------------------------------------|---------------|--------|
| `AEGEANBENCH_FOOTBALL_DATA_KEY`   | Fixtures + teams (WC, EC)          | Free (10 rpm) | request at https://football-data.org/client/register |
| `FBREF_LIVE`                      | Enable FBref xG scraping (`=1` on) | Free          | Set to enable |
| `AEGEANBENCH_SOCCERSAPI_KEY`      | Odds, lineups, h2h                 | ~$50 (mid)    | sign up at https://soccersapi.com/ |
| `OPENAI_API_KEY` + `OPENAI_MODEL` | GPT-5 predictor                    | usage-based   | https://platform.openai.com/ |
| `ANTHROPIC_API_KEY`               | Claude predictor                   | usage-based   | https://console.anthropic.com/ |
| `DEEPSEEK_API_KEY`                | DeepSeek V3 predictor              | usage-based   | https://platform.deepseek.com/ |
| `AEGEAN_CONSENSUS_URL`            | aegean-consensus HTTP endpoint     | (self-host)   | local: `http://localhost:8000` |
| `CHAT_SERVICE_URL`                | Crowd chat fetcher                 | (colleague)   | sprint default localhost:9100 |

## Default behavior without any key

The pipeline runs entirely in mock mode. `python -m aegeanbench.sports.demo`
produces a complete end-to-end run with synthetic data and 0 external calls.

## Per-source behavior

### `football-data.org`
- **No key**: mock fixture list (8 sample World Cup matches)
- **Free key set**: real `/v4/competitions/WC/teams` + `/matches`. 10 requests/minute. Sufficient for the World Cup; pre-pull and cache.
- **Free key 4xx/5xx**: automatic fall-back to mock; warning logged.

### `soccersapi.com`
- **No key**: mock odds, lineups, h2h.
- **Paid key**: real odds + injuries + lineups (stubs land once key is configured).

### FBref
- `FBREF_LIVE` not set, or `=0`: mock xG table.
- `FBREF_LIVE=1`: live scrape with 5-second rate limit + 6-hour cache.

### LLM predictors
- **No `OPENAI_API_KEY`**: `gpt-5` predictor falls back to a deterministic mock client.
- **No `ANTHROPIC_API_KEY`**: same for `claude-opus-4-7`.
- **No `DEEPSEEK_API_KEY`**: same for `deepseek-v3`.
- The factory `make_llm_predictor_from_env()` auto-selects mock or live per runner_id.

### Aegean consensus
- `AEGEAN_CONSENSUS_URL` not set: mock 6-agent average (offline simulation).
- URL set: HTTP calls to /api/v1/groups, /api/v1/groups/{id}/consensus.
- Server unreachable: graceful fall-back to mock with warning.

### Chat service
- `CHAT_SERVICE_URL` not set: defaults to `http://localhost:9100`.
- Service down: ChatAgent receives an empty window, predicts based on match context only.

## .env file pattern

Save the following as `~/.aegeanbench/.env` and source before launching:

```bash
# free sources
AEGEANBENCH_FOOTBALL_DATA_KEY=...
FBREF_LIVE=1

# paid sources (fill once procured)
AEGEANBENCH_SOCCERSAPI_KEY=...

# LLM keys (fill once procured)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-4-7
DEEPSEEK_API_KEY=...

# infrastructure
AEGEAN_CONSENSUS_URL=http://localhost:8000
CHAT_SERVICE_URL=http://chat-service:9100
```

Load it with:
```bash
set -a && source ~/.aegeanbench/.env && set +a
```

## Where the env vars are read

| Variable | Reading file (line) |
|---|---|
| `AEGEANBENCH_FOOTBALL_DATA_KEY` | `aegeanbench/sports/sources/football_data.py:124` |
| `AEGEANBENCH_SOCCERSAPI_KEY` | `aegeanbench/sports/sources/soccersapi.py` |
| `FBREF_LIVE` | `aegeanbench/sports/sources/fbref.py:_init_` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` | `aegeanbench/sports/predictors/llm.py: make_llm_predictor_from_env()` |
| `AEGEAN_CONSENSUS_URL` | `aegeanbench/sports/predictors/aegean.py: AegeanPredictor.__init__` |
| `CHAT_SERVICE_URL` | `aegean-consensus/src/aegean/agents/sports/chat_agent.py:DEFAULT_CHAT_SERVICE_URL` |

## Rotation policy

API keys should be rotated every 90 days. The benchmark stores no keys
on disk; they live exclusively in the env at process startup.

## Cost guardrails

Per-match cost estimate (when fully live):

```
Claude Opus 4.7 (Stats + Strategy)    $0.06
GPT-5 (Player)                        $0.04
DeepSeek V3 (Market)                  $0.001
Claude Haiku 4.5 (News + Occult)      $0.005
──────────────────────────────────────────
Per Aegean run                        ~$0.11

64 World Cup matches x 1 Aegean run   $7.04
64 matches x 5 predictors             ~$13.50 total
```

The Kelly stake caps in `betting/portfolio.py` keep simulated bankroll
risk separate from LLM spend; live trading is not enabled in this module.
