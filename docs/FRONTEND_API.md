# Frontend API Contract

All endpoints the product front-end consumes. Single base URL, single auth
scheme (none for sprint, bearer token for v2).

```
Base URL (dev):  http://localhost:8200
Base URL (prod): https://api.your-domain.com
Content-Type:    application/json
CORS:            * (sprint), tighten before launch
```

---

## Public read endpoints (V1)

### 1. Predictor catalog

```http
GET /api/v1/runners

200 OK
{
  "_meta": { "endpoint": "GET /api/v1/runners" },
  "runners": [
    {
      "id": "elo",
      "name": "Elo Ratings",
      "type": "classical",
      "provider": "Aegean",
      "is_ours": true,
      "description": "World Football Elo with stage-based K-factor.",
      "cost_per_match_usd": 0.0
    },
    ...
  ]
}
```

Use it for: predictor selector, model card link, brand badges.

### 2. Tournaments

```http
GET /api/v1/tournaments

200 OK
{
  "tournaments": [
    {
      "id": "fifa-world-cup-2026",
      "name": "FIFA World Cup 2026",
      "asset_class": "sports_football",
      "starts_at": "2026-06-11T00:00:00Z",
      "ends_at": "2026-07-19T00:00:00Z",
      "n_matches": 64,
      "is_active": true
    },
    ...
  ]
}
```

### 3. Leaderboard

```http
GET /api/v1/leaderboard?tournament_id=fifa-world-cup-2026

200 OK
{
  "tournament_id": "fifa-world-cup-2026",
  "evaluated_at": "2026-06-12T22:00:00Z",
  "n_runs_aggregated": 1,
  "rows": [
    {
      "rank": 1,
      "runner_id": "aegean",
      "runner_name": "Aegean Consensus",
      "is_ours": true,
      "n_matches": 8,
      "mean_brier_score": 0.5901,
      "mean_log_loss": 0.985,
      "hit_rate": 0.625,
      "mean_confidence": 0.61,
      "roi": 0.23,
      "composite_score": 64.7,
      "leakage_warning": "low"
    },
    ...
  ]
}
```

Sort: `composite_score` descending. Rank is pre-computed.

### 4. Run bundle

```http
GET /api/v1/runs/{run_id}

200 OK
{
  "run_id": "run_20260612_180000_abc12345",
  "manifest": {
    "competition": "FIFA World Cup 2026",
    "n_matches": 8,
    "runner_ids": ["elo", "dixon_coles", "gpt-5", "claude-opus-4-7", "aegean"],
    "config": { "betting_runner_id": "dixon_coles" }
  },
  "predictions": {
    "elo": [ { "match_id": "...", "p_home_win": ..., ... } ],
    "aegean": [...]
  },
  "portfolio": { "bets": [...], "total_stake": 260.33 },
  "evaluation": { "runner_evaluations": {...}, "betting_evaluation": {...} }
}

404 Not Found  — unknown run_id
```

### 5. Match drill-down

```http
GET /api/v1/runs/{run_id}/matches/{match_id}

200 OK
{
  "run_id": "...",
  "match_id": "WC2026-A1",
  "predictions_per_predictor": [
    {
      "runner_id": "aegean",
      "runner_name": "Aegean Consensus",
      "p_home_win": 0.41,
      "p_draw": 0.31,
      "p_away_win": 0.28,
      "confidence": 0.89,
      "rationale": "...",
      "latency_ms": 4200,
      "tokens_used": 1820
    },
    ...
  ],
  "bets_on_match": [...],
  "evaluation_per_predictor": [...]
}
```

### 6. Runner card

```http
GET /api/v1/runners/{runner_id}/card

200 OK
{
  "runner_id": "aegean",
  "metadata": { "name": "Aegean Consensus", "type": "consensus" },
  "lifetime": {
    "n_matches_evaluated": 56,
    "mean_brier": 0.578,
    "hit_rate": 0.65,
    "n_runs": 7
  },
  "history": [
    { "run_id": "...", "evaluated_at": "...", "n_matches": 8, "mean_brier": ..., "hit_rate": ... }
  ],
  "sample_predictions": [...]
}
```

---

## Health

```http
GET /api/v1/health

200 OK
{ "status": "ok", "runs_dir": "...", "qa_handler": true, "live_hub_subscribers": 3 }
```

---

## V2: @-mention single-agent Q&A

When a user posts `@stats_specialist 巴西今晚进几个？` in chat, the chat
service calls this endpoint to fetch a single-agent reply:

```http
POST /api/v1/agents/{agent_id}/answer
Content-Type: application/json

{
  "question": "巴西今晚进几个？",
  "match_id": "WC2026-A1",           // optional, for context
  "match_context": "...",            // optional, prebuilt prompt
  "user_name": "张三",               // optional, used in greeting
  "recent_messages": [               // optional, last few chat turns
    { "user_name": "李四", "text": "巴西稳了" }
  ]
}

200 OK
{
  "agent_id": "stats_specialist",
  "question": "巴西今晚进几个？",
  "answer": "Based on BRA xG of 2.05 over recent matches, expect 1-2 goals.",
  "confidence": 0.65,
  "rationale": "Recent form supports moderate scoring.",
  "latency_ms": 1840,
  "tokens_used": 720,
  "metadata": { "model": "claude-opus-4-7" }
}

400 Bad Request   — empty 'question'
503 Service Unavailable  — qa_handler not configured at startup
```

Available `agent_id`: `stats_specialist`, `player_specialist`,
`strategy_specialist`, `market_specialist`, `news_specialist`,
`occult_specialist`. Send any string; unknown agents return a friendly
in-band error message rather than a 4xx.

---

## V2: WebSocket live streams

### Global prediction stream

```
WS /ws/predictions

Server -> Client (after accept):
{ "type": "hello", "channel": "predictions" }

Server -> Client (on every published update):
{
  "type": "prediction_update",
  "match_id": "WC2026-A1",
  "runner_id": "aegean",
  "p_home_win": 0.42,
  "p_draw": 0.31,
  "p_away_win": 0.27,
  "ts": "2026-06-12T18:15:00Z"
}
```

### Per-match stream

```
WS /ws/matches/{match_id}

Server -> Client (after accept):
{ "type": "hello", "channel": "match:WC2026-A1" }

Server -> Client (live):
{ "type": "score", "home": 2, "away": 1, "minute": 67 }
{ "type": "event", "kind": "goal", "team": "BRA", "scorer": "Vinicius", "minute": 67 }
{ "type": "prediction_update", "runner_id": "aegean", "p_home_win": 0.78, ... }
{ "type": "settled", "outcome": "home_win" }
```

Reconnect policy: client should reconnect with exponential backoff on
WebSocketDisconnect. Server sends `hello` again on every connection so
the client can drop any stale state.

---

## V2: Live publish (admin / colleague service)

Used by the chat service (or any internal producer) to push events into
the WebSocket fan-out. Not exposed to public clients.

```http
POST /api/v1/_internal/publish

{
  "channel": "match:WC2026-A1",
  "payload": { "type": "score", "home": 1, "away": 0, "minute": 23 }
}

200 OK
{ "published_to": 14 }
```

---

## Field glossary

| Field | Range | Meaning |
|---|---|---|
| `p_home_win` / `p_draw` / `p_away_win` | [0, 1], sum=1 | Outcome probabilities |
| `confidence` | [0, 1] | Model's self-assessment, separate from probs |
| `mean_brier_score` | [0, 2], lower is better | Calibration metric, 2/3 = uniform guess |
| `mean_log_loss` | [0, +inf), lower is better | Penalises confidently wrong predictions |
| `hit_rate` | [0, 1], higher is better | Argmax-prediction accuracy |
| `roi` | (-1, +inf), 0 = break even | Realised return on staked capital |
| `composite_score` | [0, 100], higher is better | Frontend default sort key |
| `leakage_warning` | low / medium / high | Reserved for investment benchmark; always "low" for sports |

---

## Error format

All 4xx / 5xx responses follow FastAPI's default:

```json
{ "detail": "human-readable error message" }
```

For 422 (validation), `detail` is an array of per-field issues.

---

## Rate limits (sprint)

- Public reads: no per-IP cap (in-process cache makes it cheap).
- V2 Q&A POST: 1 request per agent per match per 5 seconds (server-side throttle, returns 429 when exceeded - implemented in launch buffer).
- WebSocket: no message cap; backpressure handled by LiveHub queue (drops oldest).

---

## curl examples

```bash
BASE=http://localhost:8200

# Leaderboard
curl $BASE/api/v1/leaderboard | jq

# Latest run drill-down
RUN=$(ls -t ~/.aegeanbench/worldcup_runs/ | head -1)
curl $BASE/api/v1/runs/$RUN | jq

# Match detail
curl $BASE/api/v1/runs/$RUN/matches/WC2026-A1 | jq

# Q&A
curl -X POST $BASE/api/v1/agents/stats_specialist/answer \
  -H 'Content-Type: application/json' \
  -d '{"question": "Will Brazil top the group?", "user_name": "tester"}'

# WebSocket (websocat or wscat)
websocat ws://localhost:8200/ws/predictions
```
