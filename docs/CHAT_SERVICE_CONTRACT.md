# Chat Service Integration Contract

For the colleague building the live chat service.

The chat service handles all user-facing chat (WebSockets, message
storage, moderation). AegeanBench consumes chat in two ways:

1. **Pull**: aggregate the last N minutes of chat for crowd-sentiment
   analysis (V1, already wired up).
2. **Push**: forward `@-mention` questions to AegeanBench so agents can
   answer in-character (V2).

Plus optionally:

3. **Subscribe**: the chat service can also subscribe to AegeanBench's
   live prediction WebSocket and surface updates in the chat UI.

---

## What the chat service must implement

### Endpoint A (V1, required for launch): chat window

AegeanBench polls this endpoint when a match prediction is being built
OR when an `@-mention` Q&A needs recent room context.

```http
GET <chat-service>/api/v1/chat/window?match_id=<id>&minutes=<N>&room_id=<rid>
```

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `match_id` | string | yes | - | e.g. `WC2026-A1` |
| `minutes` | int | no | 30 | window length looking backwards from now |
| `room_id` | string | no | (all rooms) | when set, return messages from this room only |

**Important about rooms**: users create their own chat rooms via your
service. Many rooms can be open for the same match in parallel.

- When `room_id` is provided, return ONLY that room's messages.
- When `room_id` is omitted, return the aggregate across all rooms for
  the match (used by the "global pulse" view; optional - return empty
  array if you don't support aggregation).

The response shape always includes `room_id` (echoing the request, or
`null` for the aggregate view) so clients can verify what they got.

Response (200 OK):

```json
{
  "match_id": "WC2026-A1",
  "room_id": "room_abc123",
  "window_start": "2026-06-12T17:30:00Z",
  "window_end": "2026-06-12T18:00:00Z",
  "total_messages": 142,
  "messages": [
    {
      "message_id": "msg_abc123",
      "user_id": "u_456",
      "user_name": "FootballFan42",
      "text": "Brazil looks unstoppable today",
      "timestamp": "2026-06-12T17:55:00Z",
      "language": "en"
    },
    {
      "message_id": "msg_def456",
      "user_id": "u_789",
      "user_name": "梅西铁粉",
      "text": "阿根廷必胜！！",
      "timestamp": "2026-06-12T17:56:00Z",
      "language": "zh"
    }
  ]
}
```

If no messages exist, return `total_messages: 0` and an empty array.
Never 404.

### Required fields per message

| Field | Type | Notes |
|---|---|---|
| `message_id` | string | unique, stable |
| `user_id` | string | opaque; never PII |
| `user_name` | string | display name |
| `text` | string | UTF-8, no markup |
| `timestamp` | ISO-8601 UTC | |
| `language` | string | `en`, `zh`, etc. ISO-639-1 |

Anything beyond these is ignored by AegeanBench but is fine to include.

### Volume expectations

Per request: up to ~500 messages (cap at the chat service). We sample
the last 30 if more arrive. Calls are made:

- once 30 min before each match starts (predict)
- once at half-time if V3 live re-evaluation is enabled
- on-demand whenever `/api/v1/agents/{id}/answer` includes recent context

---

## Endpoint B (V2, for @-mention Q&A): forward to AegeanBench

When a user @-mentions one of the agents in chat, the chat service
calls AegeanBench:

```http
POST <aegeanbench>/api/v1/agents/<agent_id>/answer
Content-Type: application/json

{
  "question": "巴西今晚进几个？",
  "match_id": "WC2026-A1",
  "room_id": "room_abc123",
  "user_name": "ZhangSan",
  "recent_messages": [
    {"user_name": "LiSi", "text": "巴西稳了"},
    {"user_name": "WangWu", "text": "梅西今晚有点累"}
  ]
}
```

`room_id` and `recent_messages` should both reference the SAME room.
The recent_messages array is the chat service's responsibility to
populate from its own message store; AegeanBench keeps no per-room state.

Recognised agent IDs:

| ID | Persona |
|---|---|
| `stats_specialist` | Numbers nerd (xG, PPDA, recent form) |
| `player_specialist` | Squad / injury analyst |
| `strategy_specialist` | Tactical analyst |
| `market_specialist` | Sharp betting reader |
| `news_specialist` | Morale / drama / news |
| `occult_specialist` | Sports astrologer (entertainment) |

AegeanBench returns:

```json
{
  "agent_id": "stats_specialist",
  "answer": "Based on BRA xG of 2.05, expect 1-2 goals.",
  "confidence": 0.65,
  "latency_ms": 1840
}
```

The chat service should:

1. Render the answer as a chat message posted **as the agent**.
2. Add a `verified by AegeanBench` badge so users can tell agent
   messages apart from human messages.
3. Log `request_id` from the response header for audit.

Timeout: 8 seconds. On timeout / 5xx, fall back to a polite "I'll
get back to you shortly" message and queue retry.

---

## Endpoint C (V2, optional): subscribe to live predictions

If the chat service wants to surface live prediction updates as in-chat
system messages (e.g. "Aegean now estimates 60% home win"):

```
WebSocket: <aegeanbench>/ws/predictions
or
WebSocket: <aegeanbench>/ws/matches/<match_id>
```

See `FRONTEND_API.md` for the message shapes. Suggested behavior:

- Surface significant probability shifts (>10% change) as system
  messages
- Throttle so chat doesn't get spammed with minor flips

---

## Test fixtures

For end-to-end tests without a real chat service, AegeanBench ships
`aegeanbench/sports/chat_mock.py`. The chat service team can:

1. Pull the `MockChatFetcher.fetch()` JSON shape as the canonical schema
2. Run AegeanBench against it before the real service is ready

Example response from the mock (with team-name context):

```bash
PYTHONPATH=. python -c "
from aegeanbench.sports.chat_mock import MockChatFetcher
import json
fetcher = MockChatFetcher(context_hint={
  'WC2026-A1': {'home': 'Brazil', 'away': 'Argentina', 'star': 'Vinicius'}
})
print(json.dumps(fetcher.fetch('WC2026-A1'), indent=2, ensure_ascii=False))
"
```

---

## Auth (sprint vs prod)

Sprint:

- No auth (same VPC).
- AegeanBench reads `CHAT_SERVICE_URL` env var.

Pre-prod hardening:

- Bearer token via `X-API-Key` header.
- AegeanBench reads `CHAT_SERVICE_TOKEN` env var.
- Chat service reads `AEGEANBENCH_TOKEN` env var for its own calls
  into AegeanBench.

---

## Health check

Both services expose:

```http
GET /api/v1/health
```

Sample: AegeanBench returns

```json
{
  "status": "ok",
  "runs_dir": "/root/.aegeanbench/worldcup_runs",
  "qa_handler": true,
  "live_hub_subscribers": 3
}
```

Chat service should expose an equivalent endpoint so each side can
detect the other dropping.

---

## Coordination checklist

| Item | Owner | Done? |
|---|---|---|
| Chat service implements `/api/v1/chat/window` | Chat team | [ ] |
| AegeanBench env var `CHAT_SERVICE_URL` set | DevOps | [ ] |
| Smoke: AegeanBench pulls 1 window from chat service | Chat + Bench | [ ] |
| Chat service implements forwarding for `@agent_name` | Chat team | [ ] |
| Smoke: chat service posts Q&A to AegeanBench, displays reply | Both | [ ] |
| (Optional) Chat service subscribes to `/ws/predictions` | Chat team | [ ] |
| Bearer auth wired both directions | DevOps | [ ] |

---

## Contact

For schema changes or new endpoints, file an issue on the AegeanBench
repo. Schema is versioned via `/api/v1/`; new versions live at `/api/v2/`
without breaking V1 consumers.
