# Data Storage Inventory

Where every piece of data lives, what survives restart, what to back up.

## TL;DR

| Restart-safe? | What |
|---|---|
| ✅ Yes | Pipeline runs, evaluations, API caches, Kaggle history |
| ❌ No | WebSocket subscribers, in-process scheduler state, in-flight LLM responses |
| 🚫 Not ours | User accounts, chat messages, points, invitations (owned by chat service) |

---

## Disk-persisted data

All under `~/.aegeanbench/` on the host (mounted into the container).

### `~/.aegeanbench/worldcup_runs/`

The single most important directory. One subdirectory per pipeline run.

```
worldcup_runs/
└── run_20260612_180000_a1b2c3d4/
    ├── manifest.json        run metadata + config
    ├── predictions.json     every predictor × every match prediction
    ├── portfolio.json       sized bets + risk caps
    └── evaluation.json      brier / hit rate / ROI (added after match completes)
```

| Property | Value |
|---|---|
| Format | JSON |
| Size | ~50 KB per run; 64 matches × 5 predictors → ~3 MB / day during tournament |
| Read | `persistence.load_run()` or `cat` |
| Written by | `WorldCupPipeline.run()` |
| Updated by | `WorldCupPipeline.evaluate()` (adds `evaluation.json`) |
| Retention | Keep all runs through tournament + 90 days |
| Backup | **Required**. Daily tar to S3 / cold storage |

### `~/.aegeanbench/sports_cache/`

API response cache. Drops latency, controls cost.

```
sports_cache/
├── <hash>.json          one file per (adapter, method, args) hash
└── ...
```

| Property | Value |
|---|---|
| Format | JSON envelope `{cached_at, key_parts, payload}` |
| Size | ~5 MB total during tournament |
| TTL | 30 min (live data) to 24 h (historical) |
| Read by | `FileCache.get()` |
| Written by | each adapter wrapping its fetch() |
| Restart-safe | yes (file-based) |
| Backup | **not required** — recomputable from APIs |
| Wipe | safe any time: `rm -rf ~/.aegeanbench/sports_cache/` |

### `~/.aegeanbench/kaggle/`

Historical international match results, used to train Elo + Dixon-Coles.

```
kaggle/
└── results.csv     ~5 MB, ~46000 rows, 1872 to present
```

| Property | Value |
|---|---|
| Format | CSV |
| Size | 5 MB |
| Read by | `KaggleHistoryLoader.load()` |
| Written by | manual download (see `docs/API_KEYS.md`) |
| Restart-safe | yes |
| Backup | one-time copy is enough |
| Refresh | annually, never during tournament |

### `/var/log/` (or docker logs)

Standard stdout/stderr. Configure your log aggregator to ship these.

---

## In-memory only (lost on restart)

### LiveHub (WebSocket fan-out)

Per-channel `asyncio.Queue` of subscribers. Lives in
`app.state.live_hub`. Resets on restart - the front-end's reconnect
handshake re-establishes subscriptions transparently.

### MatchEventScheduler state

Tracks `last_run_at`, `last_chat_trigger_at`, fired pre-match
checkpoints per match. Resets on restart.

| Resetting means... | ... |
|---|---|
| Last-run timestamp | reset to "never" - next event triggers immediately |
| Pre-match checkpoints fired | reset - T-2h and T-30m can fire again |
| Throttle window | reset - first event in the new process fires |

**Mitigation**: after restart, run the pre-launch sanity script once to
re-prime cached predictions; the throttle window protects you from
spamming consensus on the first few events.

### In-flight predictions / LLM responses

During a `pipeline.run()`, the partial state lives in Python. If the
process dies mid-run that batch is lost; just re-invoke and predictions
will be regenerated. Persistence happens at the END of `pipeline.run()`,
not incrementally.

---

## Not our data (other services own these)

| Data | Owner | Where |
|---|---|---|
| User accounts (email, password hash) | Frontend / auth service | their database |
| Chat rooms | Chat service | their database |
| Chat messages | Chat service | their database |
| User points | Chat service or frontend | their database |
| Invitation links | Frontend | their database |
| Live match TV streams | broadcaster (not in scope) | their CDN |

AegeanBench / aegean-consensus are **stateless about users**. We only
receive identifiers (user_name, room_id) as request parameters and
never store them.

---

## Per-room data — explicitly stateless (V1/V2)

Users create their own chat rooms via the chat service. Multiple rooms
can talk about the same match in parallel. For sprint:

- **Predictions are per-match**, not per-room. One Aegean consensus run
  serves every room talking about that match.
- **ChatAgent and ChatQA use the room's recent messages** to give
  per-room flavour, but we **do not persist per-room state on our side**.
- The chat service can pass `room_id` in requests; we forward it to
  prompts as a soft hint but maintain zero room-specific memory.

V3 (post-tournament) may introduce per-room knowledge bases. Until then,
each room is just a different recent-chat slice.

---

## Restart playbook

Best time to restart: between matches (gaps of >2 hours during group
stage, longer during knockout).

```bash
# Graceful shutdown
docker compose stop aegeanbench
# (aegean-consensus can stay up; restart it separately if needed)

# Pull / rebuild / restart
git pull
docker compose up -d --build aegeanbench

# Verify
curl http://localhost:8200/api/v1/health
docker compose exec aegeanbench python -m aegeanbench.sports.demo

# Front-end will auto-reconnect WebSockets within seconds.
```

If you must restart during a live match: stop AegeanBench, let
aegean-consensus keep running, restart, verify the latest match's
prediction is still in `~/.aegeanbench/worldcup_runs/`.

---

## Backup script (cron, daily 3am)

```bash
#!/usr/bin/env bash
DEST=/backup/aegeanbench
DATE=$(date +%Y%m%d)
mkdir -p "$DEST"
tar czf "$DEST/runs-$DATE.tar.gz" /opt/AegeanBench/data/runs/
# Optional: upload to object storage
aws s3 cp "$DEST/runs-$DATE.tar.gz" "s3://your-bucket/aegeanbench/"
# Retention: keep 30 days locally
find "$DEST" -name 'runs-*.tar.gz' -mtime +30 -delete
```

Crontab entry:

```
0 3 * * * /opt/AegeanBench/scripts/backup_runs.sh >> /var/log/aegeanbench-backup.log 2>&1
```

---

## What recovery looks like

| Scenario | What's lost | What survives | Recovery |
|---|---|---|---|
| Crash mid-pipeline | One in-flight batch of predictions | Everything else | Re-invoke pipeline; ~30 s to recompute |
| Disk wiped | All runs + caches | Code | Restore from S3 backup |
| Server replaced | All runs + caches | Code | Restore + re-init Kaggle CSV |
| LLM provider outage | Nothing (predictor falls back to mock) | All persisted data | Wait + retry |
| Chat service down | Chat-derived prompts go empty | All match-level predictions | Predictions degrade gracefully |
| aegean-consensus down | AegeanPredictor falls back to mock | All persisted data | Restart consensus service |

The system is designed so **persistence boundaries match data
importance**: predictions / evaluations persisted hard, everything else
recomputable.
