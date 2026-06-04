# Deployment Guide

How to ship AegeanBench + aegean-consensus together for the World Cup 2026
launch. Designed to fit on a single 2C4G VM; can be split later if needed.

---

## Architecture (one VM, two services)

```
┌─────────────────────────────────────────────────────────┐
│ Single host (e.g. 2C4G Linux VM)                         │
│                                                          │
│  ┌────────────────────────┐  ┌─────────────────────────┐│
│  │ AegeanBench Reporter   │  │ aegean-consensus        ││
│  │ Port 8200 (public)     │──│ Port 8000 (internal)    ││
│  │ FastAPI + uvicorn      │  │ FastAPI + uvicorn       ││
│  └────────────────────────┘  └─────────────────────────┘│
│           │                              │              │
│           └────┬─────────────────────────┘              │
│                ▼                                         │
│  ┌────────────────────────────────────────┐             │
│  │ Persisted state (volume mount)          │             │
│  │   ~/.aegeanbench/worldcup_runs/         │             │
│  │   ~/.aegeanbench/sports_cache/          │             │
│  │   ~/.aegeanbench/kaggle/                │             │
│  └────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────┘
            │ HTTPS via nginx / Caddy
            ▼
        Public Internet
            │
       ┌────┴────┐
       │ Browser │ -> only sees port 8200
       └─────────┘
```

**Key point**: the frontend ONLY talks to port 8200 (AegeanBench). It
never sees aegean-consensus directly. AegeanBench calls aegean-consensus
internally over `http://127.0.0.1:8000` (or service-mesh DNS).

---

## One-command launch (recommended)

`docker-compose up -d` brings both services up with all the right
networking and volumes pre-configured. See `docker-compose.yml` at the
repo root.

```bash
# 1. Clone both repos side-by-side
git clone <aegean-consensus-url> /opt/aegean-consensus
git clone <aegeanbench-url>      /opt/AegeanBench
cd /opt/AegeanBench

# 2. Populate env
cp docs/.env.example .env
vim .env   # fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

# 3. (Optional, one-time) Download Kaggle historical data
mkdir -p data/kaggle
curl -L "<your-mirror-of-kaggle-csv>" -o data/kaggle/results.csv

# 4. Up
docker-compose up -d

# 5. Smoke test
curl http://localhost:8200/api/v1/health
curl http://localhost:8000/api/v1/health   # only from the host itself
```

---

## Pre-launch benchmark (before opening kick-off)

Once both services are up, you can dry-run the full pipeline against any
historical fixtures to validate model calibration. Two ways:

### A) Synthetic ground truth (offline, ~2 seconds)

```bash
docker-compose exec aegeanbench python -m aegeanbench.sports.demo
```

This builds a complete run with mock fixtures, injects fake outcomes,
and prints the leaderboard. Useful for verifying the whole pipeline is
wired correctly.

### B) Real historical replay (offline, depends on data)

```bash
docker-compose exec aegeanbench python scripts/pre_launch_benchmark.py
```

This runs every predictor against Euro 2024 fixtures (already played -
real ground truth available from football-data), evaluates them, and
emits a calibration report. If a predictor's Brier score is way out of
line with expectations (>0.75 = worse than uniform), adjust its
configuration before the World Cup.

Adjust knobs based on results:

- Aegean too conservative? Lower `min_edge` in `PortfolioConfig`.
- LLM too random? Lower temperature in `LLMClient` config.
- Dixon-Coles off? Re-train with more historical data (`--train-history 2000`).

---

## Detailed start sequence

### Without docker-compose (bare metal)

Terminal 1 (aegean-consensus):

```bash
cd /opt/aegean-consensus
source .venv/bin/activate
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export DEEPSEEK_API_KEY=...
python main.py --port 8000
```

Terminal 2 (AegeanBench reporter):

```bash
cd /opt/AegeanBench
source .venv/bin/activate
export AEGEAN_CONSENSUS_URL=http://127.0.0.1:8000
export AEGEANBENCH_FOOTBALL_DATA_KEY=...
export AEGEANBENCH_SOCCERSAPI_KEY=...
export OPENWEATHER_API_KEY=...
export CHAT_SERVICE_URL=http://chat-service:9100
export FBREF_LIVE=1
uvicorn aegeanbench.sports.reporter.server:app --host 0.0.0.0 --port 8200
```

Terminal 3 (optional - cron job that generates predictions ahead of each match):

```bash
# Hourly cron
0 * * * * /opt/AegeanBench/.venv/bin/python -m aegeanbench.sports.cli predict \
            --runners=elo,dixon-coles,gpt-5,claude,deepseek,aegean \
            --label="$(date +%Y%m%d_%H)"
```

### Process supervision

For production, wrap each service in systemd so they auto-restart:

```ini
# /etc/systemd/system/aegean-consensus.service
[Unit]
Description=Aegean Consensus
After=network.target

[Service]
WorkingDirectory=/opt/aegean-consensus
EnvironmentFile=/opt/aegean-consensus/.env
ExecStart=/opt/aegean-consensus/.venv/bin/python main.py --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/aegeanbench.service
[Unit]
Description=AegeanBench Reporter
After=network.target aegean-consensus.service

[Service]
WorkingDirectory=/opt/AegeanBench
EnvironmentFile=/opt/AegeanBench/.env
ExecStart=/opt/AegeanBench/.venv/bin/uvicorn aegeanbench.sports.reporter.server:app \
          --host 0.0.0.0 --port 8200 --workers 2
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable + start:

```bash
sudo systemctl enable --now aegean-consensus aegeanbench
```

---

## Reverse proxy (nginx / Caddy)

Terminate TLS and expose only port 8200. Example nginx snippet:

```nginx
server {
    listen 443 ssl http2;
    server_name api.your-domain.com;
    ssl_certificate     /etc/letsencrypt/live/api.your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.your-domain.com/privkey.pem;

    # WebSocket support
    location /ws/ {
        proxy_pass http://127.0.0.1:8200;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_read_timeout 3600;
    }

    location / {
        proxy_pass http://127.0.0.1:8200;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

aegean-consensus stays on 127.0.0.1:8000, never exposed.

---

## Volumes

Mount three host directories into the container(s) so state survives
restarts and image rebuilds:

| Host path | Container path | Purpose |
|---|---|---|
| `./data/runs/` | `/root/.aegeanbench/worldcup_runs/` | Pipeline run history |
| `./data/cache/` | `/root/.aegeanbench/sports_cache/` | API response cache |
| `./data/kaggle/` | `/root/.aegeanbench/kaggle/` | Historical CSV |

Back up `./data/runs/` weekly during the tournament.

---

## Logs

Default: stdout. Redirect via systemd or docker-compose to your log
aggregator. Important log levels:

- `INFO`: pipeline progress, runner counts
- `WARNING`: data-source fallbacks (mock kicking in), rate-limit hits
- `ERROR`: predictor failures, persistence write errors

Tail during launch day:

```bash
docker-compose logs -f aegeanbench
docker-compose logs -f aegean-consensus
```

---

## Monitoring (optional but recommended)

Two lightweight checks:

1. **Health probes**: Uptime monitor hits `/api/v1/health` every minute
2. **Run freshness**: `ls -1t ~/.aegeanbench/worldcup_runs/ | head -1`
   should be < 1 hour old during the tournament

If you want metrics in Prometheus format, FastAPI has a `prometheus-fastapi-instrumentator` package - add post-launch if needed.

---

## Rollback

`~/.aegeanbench/worldcup_runs/` is the only stateful directory and is
write-once-per-run, so rollback is straightforward:

```bash
# 1. Stop services
sudo systemctl stop aegeanbench

# 2. Front-end keeps showing the last successfully built JSON snapshot
#    from /tmp/aegean_demo_api/ (or wherever build_all_endpoints wrote)

# 3. Investigate via persisted run dirs
ls -la ~/.aegeanbench/worldcup_runs/

# 4. Resume
sudo systemctl start aegeanbench
```

No destructive operations - the worst case is "frontend stale for an hour".

---

## Cost / resource sizing

For a 2C4G VM running both services:

| Metric | Idle | One pipeline run | Steady (10 RPS reads) |
|---|---|---|---|
| CPU | ~5% | spikes to 80% for ~10s | ~30% |
| Memory | ~600 MB | + 400 MB during run | ~1.5 GB |
| Disk | n/a | +50 KB per run dir | +500 KB / day |
| Network | n/a | + LLM API outbound (~2-5 MB) | + chat / fb-data |

A 64-match tournament generates < 10 MB of persisted data. The VM has
ample headroom.

---

## Pre-launch sanity script

Before going live, run this from the host to confirm everything is wired:

```bash
#!/usr/bin/env bash
set -e
echo "=== Health checks ==="
curl -sf http://localhost:8200/api/v1/health | jq
curl -sf http://localhost:8000/api/v1/health | jq

echo "=== Reporter endpoints ==="
curl -sf http://localhost:8200/api/v1/runners | jq '.runners | length'
curl -sf http://localhost:8200/api/v1/tournaments | jq '.tournaments[0].id'
curl -sf http://localhost:8200/api/v1/leaderboard | jq '.rows | length'

echo "=== Q&A smoke ==="
curl -sf -X POST http://localhost:8200/api/v1/agents/stats_specialist/answer \
  -H 'Content-Type: application/json' \
  -d '{"question": "Hello"}'

echo "=== Pre-launch benchmark ==="
docker-compose exec -T aegeanbench python -m aegeanbench.sports.demo > /tmp/demo.log
tail -20 /tmp/demo.log

echo "=== Pass ==="
```

Save as `scripts/pre_launch_check.sh` and run on T-0 morning.
