# Launch Checklist · World Cup 2026

Target launch: **6/11 kickoff** (Mexico City, 18:00 local). T-minus column
tracks how far ahead the item must be completed.

## T-2 (by 6/9)

Critical-path items: anything missed here pushes the launch.

- [ ] Procure paid API keys (football-data TIER ONE, soccersapi)
- [ ] Procure LLM API keys (OpenAI, Anthropic, DeepSeek)
- [ ] Configure all env vars (`docs/API_KEYS.md`)
- [ ] Run full pipeline once with real keys: `python -m aegeanbench.sports.cli predict`
- [ ] Verify Reporter JSONs render with real data:
      `python -m aegeanbench.sports.demo --output /tmp/staging`
- [ ] Hand JSON folder to product / frontend team for last-mile UI checks
- [ ] aegean-consensus deployed and reachable at `AEGEAN_CONSENSUS_URL`
- [ ] Verify aegean health endpoint: `curl <AEGEAN_CONSENSUS_URL>/api/v1/health`
- [ ] Chat service end-to-end test with colleague's mock messages

## T-1 (by 6/10 evening)

- [ ] Kaggle historical CSV downloaded to `~/.aegeanbench/kaggle/results.csv`
- [ ] Train Dixon-Coles on the real history (logs show 5000+ matches loaded)
- [ ] Smoke test the FastAPI server:
      `uvicorn aegeanbench.sports.reporter.server:app --port 8200`
      `curl http://localhost:8200/api/v1/health`
- [ ] Generate predictions for the 6/11 opening matches:
      ```
      python -m aegeanbench.sports.cli predict \
          --runners=elo,dixon-coles,gpt-5,claude,deepseek,aegean \
          --label=opening-day
      ```
- [ ] Manual review of opening-match predictions (do they look sane?)
- [ ] Front-end team confirms the leaderboard / match drill pages render
- [ ] Backup plan: confirm mock mode still works if any paid API rate-limits

## T-0 (6/11 morning)

- [ ] Final smoke run with `--no-persist` to verify the live pipeline
- [ ] Server processes started: reporter (port 8200), aegean-consensus
- [ ] Logging configured at INFO level (not DEBUG; INFO is enough)
- [ ] Monitoring: tail `journalctl -u aegeanbench-reporter -f`
- [ ] On-call rotation confirmed for 6/11 - 7/19 (1 person, can be light)

## During tournament (6/11 - 7/19)

Daily cadence:
- [ ] **2 hours before kickoff**: run pipeline for the day's fixtures
- [ ] **30 min before kickoff**: confirm predictions visible on the front-end
- [ ] **After final whistle**: results auto-ingested, `evaluate()` called
- [ ] **End of day**: review leaderboard, file any unexpected results

Per-match real-time (V2 / optional):
- [ ] WebSocket subscribers receiving live updates
- [ ] @-mention Q&A endpoint responding under 5 seconds

Weekly:
- [ ] Monday: review previous week's calibration; tweak agent weights if needed
- [ ] Wednesday: snapshot leaderboard for the demo / blog
- [ ] Friday: backup `~/.aegeanbench/worldcup_runs/` to off-host storage

## Risk mitigations

| Risk | Mitigation | Owner |
|---|---|---|
| LLM provider rate-limits during peak | Aegean falls back to mock automatically | Pipeline |
| football-data 5xx | Adapter falls back to mock + warning logged | Pipeline |
| Chat service outage | ChatAgent receives empty window, doesn't block | Pipeline |
| aegean-consensus crash | AegeanPredictor falls back to mock 6-agent average | Pipeline |
| Bad prediction goes viral | Manual override possible; Aegean confidence is shown | Comms |
| Reporter server OOM | Stateless, restart safe; runs persisted on disk | Ops |

## Post-tournament (7/20+)

- [ ] Final leaderboard snapshot + blog post
- [ ] Compare aegean vs single-LLM mean Brier over 64 matches
- [ ] Aggregate ROI report (we did not run live trades; this is paper-only)
- [ ] Retrospective: which agent weights surprised us?
- [ ] Decide whether to keep the sports module or move on to investments
      benchmark (`plan_prediction_market.md`)

## Communication contacts

| Role | Person | Contact |
|---|---|---|
| Product lead | TBD | - |
| Frontend lead | TBD | - |
| Chat service owner | TBD | - |
| On-call (6/11-7/19) | TBD | - |
| Finance (API key procurement) | TBD | - |

## Rollback procedure

If something is on fire post-launch:

1. **Stop the reporter server** (front-end keeps showing the last cached JSON
   from the static `build_all_endpoints` output, which is safe).
   ```bash
   systemctl stop aegeanbench-reporter
   ```
2. **Investigate** with the saved run directories under
   `~/.aegeanbench/worldcup_runs/`. Each run is self-contained JSON.
3. **Re-run with `--no-persist`** locally to validate fixes.
4. **Resume** when the new run looks clean:
   ```bash
   systemctl start aegeanbench-reporter
   ```

There is no destructive state - the worst case is "frontend shows stale
predictions for an hour".
