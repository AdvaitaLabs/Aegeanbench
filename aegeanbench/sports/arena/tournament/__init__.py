"""
Full-tournament prediction for the arena.

Each model forecasts the WHOLE World Cup — champion / runner-up / third,
all 12 group tables, the knockout bracket, and a top-scorer board.

Design (confirmed with product):
  * "Real results so far + predicted remainder": completed matches are
    locked to actual results (from real_state); models only predict what
    hasn't been played. As the tournament progresses the forecast
    self-updates.
  * Regenerated daily + after each match-day completes.
  * Segmented generation with deterministic validation: the LLM predicts
    match scores, but standings / qualification are computed in CODE
    (standings.py) so the tables, advancers, and bracket stay consistent.
"""
