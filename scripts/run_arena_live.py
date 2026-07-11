#!/usr/bin/env python3
"""Moved: this script is now scripts/run_benchmark_arena.py.

Renamed to stop the name collision with the (unrelated) sports arena
(`aegeanbench/sports/arena`, `GET /api/v1/arena/models` — the FIFA World Cup
feature). This shim forwards so old commands keep working; switch your
invocations to:

    python scripts/run_benchmark_arena.py [--emit-js] [--dry]
"""

import runpy
import sys
from pathlib import Path

print("[run_arena_live.py] moved → forwarding to scripts/run_benchmark_arena.py "
      "(update your command; this shim will eventually be removed)", file=sys.stderr)
runpy.run_path(str(Path(__file__).with_name("run_benchmark_arena.py")), run_name="__main__")
