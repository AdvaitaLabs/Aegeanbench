"""
Decides WHEN to re-trigger consensus during a World Cup match.

Design:
    - Pre-match (more than 30 min before kickoff): consensus runs at
      scheduled checkpoints (T-2h, T-30m). No reaction to chat or polls.
    - In-match: events from SoccersAPI livescore polling drive consensus.
      High-priority events (goal, red card, half-time, full-time) trigger
      immediately. User chat heat triggers softly if no other event fired
      recently.
    - Throttle: never re-trigger within MIN_RECONSENSUS_SECONDS of the
      previous run for the same match.

The scheduler does not run the consensus itself - it just emits
"trigger" instructions. The caller (a background worker) is responsible
for actually invoking the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

from aegeanbench.sports.sources.soccersapi_live import (
    LiveEvent,
    LiveEventKind,
    LiveMatchState,
)

logger = logging.getLogger(__name__)


# Minimum gap between consecutive consensus runs for the same match.
MIN_RECONSENSUS_SECONDS = 300   # 5 minutes
PRE_MATCH_CHECKPOINTS_HOURS = (2.0, 0.5)   # T-2h and T-30m

# Soft chat-heat triggers (only fire when nothing else has lately).
CHAT_HEAT_MIN_MESSAGES = 30
CHAT_HEAT_WINDOW_SECONDS = 300        # 5 minutes
CHAT_HEAT_COOLDOWN_SECONDS = 600      # 10 minutes after a chat-driven run


class TriggerReason(str, Enum):
    PRE_MATCH_SCHEDULED = "pre_match_scheduled"
    LIVE_EVENT_HIGH_PRIORITY = "live_event_high_priority"
    LIVE_EVENT_HALF_TIME = "live_event_half_time"
    LIVE_EVENT_FULL_TIME = "live_event_full_time"
    CHAT_HEAT = "chat_heat"
    MANUAL = "manual"


@dataclass
class ConsensusTrigger:
    """Instruction emitted by the scheduler to re-run consensus."""
    match_id: str
    reason: TriggerReason
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 5             # 1 = highest
    detail: str = ""

    def to_dict(self) -> Dict:
        return {
            "match_id": self.match_id,
            "reason": self.reason.value,
            "triggered_at": self.triggered_at.isoformat(),
            "priority": self.priority,
            "detail": self.detail,
        }


# ----------------------------- scheduler -----------------------------


class MatchEventScheduler:
    """
    Stateful scheduler tracking when each match was last evaluated.

    Public API:
        record_last_run(match_id)       # call after a consensus run completes
        on_live_event(event)            # feed a LiveEvent in
        on_chat_heat(...)               # feed an aggregated chat signal in
        on_clock_tick(now, kickoff_at)  # called by the pre-match driver
        manual_trigger(match_id, why)   # operator override

    Each method returns either None (no trigger) or a ConsensusTrigger.
    """

    def __init__(
        self,
        min_reconsensus_seconds: float = MIN_RECONSENSUS_SECONDS,
        chat_heat_threshold: int = CHAT_HEAT_MIN_MESSAGES,
        chat_heat_window: float = CHAT_HEAT_WINDOW_SECONDS,
    ):
        self.min_gap = timedelta(seconds=min_reconsensus_seconds)
        self.chat_heat_threshold = chat_heat_threshold
        self.chat_heat_window = timedelta(seconds=chat_heat_window)
        self._last_run_at: Dict[str, datetime] = {}
        self._last_chat_trigger_at: Dict[str, datetime] = {}
        self._fired_pre_match: Dict[str, set] = {}   # match_id -> set of checkpoints fired

    # ---------- bookkeeping ----------

    def record_last_run(self, match_id: str, when: Optional[datetime] = None) -> None:
        self._last_run_at[match_id] = when or datetime.now(timezone.utc)

    def _enforce_throttle(self, match_id: str, now: datetime) -> bool:
        """Return True if a new run is allowed under the throttle."""
        last = self._last_run_at.get(match_id)
        if last is None:
            return True
        return now - last >= self.min_gap

    # ---------- triggers ----------

    def on_live_event(self, event: LiveEvent) -> Optional[ConsensusTrigger]:
        now = datetime.now(timezone.utc)
        if not self._enforce_throttle(event.match_id, now):
            logger.debug(
                "throttled: %s event for %s within %ds of last run",
                event.kind.value, event.match_id, self.min_gap.total_seconds(),
            )
            return None

        if event.kind == LiveEventKind.HALF_TIME:
            reason = TriggerReason.LIVE_EVENT_HALF_TIME
            priority = 2
        elif event.kind == LiveEventKind.FULL_TIME:
            reason = TriggerReason.LIVE_EVENT_FULL_TIME
            priority = 1
        elif event.is_high_priority:
            reason = TriggerReason.LIVE_EVENT_HIGH_PRIORITY
            priority = 1
        else:
            return None   # low-priority events (subs, yellow cards) don't trigger

        self._last_run_at[event.match_id] = now
        return ConsensusTrigger(
            match_id=event.match_id,
            reason=reason,
            triggered_at=now,
            priority=priority,
            detail=f"{event.kind.value} @ {event.minute}'",
        )

    def on_chat_heat(
        self,
        match_id: str,
        message_count_in_window: int,
        strong_sentiment: bool = False,
    ) -> Optional[ConsensusTrigger]:
        """
        Soft trigger when chat volume spikes. Only fires if:
          - count is above threshold
          - no other consensus run in the last MIN_RECONSENSUS_SECONDS
          - no chat-driven trigger in the last CHAT_HEAT_COOLDOWN_SECONDS
        """
        now = datetime.now(timezone.utc)
        if message_count_in_window < self.chat_heat_threshold and not strong_sentiment:
            return None
        if not self._enforce_throttle(match_id, now):
            return None
        last_chat = self._last_chat_trigger_at.get(match_id)
        if last_chat and (now - last_chat).total_seconds() < CHAT_HEAT_COOLDOWN_SECONDS:
            return None

        self._last_run_at[match_id] = now
        self._last_chat_trigger_at[match_id] = now
        return ConsensusTrigger(
            match_id=match_id,
            reason=TriggerReason.CHAT_HEAT,
            triggered_at=now,
            priority=3,
            detail=f"{message_count_in_window} msgs/{int(self.chat_heat_window.total_seconds())}s",
        )

    def on_clock_tick(
        self,
        match_id: str,
        kickoff_at: datetime,
        now: Optional[datetime] = None,
    ) -> Optional[ConsensusTrigger]:
        """
        Pre-match scheduled checkpoint. Driver should call this for every
        upcoming fixture at least once a minute - the scheduler decides
        whether the current moment is "T-2h" or "T-30m" boundary and
        fires only once per checkpoint per match.
        """
        now = now or datetime.now(timezone.utc)
        if kickoff_at.tzinfo is None:
            kickoff_at = kickoff_at.replace(tzinfo=timezone.utc)
        seconds_to_kickoff = (kickoff_at - now).total_seconds()
        if seconds_to_kickoff <= 0:
            return None   # match started; in-play handler takes over

        fired = self._fired_pre_match.setdefault(match_id, set())

        for hours in PRE_MATCH_CHECKPOINTS_HOURS:
            window_lo = hours * 3600 - 60     # 60-second window around the checkpoint
            window_hi = hours * 3600 + 60
            if window_lo <= seconds_to_kickoff <= window_hi and hours not in fired:
                fired.add(hours)
                self._last_run_at[match_id] = now
                return ConsensusTrigger(
                    match_id=match_id,
                    reason=TriggerReason.PRE_MATCH_SCHEDULED,
                    triggered_at=now,
                    priority=4,
                    detail=f"T-{hours}h checkpoint",
                )
        return None

    def manual_trigger(self, match_id: str, detail: str = "") -> ConsensusTrigger:
        """Operator override - always fires regardless of throttle."""
        now = datetime.now(timezone.utc)
        self._last_run_at[match_id] = now
        return ConsensusTrigger(
            match_id=match_id,
            reason=TriggerReason.MANUAL,
            triggered_at=now,
            priority=1,
            detail=detail,
        )

    # ---------- introspection ----------

    def last_run_at(self, match_id: str) -> Optional[datetime]:
        return self._last_run_at.get(match_id)

    def state_snapshot(self) -> Dict[str, Dict]:
        """For debugging / dashboards."""
        return {
            mid: {
                "last_run": ts.isoformat(),
                "pre_match_fired": list(self._fired_pre_match.get(mid, set())),
                "last_chat_trigger": (
                    self._last_chat_trigger_at[mid].isoformat()
                    if mid in self._last_chat_trigger_at else None
                ),
            }
            for mid, ts in self._last_run_at.items()
        }
