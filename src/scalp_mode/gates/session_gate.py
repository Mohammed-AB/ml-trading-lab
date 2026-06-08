"""Session Gate — Step 1 in the decision pipeline.

Determines if trading is allowed based on the current UTC time,
session windows, and DST rules per spec A.1.

DST Rules (hardcoded table, NOT timezone library per spec):
- US DST: 2nd Sunday of March → 1st Sunday of November
- UK DST: Last Sunday of March → Last Sunday of October
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Optional

# sessions.mode in settings.yaml
MODE_OVERLAP_ONLY = "overlap_only"
MODE_WEEKDAY_EXTENDED = "weekday_extended"


@dataclass
class SessionResult:
    allowed: bool
    window_name: str
    next_open_utc: Optional[datetime] = None


# --- DST Table ---

def _nth_sunday(year: int, month: int, n: int) -> datetime:
    """Return the nth Sunday of a given month (1-indexed)."""
    first_day = datetime(year, month, 1, tzinfo=timezone.utc)
    # Days until first Sunday
    days_to_sunday = (6 - first_day.weekday()) % 7
    first_sunday = first_day + timedelta(days=days_to_sunday)
    return first_sunday + timedelta(weeks=n - 1)


def _last_sunday(year: int, month: int) -> datetime:
    """Return the last Sunday of a given month."""
    if month == 12:
        next_month_first = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_month_first = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    last_day = next_month_first - timedelta(days=1)
    days_since_sunday = (last_day.weekday() + 1) % 7
    return last_day - timedelta(days=days_since_sunday)


def is_us_dst(utc_now: datetime) -> bool:
    """Is US currently in DST? (2nd Sunday Mar 07:00 UTC → 1st Sunday Nov 06:00 UTC)."""
    year = utc_now.year
    # US springs forward at 2am EST = 07:00 UTC
    dst_start = _nth_sunday(year, 3, 2).replace(hour=7)
    # US falls back at 2am EDT = 06:00 UTC
    dst_end = _nth_sunday(year, 11, 1).replace(hour=6)
    return dst_start <= utc_now < dst_end


def is_uk_dst(utc_now: datetime) -> bool:
    """Is UK currently in BST? (Last Sunday Mar 01:00 UTC → Last Sunday Oct 01:00 UTC)."""
    year = utc_now.year
    dst_start = _last_sunday(year, 3).replace(hour=1)
    dst_end = _last_sunday(year, 10).replace(hour=1)
    return dst_start <= utc_now < dst_end


# --- Session Windows ---

# Windows defined as (start_hour, start_min, end_hour, end_min)
# Separate winter/summer entries; DST shifts apply to both US and UK independently

OVERLAP_WINDOWS = {
    # (UK_DST, US_DST): (start_h, start_m, end_h, end_m)
    (False, False): (13, 0, 16, 30),   # Both winter
    (True, True):   (12, 0, 15, 30),   # Both summer
    (True, False):  (13, 0, 15, 30),   # UK summer, US winter (narrowed by 1h)
    (False, True):  (12, 0, 16, 30),   # UK winter, US summer (rare, ~1 week)
}

BLOCKED_WINDOWS = {
    "witching_hour": {
        False: (22, 0, 24, 0),   # Winter
        True:  (21, 0, 23, 0),   # Summer (US DST)
    },
    "rollover": {
        False: (21, 55, 22, 10),
        True:  (20, 55, 21, 10),
    },
}


def _time_in_range(utc_now: datetime, start_h: int, start_m: int,
                   end_h: int, end_m: int) -> bool:
    """Check if current UTC time falls within an hour:minute range.

    Handles midnight crossing (e.g., 22:00-00:00).
    """
    current = utc_now.hour * 60 + utc_now.minute
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m

    if end <= start:
        # Crosses midnight
        return current >= start or current < end
    return start <= current < end


def _next_overlap_open(utc_now: datetime, window: tuple) -> datetime:
    """Calculate the next overlap window opening."""
    start_h, start_m = window[0], window[1]
    candidate = utc_now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    if candidate <= utc_now:
        candidate += timedelta(days=1)
    # Skip weekends (Sat=5, Sun=6)
    while candidate.weekday() in (5, 6):
        candidate += timedelta(days=1)
    return candidate


def is_session_allowed(
    utc_now: datetime,
    *,
    mode: str = MODE_OVERLAP_ONLY,
    block: Optional[list[str]] = None,
) -> SessionResult:
    """Determine if trading is allowed at the given UTC time.

    Returns SessionResult with:
    - allowed: bool
    - window_name: descriptive name of the current window
    - next_open_utc: when the next trading window opens (if blocked)

    Args:
        utc_now: Current time in UTC.
        mode: ``overlap_only`` (default) = London–NY overlap only (spec V1).
              ``weekday_extended`` = Mon–Fri any time, still blocked on weekend and
              by optional ``block`` windows (witching_hour, rollover).
        block: When mode is ``weekday_extended``, which sub-windows to enforce.
               Default ``None`` means ``["witching_hour", "rollover"]``.
               Empty list = no sub-blocks (trade all weekday hours; use with care).

    Per spec: V1 Conservative = overlap only.
    """
    # Weekend check (forex market closed)
    weekday = utc_now.weekday()
    if weekday == 5:  # Saturday
        next_open = utc_now + timedelta(days=2)
        next_open = next_open.replace(hour=0, minute=0, second=0, microsecond=0)
        return SessionResult(False, "weekend", next_open)
    if weekday == 6:  # Sunday — market opens ~22:00 UTC but overlap is next day
        next_open = (utc_now + timedelta(days=1)).replace(
            hour=12, minute=0, second=0, microsecond=0
        )
        return SessionResult(False, "weekend", next_open)

    uk_dst = is_uk_dst(utc_now)
    us_dst = is_us_dst(utc_now)
    overlap_window = OVERLAP_WINDOWS[(uk_dst, us_dst)]

    def _blocked_by_named_windows(names: list[str]) -> Optional[SessionResult]:
        for block_name in names:
            if block_name not in BLOCKED_WINDOWS:
                continue
            dst_windows = BLOCKED_WINDOWS[block_name]
            window = dst_windows[us_dst]
            if _time_in_range(utc_now, *window):
                return SessionResult(
                    False,
                    block_name,
                    _next_overlap_open(utc_now, overlap_window),
                )
        return None

    # --- weekday_extended: Mon–Fri, optional witching/rollover only ---
    if mode == MODE_WEEKDAY_EXTENDED:
        block_names = (
            block if block is not None else ["witching_hour", "rollover"])
        hit = _blocked_by_named_windows(block_names)
        if hit:
            return hit
        return SessionResult(True, "weekday_extended", None)

    # --- overlap_only (default): V1 overlap + blocked windows ---
    hit = _blocked_by_named_windows(list(BLOCKED_WINDOWS.keys()))
    if hit:
        return hit

    if _time_in_range(utc_now, *overlap_window):
        return SessionResult(True, "london_newyork_overlap")

    return SessionResult(
        False,
        "outside_overlap",
        _next_overlap_open(utc_now, overlap_window),
    )
