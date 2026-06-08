from datetime import datetime, timezone
from typing import Optional


def parse_oanda_timestamp(raw: Optional[str]) -> datetime:
    """Parse OANDA timestamp to timezone-aware UTC datetime.

    Handles:
      "2026-01-07T14:00:01.123456789Z"       (nanoseconds + Z)
      "2026-01-07T14:00:01Z"                  (no fractional + Z)
      "2026-01-07T14:00:01+00:00"             (explicit tz, no fractional)
      "2026-01-07T14:00:01.123456789+00:00"   (nanoseconds + explicit tz)
      None / ""                                → raises ValueError
    """
    if not raw:
        raise ValueError("Empty or None timestamp")

    s = raw.strip()

    # Extract timezone suffix if present (+00:00, -04:00, Z)
    tz_suffix = ""
    if s.endswith("Z"):
        s = s[:-1]
        tz_suffix = "+00:00"
    elif len(s) > 6 and s[-6] in ('+', '-') and ':' in s[-5:]:
        tz_suffix = s[-6:]
        s = s[:-6]

    # Strip fractional seconds (nanoseconds)
    if "." in s:
        s = s.split(".")[0]

    # Add timezone if none found
    if not tz_suffix:
        tz_suffix = "+00:00"

    return datetime.fromisoformat(s + tz_suffix).astimezone(timezone.utc)
