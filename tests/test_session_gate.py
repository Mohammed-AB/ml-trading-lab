"""Tests for Session Gate."""

import pytest
from datetime import datetime, timezone
from src.scalp_mode.gates.session_gate import (
    is_session_allowed,
    is_us_dst,
    is_uk_dst,
    _nth_sunday,
    _last_sunday,
)


class TestDSTCalculations:
    def test_nth_sunday_march_2026(self):
        # 2nd Sunday of March 2026 = March 8
        result = _nth_sunday(2026, 3, 2)
        assert result.month == 3
        assert result.day == 8
        assert result.weekday() == 6  # Sunday

    def test_last_sunday_october_2026(self):
        result = _last_sunday(2026, 10)
        assert result.month == 10
        assert result.weekday() == 6  # Sunday
        assert result.day == 25

    def test_last_sunday_march_2026(self):
        result = _last_sunday(2026, 3)
        assert result.month == 3
        assert result.weekday() == 6
        assert result.day == 29

    def test_us_dst_summer(self):
        # July 2026 — US in DST
        summer = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        assert is_us_dst(summer) is True

    def test_us_dst_winter(self):
        # January 2026 — US not in DST
        winter = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        assert is_us_dst(winter) is False

    def test_uk_dst_summer(self):
        summer = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        assert is_uk_dst(summer) is True

    def test_uk_dst_winter(self):
        winter = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        assert is_uk_dst(winter) is False

    def test_gap_week_uk_summer_us_winter(self):
        # Last week of March 2026: UK switched (Mar 29), US switched (Mar 8)
        # Actually both are in DST by late March. Let's test mid-March:
        # March 15: US DST started (Mar 8), UK not yet (Mar 29)
        gap = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
        assert is_us_dst(gap) is True
        assert is_uk_dst(gap) is False


class TestSessionAllowed:
    def test_overlap_winter_allowed(self):
        # January, Wednesday 14:00 UTC — winter overlap (13:00-16:30)
        dt = datetime(2026, 1, 7, 14, 0, tzinfo=timezone.utc)
        result = is_session_allowed(dt)
        assert result.allowed is True
        assert result.window_name == "london_newyork_overlap"

    def test_overlap_summer_allowed(self):
        # July, Wednesday 13:00 UTC — summer overlap (12:00-15:30)
        dt = datetime(2026, 7, 8, 13, 0, tzinfo=timezone.utc)
        result = is_session_allowed(dt)
        assert result.allowed is True
        assert result.window_name == "london_newyork_overlap"

    def test_outside_overlap_blocked(self):
        # January, Wednesday 10:00 UTC — London session but not overlap
        dt = datetime(2026, 1, 7, 10, 0, tzinfo=timezone.utc)
        result = is_session_allowed(dt)
        assert result.allowed is False
        assert result.window_name == "outside_overlap"
        assert result.next_open_utc is not None

    def test_witching_hour_blocked(self):
        # January, Wednesday 22:30 UTC — witching hour (22:00-00:00 winter)
        dt = datetime(2026, 1, 7, 22, 30, tzinfo=timezone.utc)
        result = is_session_allowed(dt)
        assert result.allowed is False
        assert result.window_name == "witching_hour"

    def test_rollover_blocked(self):
        # January, Wednesday 22:00 UTC — rollover (21:55-22:10 winter)
        dt = datetime(2026, 1, 7, 22, 0, tzinfo=timezone.utc)
        result = is_session_allowed(dt)
        assert result.allowed is False
        # Could be witching_hour or rollover since they overlap

    def test_saturday_blocked(self):
        dt = datetime(2026, 1, 10, 14, 0, tzinfo=timezone.utc)  # Saturday
        result = is_session_allowed(dt)
        assert result.allowed is False
        assert result.window_name == "weekend"

    def test_sunday_blocked(self):
        dt = datetime(2026, 1, 11, 14, 0, tzinfo=timezone.utc)  # Sunday
        result = is_session_allowed(dt)
        assert result.allowed is False
        assert result.window_name == "weekend"

    def test_early_morning_blocked(self):
        # January, Wednesday 03:00 UTC — Asian session, not overlap
        dt = datetime(2026, 1, 7, 3, 0, tzinfo=timezone.utc)
        result = is_session_allowed(dt)
        assert result.allowed is False
        assert result.window_name == "outside_overlap"

    def test_overlap_end_boundary(self):
        # Winter overlap ends at 16:30 — 16:29 should be allowed
        dt = datetime(2026, 1, 7, 16, 29, tzinfo=timezone.utc)
        result = is_session_allowed(dt)
        assert result.allowed is True

    def test_overlap_end_boundary_closed(self):
        # Winter overlap ends at 16:30 — 16:30 should be blocked
        dt = datetime(2026, 1, 7, 16, 30, tzinfo=timezone.utc)
        result = is_session_allowed(dt)
        assert result.allowed is False

    def test_weekday_extended_allows_outside_overlap(self):
        # Asian session Wednesday 03:00 UTC — blocked in overlap_only
        dt = datetime(2026, 1, 7, 3, 0, tzinfo=timezone.utc)
        assert is_session_allowed(dt).allowed is False
        r = is_session_allowed(dt, mode="weekday_extended")
        assert r.allowed is True
        assert r.window_name == "weekday_extended"

    def test_weekday_extended_blocks_witching(self):
        dt = datetime(2026, 1, 7, 22, 30, tzinfo=timezone.utc)
        r = is_session_allowed(dt, mode="weekday_extended")
        assert r.allowed is False
        assert r.window_name == "witching_hour"

    def test_weekday_extended_saturday_still_weekend(self):
        dt = datetime(2026, 1, 10, 3, 0, tzinfo=timezone.utc)
        r = is_session_allowed(dt, mode="weekday_extended")
        assert r.allowed is False
        assert r.window_name == "weekend"
