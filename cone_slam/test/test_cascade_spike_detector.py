"""Tests for the DA-failure cascade-spike detector.

The detector decides whether a single cone-scan looks anomalous enough
that committing its cone factors would corrupt the factor graph (cf.
issue #447 — the cascade pattern observed on lap_postveto_20260511_142317
and lap_ekf_inloop_20260511_235510).

Two gates are checked, AND'd with a post-discovery step floor:

  * percentage gate (historical, #441) — >60 % of obs flagged new
  * count gate (post-#441 finding) — ≥3 new in one scan

These tests pin both gates and the discovery-step floor.
"""
from __future__ import annotations

from cone_slam.cone_graph_slam_node import cascade_spike_triggered


# ----- Discovery-step floor -----

def test_no_trigger_during_discovery_phase() -> None:
    """Even an obviously bad scan doesn't trigger before step 30 —
    early in the run, EVERY observation is legitimately new."""
    triggered, _ = cascade_spike_triggered(
        n_new=10, total=14, step=15)
    assert triggered is False


def test_step_boundary_floor_is_inclusive() -> None:
    """Step == 30 still in discovery; step == 31 may trigger."""
    triggered30, _ = cascade_spike_triggered(
        n_new=10, total=10, step=30)
    triggered31, _ = cascade_spike_triggered(
        n_new=10, total=10, step=31)
    assert triggered30 is False
    assert triggered31 is True


# ----- Percentage gate (legacy) -----

def test_pct_gate_fires_at_high_new_ratio() -> None:
    """8 of 13 = 62 %, above the 60 % gate; should trigger."""
    triggered, reasons = cascade_spike_triggered(
        n_new=8, total=13, step=100)
    assert triggered is True
    assert any("pct" in r for r in reasons)


def test_pct_gate_silent_below_min_obs() -> None:
    """3 of 4 = 75 % but total < 5 → percentage gate stays silent
    (small samples are statistically unreliable). Count gate may
    still fire — for this test we set count_threshold high so only
    the percentage gate is evaluated."""
    triggered, _ = cascade_spike_triggered(
        n_new=3, total=4, step=100, count_threshold=10)
    assert triggered is False


# ----- Count gate (new, post-#441) -----

def test_count_gate_requires_zero_associations() -> None:
    """Live first-turn signature observed 2026-05-12: 7 new + 5
    associated out of 12 obs. Legitimate cornering discovery, NOT a
    cascade. Must not fire — the assoc>0 requirement on the count
    gate exists specifically to discriminate this case."""
    triggered, _ = cascade_spike_triggered(
        n_new=7, total=12, step=100)
    assert triggered is False


def test_count_gate_fires_on_full_cascade_signature() -> None:
    """The pure-cascade signature observed in the same bag:
    obs=9 new=9 assoc=0 — pose drifted off the map, nothing
    associates, every blob looks new. Must fire on the count gate."""
    triggered, reasons = cascade_spike_triggered(
        n_new=9, total=9, step=100)
    assert triggered is True
    assert any("n_new" in r for r in reasons)


def test_count_gate_at_threshold_boundary() -> None:
    """At assoc=0: n_new = 2 stays silent; n_new = 3 fires (default
    threshold). Percentage gate is silenced here by passing total<5."""
    triggered2, _ = cascade_spike_triggered(
        n_new=2, total=2, step=100)
    triggered3, reasons3 = cascade_spike_triggered(
        n_new=3, total=3, step=100)
    assert triggered2 is False
    assert triggered3 is True
    assert any("n_new≥3" in r for r in reasons3)


def test_count_gate_disabled_when_threshold_zero() -> None:
    """Setting count_threshold=0 disables the count gate completely."""
    # 10/14 = 71 %, so percentage gate still fires — we test that
    # count_threshold=0 doesn't add extra reasons.
    _, reasons = cascade_spike_triggered(
        n_new=10, total=14, step=100, count_threshold=0)
    assert any("pct" in r for r in reasons)
    assert all("n_new" not in r for r in reasons)


# ----- Both gates -----

def test_both_gates_can_fire_at_once() -> None:
    """A scan that exceeds both percentage and count thresholds AND
    has zero associations (n_new == total) should list both reasons."""
    triggered, reasons = cascade_spike_triggered(
        n_new=10, total=10, step=100)
    assert triggered is True
    # 10/10 = 100 % > 60 % ✓
    assert any("pct" in r for r in reasons)
    # 10 ≥ 3 AND assoc == 0 ✓
    assert any("n_new" in r for r in reasons)


def test_percentage_gate_still_fires_with_some_associations() -> None:
    """The percentage gate intentionally does NOT require assoc==0
    — a 9-of-12 (75 %) burst with 3 associations is still cascade-
    like and should be caught by the legacy pct gate."""
    triggered, reasons = cascade_spike_triggered(
        n_new=9, total=12, step=100)
    assert triggered is True
    assert any("pct" in r for r in reasons)
    assert not any("n_new" in r for r in reasons), (
        "count gate must stay silent when assoc>0")


def test_neither_gate_fires_on_healthy_scan() -> None:
    """A normal scan — 0 new, 14 associated — must stay silent."""
    triggered, reasons = cascade_spike_triggered(
        n_new=0, total=14, step=100)
    assert triggered is False
    assert reasons == []
