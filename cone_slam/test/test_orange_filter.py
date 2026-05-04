"""Smoke test for the orange-observation filter.

The cone_graph_slam node drops ORANGE / BIG_ORANGE observations before
they reach data association — see the rationale comment in
`cone_graph_slam_node._on_cones`. Pin that contract at the
list-comprehension level here without spinning up the whole node.
"""
from __future__ import annotations

from cone_slam.color_classifier import ConeColor
from cone_slam.data_association import Observation


def _filter_orange(obs):
    return [o for o in obs
            if o.color not in (ConeColor.ORANGE, ConeColor.BIG_ORANGE)]


def test_filter_drops_orange_observations() -> None:
    obs = [
        Observation(body_x=1.0, body_y=0.0, height=0.3, color=ConeColor.YELLOW),
        Observation(body_x=2.0, body_y=0.0, height=0.3, color=ConeColor.ORANGE),
        Observation(body_x=3.0, body_y=0.0, height=0.3, color=ConeColor.BLUE),
        Observation(body_x=4.0, body_y=0.0, height=0.3, color=ConeColor.BIG_ORANGE),
    ]
    out = _filter_orange(obs)
    assert len(out) == 2
    assert {o.color for o in out} == {ConeColor.YELLOW, ConeColor.BLUE}


def test_filter_preserves_all_non_orange() -> None:
    obs = [
        Observation(body_x=1.0, body_y=0.0, height=0.3, color=ConeColor.YELLOW),
        Observation(body_x=2.0, body_y=0.0, height=0.3, color=ConeColor.BLUE),
    ]
    assert _filter_orange(obs) == obs


def test_filter_handles_empty() -> None:
    assert _filter_orange([]) == []
