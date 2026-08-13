# Vixen Intelligence c.2026
"""What the bars mean, which is the only thing a live display is for.

Every one of them is an estimate — a scanner bundle reports nothing from inside ``scan`` — so
the tests here are about the two rules that keep an estimate useful rather than decorative: a bar
belongs to a stage, and the corpus bar counts the scan.
"""
from __future__ import annotations

import pytest

from siarapp.cli.progress import Throughput
from siarapp.io.performance import TYPICAL_SHARES

#: An hour of audio that cost ten minutes of one worker, nearly all of it in the scan.
HOUR = 3600.0
PHASES_OF_AN_HOUR = {"decode": 6.0, "fft": 30.0, "scan": 560.0, "write": 2.0, "thumbnail": 2.0}


def test_nothing_is_estimated_without_something_to_estimate_from():
    """A first run of a new algorithm has no history and no results: no bar, not a wrong one."""
    cost = Throughput()
    assert cost.cost_rate() == 0.0
    assert cost.stage_seconds("scan", HOUR) == 0.0
    assert cost.stage_fraction("scan", HOUR, 120.0) == 0.0


def test_the_last_run_on_this_machine_seeds_the_first_bar():
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    # Ten minutes of work on an hour of audio, 93% of it scan.
    assert cost.stage_seconds("scan", HOUR) == pytest.approx(560.0, rel=0.01)
    assert cost.stage_seconds("fft", HOUR) == pytest.approx(30.0, rel=0.01)
    # Half way through the scan is half a bar — of the scan's bar.
    assert cost.stage_fraction("scan", HOUR, 280.0) == pytest.approx(0.5, rel=0.01)


def test_a_bar_belongs_to_its_stage_and_starts_again_at_each_one():
    """The whole point: 40 seconds in, `fft` is finished and `scan` has barely begun."""
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    # 30s into a 30s fft: full. The same 30s counted against the whole file would be 5%, and by
    # the time the scan started the file-wide bar would read 99% — which is the bug this fixes.
    assert cost.stage_fraction("fft", HOUR, 30.0) == pytest.approx(0.99)
    assert cost.stage_fraction("scan", HOUR, 5.0) < 0.02


def test_a_stage_that_overruns_holds_at_the_cap_rather_than_rolling_over():
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    assert cost.stage_fraction("scan", HOUR, 560.0) == pytest.approx(0.99)
    assert cost.stage_fraction("scan", HOUR, 99_999.0) == pytest.approx(0.99)


def test_the_corpus_bar_counts_the_scan_and_nothing_before_it():
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    # Being read off a disk is not progress through the algorithm.
    assert cost.scanned_fraction("decode", HOUR, 5.0) == 0.0
    assert cost.scanned_fraction("fft", HOUR, 25.0) == 0.0
    assert cost.scanned_fraction("scan", HOUR, 280.0) == pytest.approx(0.5, rel=0.01)
    # Past the scan the algorithm has seen every frame; the audio copy is not what anyone waits on.
    assert cost.scanned_fraction("write", HOUR, 0.1) == pytest.approx(0.99)
    assert cost.scanned_fraction("thumbnail", HOUR, 0.1) == pytest.approx(0.99)


def test_measurement_replaces_the_seed():
    """A prior is a starting point. One finished recording outranks it."""
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    cost.learn(HOUR, 120.0, {"decode": 2.0, "fft": 8.0, "scan": 108.0})
    assert cost.measured
    assert cost.cost_rate() == pytest.approx(120.0 / HOUR)
    assert cost.shares()["scan"] == pytest.approx(108.0 / 118.0, rel=0.01)
    assert cost.stage_seconds("scan", HOUR) == pytest.approx(109.8, rel=0.02)


def test_a_recording_that_failed_teaches_nothing():
    cost = Throughput()
    cost.learn(0.0, 4.0, {"decode": 4.0})
    cost.learn(HOUR, 0.0, {})
    assert not cost.measured
    assert cost.cost_rate() == 0.0


def test_the_default_split_is_mostly_scan():
    """Because the alternative — treating a recording as uniform — is wrong by a factor of ten."""
    assert sum(TYPICAL_SHARES.values()) == pytest.approx(1.0)
    assert TYPICAL_SHARES["scan"] > 0.8
    cost = Throughput(rate=600.0 / HOUR)
    assert cost.stage_seconds("scan", HOUR) > 8 * cost.stage_seconds("fft", HOUR)
