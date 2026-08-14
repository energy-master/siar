# Vixen Intelligence c.2026
"""What the bars mean, which is the only thing a live display is for.

The scan is an estimate and can be nothing else — a scanner bundle reports nothing from inside
``scan`` — so most of what is tested here is the arithmetic that keeps an estimate useful rather
than decorative: a bar belongs to a stage, the corpus bar counts the scan, a bar never runs
backwards when the estimate under it is revised, and a stage that can count its own work is not
estimated at all.
"""
from __future__ import annotations

import pytest

from siarapp.cli.progress import Passage, Throughput
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
    # Half way through the scan is half a bar — of the scan's bar. Nine tenths of the bar is
    # the estimate; the last tenth belongs to whatever the estimate got wrong.
    assert cost.stage_fraction("scan", HOUR, 280.0) == pytest.approx(0.45, rel=0.02)
    assert cost.stage_fraction("scan", HOUR, 560.0) == pytest.approx(0.9, rel=0.02)


def test_a_bar_belongs_to_its_stage_and_starts_again_at_each_one():
    """The whole point: 30 seconds in, `fft` is done and `scan` has barely begun."""
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    # 30s into a 30s fft: the estimate is spent. The same 30s counted against the whole file
    # would be 5%, and by the time the scan started the file-wide bar would read 99% — which is
    # the bug this fixes.
    assert cost.stage_fraction("fft", HOUR, 30.0) == pytest.approx(0.9, rel=0.02)
    assert cost.stage_fraction("scan", HOUR, 5.0) < 0.02


def test_a_stage_that_overruns_keeps_creeping_and_never_fills():
    """A frozen bar and a hung run look identical, and only one is worth panicking over."""
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    twice = cost.stage_fraction("scan", HOUR, 1120.0)
    ten_times = cost.stage_fraction("scan", HOUR, 5600.0)
    assert 0.9 < twice < ten_times < 0.99
    assert cost.stage_fraction("scan", HOUR, 99_999_999.0) < 0.99


def test_the_corpus_bar_counts_the_scan_and_nothing_before_it():
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    # Being read off a disk is not progress through the algorithm.
    assert cost.scanned_fraction("decode", HOUR, 5.0) == 0.0
    assert cost.scanned_fraction("fft", HOUR, 25.0) == 0.0
    assert cost.scanned_fraction("scan", HOUR, 280.0) == pytest.approx(0.45, rel=0.02)
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


def test_a_recordings_own_stages_correct_a_seed_that_is_wrong():
    """The estimate that matters is a ratio, and the ratio is measured on the file in hand.

    A prior from a different corpus can be ten times out. Believing it for the whole recording is
    what put every worker bar at 99% within seconds of the run starting; one finished stage of
    this recording is enough to fix it, and no history at all is needed after that.
    """
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    assert cost.stage_seconds("scan", HOUR) == pytest.approx(560.0, rel=0.01)
    # This file's fft actually took ten times what the seed predicted, so its scan will too.
    spent = {"decode": 60.0, "fft": 300.0}
    assert cost.stage_seconds("scan", HOUR, spent) == pytest.approx(5600.0, rel=0.02)
    assert cost.stage_fraction("scan", HOUR, 560.0, spent) < 0.1


def test_a_bar_appears_with_no_history_at_all_once_one_stage_has_finished():
    """A first run of a new algorithm: no rate to be had, but the ratios still hold."""
    cost = Throughput()
    assert cost.stage_seconds("scan", HOUR) == 0.0, "nothing measured, nothing drawn"
    # 20 seconds of fft on this file, and the default split says scan is about fifteen times it.
    expected = cost.stage_seconds("scan", HOUR, {"fft": 20.0})
    assert expected == pytest.approx(20.0 * TYPICAL_SHARES["scan"] / TYPICAL_SHARES["fft"], rel=0.01)
    assert cost.stage_fraction("scan", HOUR, expected / 2, {"fft": 20.0}) == pytest.approx(0.45,
                                                                                          rel=0.02)


def test_a_passage_banks_each_stage_as_it_leaves_it():
    passage = Passage(1, "station/0410.wav", HOUR, 3_500_000_000, at=1000.0)
    assert passage.stage == "" and passage.spent == {}
    passage.advance("decode", 1000.0)
    passage.advance("fft", 1012.0)
    passage.advance("scan", 1042.0)
    assert passage.spent == {"decode": 12.0, "fft": 30.0}
    assert passage.stage == "scan" and passage.stage_at == 1042.0

    cost = Throughput()
    # Sized off this recording's own stages, with nothing else to go on anywhere.
    assert passage.expected(cost) > 0
    assert passage.fraction(cost, 1042.0) == 0.0


def test_a_stage_that_counts_its_own_work_is_not_estimated():
    """`decode` and `fft` know how many frames they have done. Nothing needs guessing for them."""
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    passage = Passage(1, "station/0410.wav", HOUR, 3_500_000_000, at=1000.0)
    passage.report("decode", 0.0, 1000.0)
    assert not passage.counted()

    passage.report("decode", 0.5, 1001.0)
    assert passage.counted()
    # Half the frames read is half the bar, whatever the estimate would have said and whatever
    # the clock says — this is the one number on screen that is not an opinion.
    assert passage.fraction(cost, 1001.0) == pytest.approx(0.495, rel=0.02)
    # ...and a stage that has been quicker than the whole seed said still reads as half done.
    assert passage.fraction(cost, 9999.0) == pytest.approx(0.495, rel=0.02)


def test_counted_work_is_drawn_with_no_history_and_no_finished_stage():
    """The first stage of the first recording of a first run used to have no bar at all."""
    cost = Throughput()
    passage = Passage(1, "station/0410.wav", HOUR, 3_500_000_000, at=1000.0)
    passage.report("decode", 0.0, 1000.0)
    assert not passage.drawable(cost), "nothing measured, nothing known: elapsed seconds only"
    passage.report("decode", 0.25, 1030.0)
    assert passage.drawable(cost) and passage.fraction(cost, 1030.0) > 0.2


def test_a_new_stage_starts_the_bar_again_and_forgets_what_was_counted():
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    passage = Passage(1, "station/0410.wav", HOUR, 3_500_000_000, at=1000.0)
    passage.report("decode", 0.0, 1000.0)
    passage.report("decode", 0.9, 1006.0)
    passage.report("fft", 0.0, 1006.0)
    assert passage.stage == "fft" and not passage.counted()
    assert passage.spent == {"decode": 6.0}
    assert passage.fraction(cost, 1006.0) == pytest.approx(0.0, abs=0.01)


def test_a_bar_holds_rather_than_going_backwards_when_the_estimate_is_revised():
    """The first recording to land rewrites every lane's estimate. None of them may drop.

    A bar that falls says work has been lost, which is never what happened: what changed is the
    guess about how much is left. It holds until the new estimate catches it up, which reads as
    a pause — and the elapsed seconds beside it keep moving, so the run still looks alive.
    """
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    passage = Passage(1, "station/0410.wav", HOUR, 3_500_000_000, at=1000.0)
    passage.report("scan", 0.0, 1000.0)
    half = passage.fraction(cost, 1280.0)
    assert half == pytest.approx(0.45, rel=0.02)

    # A recording lands and says this algorithm's scan is four times what the seed thought.
    cost.learn(HOUR, 2400.0, {"decode": 6.0, "fft": 30.0, "scan": 2360.0})
    assert cost.stage_fraction("scan", HOUR, 280.0) < half, "the estimate itself did drop"
    assert passage.fraction(cost, 1280.0) == pytest.approx(half), "the bar did not"
    # And it goes on from where it held, rather than starting the climb again.
    assert passage.fraction(cost, 2400.0) > half


def test_the_corpus_bar_holds_with_the_lanes_it_is_built_from():
    """A corpus bar assembled from lanes that go backwards goes backwards too."""
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    passage = Passage(1, "station/0410.wav", HOUR, 3_500_000_000, at=1000.0)
    passage.report("scan", 0.0, 1000.0)
    scanned = passage.scanned(cost, 1280.0)
    cost.learn(HOUR, 2400.0, {"decode": 6.0, "fft": 30.0, "scan": 2360.0})
    assert passage.scanned(cost, 1280.0) == pytest.approx(scanned)
    # The stages after the scan still count in full, ratchet or no ratchet.
    passage.report("write", 0.0, 1400.0)
    assert passage.scanned(cost, 1400.0) == pytest.approx(0.99)


def test_only_the_compute_stages_may_size_the_ones_after_them():
    """A recording read out of the page cache must not predict a scan that finishes instantly."""
    cost = Throughput(rate=600.0 / HOUR, shares=PHASES_OF_AN_HOUR)
    seeded = cost.stage_seconds("scan", HOUR)
    # 20 ms of "decode" because the file was already in memory. It says nothing about the scan.
    assert cost.stage_seconds("scan", HOUR, {"decode": 0.02}) == pytest.approx(seeded, rel=0.01)
    # The transform walks the same grid the scan does, so that one counts: an FFT that took
    # twice its estimate means a scan that will take about twice its own.
    assert cost.stage_seconds("scan", HOUR, {"decode": 0.02, "fft": 60.0}) == pytest.approx(
        2 * seeded, rel=0.02)
