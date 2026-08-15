# Vixen Intelligence c.2026
"""The metrics table printed at the end of a run.

Worth pinning because the realtime factor is the one number a user plans a survey around, and
the way it can be wrong is silent: count the audio a ``--resume`` run skipped and the table
reports a machine twice as fast as the one on the desk.
"""
from __future__ import annotations

from siarapp.cli.commands import metric_rows, _print_summary
from siarapp.cli.format import factor_text
from siarapp.io.performance import realtime_factor


def _manifest(rows, *, elapsed_sec=10.0, workers=1, structures=0, shapes=None, phases=None):
    """A run manifest with just the fields the summary reads."""
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    return {
        "files": len(rows),
        "by_status": by_status,
        "elapsed_sec": elapsed_sec,
        "workers": workers,
        "structures": structures,
        "shapes": shapes or {},
        "audio_sec": round(sum(r["duration_sec"] for r in rows), 2),
        "phases": phases or {},
        "manifest": rows,
    }


def _row(path, status, duration_sec=0.0):
    return {"path": path, "status": status, "duration_sec": duration_sec, "elapsed_sec": 0.1}


def _value(rows, label):
    """The value cell for one metric label, or None if the row was not rendered."""
    for cell_label, value, _share in rows:
        if cell_label == label:
            return value
    return None


def _share(rows, label):
    """The share cell for one metric label."""
    for cell_label, _value, share in rows:
        if cell_label == label:
            return share
    return None


def test_realtime_counts_only_audio_this_run_scanned():
    manifest = _manifest(
        [_row("a.wav", "scanned", 60.0), _row("b.wav", "skipped")],
        elapsed_sec=10.0,
    )
    rows = metric_rows(manifest)
    assert _value(rows, "audio scanned") == "1.0 min"
    assert _value(rows, "realtime") == "6.00x"
    assert _value(rows, "skipped (resume)") == "1"


def test_outcome_rows_appear_only_when_they_happened():
    rows = metric_rows(_manifest([_row("a.wav", "scanned", 30.0)]))
    labels = [label for label, _value, _share in rows]
    assert "scanned" in labels
    assert "skipped (resume)" not in labels
    assert "too short" not in labels
    assert "errors" not in labels


def test_a_parallel_run_states_the_factor_per_worker_as_well():
    manifest = _manifest(
        [_row(f"{i}.wav", "scanned", 100.0) for i in range(4)],
        elapsed_sec=50.0,
        workers=4,
    )
    rows = metric_rows(manifest)
    assert _value(rows, "realtime") == "8.00x"
    # 400s of audio on 4 workers in 50s wall: each worker managed twice realtime, not eight times.
    assert _value(rows, "realtime per worker") == "2.00x"


def test_counts_are_grouped_for_a_corpus_sized_run():
    manifest = _manifest(
        [_row(f"{i}.wav", "scanned", 1.0) for i in range(3)],
        structures=9043,
    )
    rows = metric_rows(manifest)
    assert _value(rows, "structures") == "9,043"


def test_a_serial_runs_phases_account_for_its_wall_time():
    manifest = _manifest(
        [_row("a.wav", "scanned", 60.0)],
        elapsed_sec=10.0,
        phases={"decode": 1.0, "fft": 2.0, "scan": 6.0},
    )
    rows = metric_rows(manifest)
    assert _value(rows, "  scan") == "6.0 s"
    assert _share(rows, "  scan") == "60%"
    # The second the stages do not claim is named rather than left for the reader to notice.
    assert _value(rows, "  overhead") == "1.0 s"
    assert _share(rows, "  overhead") == "10%"


def test_a_parallel_runs_phases_are_worker_time_not_wall_time():
    manifest = _manifest(
        [_row(f"{i}.wav", "scanned", 100.0) for i in range(4)],
        elapsed_sec=50.0,
        workers=4,
        phases={"fft": 40.0, "scan": 120.0},
    )
    rows = metric_rows(manifest)
    # 160 worker seconds inside a 50-second run: totalled under their own heading, and shared out
    # against that rather than against the wall clock, which would give 320%.
    assert _value(rows, "worker time") == "2.7 min"
    assert _share(rows, "  scan") == "75%"
    assert _value(rows, "  overhead") is None


def test_phases_are_listed_in_the_order_they_happen():
    manifest = _manifest(
        [_row("a.wav", "scanned", 60.0)],
        phases={"thumbnail": 1.0, "decode": 1.0, "scan": 1.0, "fft": 1.0, "write": 1.0},
    )
    listed = [label.strip() for label, _v, _s in metric_rows(manifest) if label.startswith("  ")]
    assert listed[:5] == ["decode", "fft", "scan", "write", "thumbnail"]


def test_a_run_that_measured_nothing_shows_no_breakdown():
    rows = metric_rows(_manifest([_row("a.wav", "scanned", 60.0)]))
    assert not [label for label, _v, _s in rows if label.startswith("  ")]


def test_a_run_with_no_audio_states_no_speed():
    assert factor_text(realtime_factor(0.0, 12.0)) == "—"
    assert factor_text(realtime_factor(5.0, 0.0)) == "—"


def test_the_factor_keeps_the_digit_that_matters():
    assert factor_text(0.354) == "0.35x"
    assert factor_text(9.99) == "9.99x"
    assert factor_text(38.14) == "38.1x"


def test_the_summary_prints_both_tables(capsys):
    manifest = _manifest(
        [_row("a.wav", "scanned", 60.0)],
        structures=3,
        shapes={"click": 2, "tonal": 1},
    )
    _print_summary(manifest, "/tmp/out")
    text = capsys.readouterr().out
    assert "METRIC" in text and "realtime" in text
    assert "STRUCTURE" in text and "click" in text
    assert "/tmp/out" in text


def test_a_fully_resumed_run_says_there_was_nothing_to_do(capsys):
    _print_summary(_manifest([_row("a.wav", "skipped")]), "/tmp/out")
    text = capsys.readouterr().out
    assert "Nothing to do" in text
    assert "METRIC" not in text
