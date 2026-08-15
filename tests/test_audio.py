# Vixen Intelligence c.2026
"""Decoding: the one stage before the seam, and the one that reads whole survey drives.

Two things are pinned here. The **signal** must not depend on how the file was read — it is
decoded a block at a time, and a downmix that drifted across a block boundary would move every
box the scanners find on that recording. And the **count** must be real: the decode knows how
many frames the header promised and how many it has read, which is what lets `[decode]` draw a
bar from measurement instead of from elapsed time and a guess.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from siarapp.io import audio
from siarapp.io.audio import load_mono, probe, to_mono


def _write(path, data, rate=8000):
    """Write ``(frames, channels)`` float32 to ``path`` and return it as a string."""
    sf.write(str(path), np.asarray(data, dtype=np.float32), rate)
    return str(path)


@pytest.fixture()
def stereo(tmp_path):
    """A stereo recording long enough to cross several read blocks."""
    rng = np.random.default_rng(5)
    return _write(tmp_path / "stereo.wav", rng.standard_normal((7000, 2)) * 0.25)


def test_reading_in_blocks_gives_the_same_signal_as_reading_it_whole(stereo, monkeypatch):
    """A block boundary must be invisible, or the boxes on that recording move."""
    whole = load_mono(stereo)
    monkeypatch.setattr(audio, "_DECODE_BLOCK_FRAMES", 333)  # divides nothing evenly
    awkward = load_mono(stereo)

    np.testing.assert_array_equal(whole.samples, awkward.samples)
    assert whole.samples.dtype == np.float32 and whole.samples.ndim == 1
    assert whole.sample_rate == 8000.0 and whole.channels == 2
    # And it is the mix the browser makes: the average of the channels, not the first of them.
    data, _ = sf.read(stereo, dtype="float32", always_2d=True)
    np.testing.assert_allclose(whole.samples, data.mean(axis=1), rtol=0, atol=1e-6)


@pytest.mark.parametrize("channel", ["mix", "left", "right", "1"])
def test_every_channel_selection_survives_blocking(stereo, monkeypatch, channel):
    whole = load_mono(stereo, channel=channel)
    monkeypatch.setattr(audio, "_DECODE_BLOCK_FRAMES", 333)
    np.testing.assert_array_equal(whole.samples, load_mono(stereo, channel=channel).samples)

    data, _ = sf.read(stereo, dtype="float32", always_2d=True)
    np.testing.assert_allclose(whole.samples, to_mono(data, channel), rtol=0, atol=1e-6)


def test_the_decode_counts_the_frames_it_has_read(stereo, monkeypatch):
    """`[decode]` is measured, not estimated: on a multi-gigabyte recording that is the
    difference between a bar and a guess ten times out."""
    monkeypatch.setattr(audio, "_DECODE_BLOCK_FRAMES", 1000)
    seen: list[tuple[int, int]] = []
    recording = load_mono(stereo, on_progress=lambda done, total: seen.append((done, total)))

    assert len(seen) == 7, "seven blocks of a seven-thousand-frame recording"
    assert [d for d, _ in seen] == sorted(d for d, _ in seen), "frames read only ever rises"
    assert {t for _, t in seen} == {7000}, "the header knows the length before a byte is decoded"
    assert seen[-1] == (7000, 7000)
    assert recording.samples.shape == (7000,)


def test_a_bad_channel_is_the_users_mistake_not_an_unreadable_file(stereo):
    """It is the same for every recording in the folder, so it must not become a manifest row."""
    with pytest.raises(ValueError, match="channel 9"):
        load_mono(stereo, channel="9")


def test_an_unreadable_file_is_an_oserror_the_per_file_loop_can_catch(tmp_path):
    truncated = tmp_path / "truncated.wav"
    truncated.write_bytes(b"RIFF____WAVEfmt ")
    with pytest.raises(OSError, match="could not decode"):
        load_mono(str(truncated))
    assert probe(str(truncated)) is None


def test_a_mono_file_ignores_the_channel_request(tmp_path):
    path = _write(tmp_path / "mono.wav", np.linspace(-0.5, 0.5, 400))
    for channel in ("mix", "left", "right"):
        assert load_mono(path, channel=channel).samples.shape == (400,)
    assert load_mono(path).channels == 1


# -- counting a folder nobody has vouched for -------------------------------------------------


@pytest.fixture()
def survey(tmp_path):
    """Recordings at two levels, one in a hidden tree, and a file that is not audio."""
    (tmp_path / "stationA").mkdir()
    (tmp_path / ".thumbs").mkdir()
    for name in ("a.wav", "b.flac"):
        (tmp_path / name).write_bytes(b"RIFF")
    (tmp_path / "stationA" / "c.wav").write_bytes(b"RIFF")
    (tmp_path / ".thumbs" / "d.wav").write_bytes(b"RIFF")
    (tmp_path / "notes.txt").write_text("not a recording")
    return tmp_path


def test_the_count_agrees_with_the_walk_the_run_will_do(survey):
    """A form field that promised files the runner then skips would be a form field lying."""
    assert audio.count_recordings(str(survey)) == (3, True)
    assert len(audio.find_recordings(str(survey))) == 3


def test_the_count_stops_at_its_limit_and_says_the_figure_is_a_floor(survey):
    assert audio.count_recordings(str(survey), limit=2) == (2, False)


def test_a_spent_budget_comes_back_unknown_rather_than_empty(survey):
    """The difference the whole bound turns on: "not counted" must never read as "nothing here"."""
    assert audio.count_recordings(str(survey), budget_sec=0.0) == (0, False)


def test_one_recording_is_answered_without_walking_anything(survey):
    assert audio.count_recordings(str(survey / "a.wav")) == (1, True)
    assert audio.count_recordings(str(survey / "notes.txt")) == (0, True)


def test_a_walk_that_takes_a_while_says_so_and_a_quick_one_stays_quiet(survey, monkeypatch):
    quick = []
    audio.count_recordings(str(survey), on_progress=quick.append)
    assert quick == [], "a folder of clips must not flash a spinner"

    monkeypatch.setattr(audio, "_COUNT_TICK_SEC", 0.0)
    slow = []
    assert audio.count_recordings(str(survey), on_progress=slow.append) == (3, True)
    assert slow, "a walk in progress reports what it has found so far"
    assert slow == sorted(slow), "the count it reports only ever rises"
