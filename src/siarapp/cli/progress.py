# Vixen Intelligence c.2026
"""How far through a recording a worker is, when the only clock is elapsed time.

A scanner bundle reports nothing from inside ``scan``. So every bar this CLI draws is an estimate
built from two things it *can* know: how long a recording is, and what a second of audio has cost
so far. This module owns that arithmetic, because all three displays
(:class:`~siarapp.cli.commands.ScanReporter`, :class:`~siarapp.cli.commands.WorkerPanel`,
:mod:`siarapp.cli.tui`) draw the same bars and a run whose panel and whose line disagree about
how far in it is has told the reader that neither can be trusted.

Two rules, both learned the hard way from watching real runs:

* **A bar is per stage, not per recording.** The five stages are nothing like equal —
  :data:`~siarapp.io.performance.TYPICAL_SHARES` puts nearly nine tenths of a recording inside
  ``scan`` — so a bar drawn against the whole file reads 99% before the scan has even started and
  then sits there for an hour. Each stage gets its own bar, from zero, against what that stage
  costs; the stage's name is beside it, so what the bar means is on screen next to it.
* **The corpus bar counts the scan.** A recording in ``decode`` has not been scanned in any sense
  worth reporting; one half-way through ``scan`` genuinely is half done, and one that has reached
  ``write`` is finished as far as the algorithm is concerned. Counting it any other way is how a
  bar reads 2% while every worker under it reads 40%.

Nothing here draws or formats: it answers "what fraction" and "how many seconds", and the
displays decide what that looks like.
"""
from __future__ import annotations

from siarapp.io.performance import MAIN_PHASE, PHASES, TYPICAL_SHARES

__all__ = ["Passage", "Throughput"]

#: No estimated bar is ever drawn full. A bar that reaches 100% and then keeps going has misled
#: the reader twice; one that waits just short of it has only ever said "nearly", which is true.
#: The same cap :data:`siarapp.cli.format.BAR_CAP` applies to what is printed, kept here as a
#: fraction because this module is where the fraction is decided.
_CAP = 0.99

#: Where a stage's bar stands when it has taken exactly as long as it was expected to. The rest of
#: the bar belongs to the overrun: past its estimate a stage keeps creeping towards :data:`_CAP`
#: without ever reaching it, so a bad estimate reads as a bar that slows down rather than one that
#: stops. A frozen bar and a hung run look identical, and only one of them is worth panicking over.
_ON_TIME = 0.9

#: Stages whose measured time may be used to size the ones after them.
#:
#: The transform and the scan both walk the same magnitude grid, so one predicts the other across
#: any corpus, any sample rate and any machine: if the FFT on this recording took eight seconds,
#: the scan will take roughly whatever multiple of eight the algorithm's split says. The stages
#: left out are the ones that wait on a disk — reading the audio, copying it, writing a preview.
#: How long those take says more about the page cache than about the work, and a recording read
#: from cache in twenty milliseconds would predict a scan that finishes before it starts.
_CALIBRATORS = ("fft", "scan")


class Throughput:
    """What a second of audio costs, stage by stage, and what that says about a lane.

    Args:
        rate: Wall seconds per second of audio, from the last run of this algorithm on this
            machine (see :func:`siarapp.config.recent_cost_rate`). ``0.0`` when there is no
            history — a first run has nothing to draw with until its first recording lands.
        shares: How that time divided between the stages last time, as a mapping of stage name to
            seconds or to a fraction; only the proportions are used. Falls back to
            :data:`~siarapp.io.performance.TYPICAL_SHARES`.
    """

    __slots__ = ("_prior_rate", "_prior_shares", "_audio", "_wall", "_phases")

    def __init__(self, rate: float = 0.0, shares: dict | None = None) -> None:
        self._prior_rate = max(0.0, float(rate))
        self._prior_shares = _normalised(shares) or dict(TYPICAL_SHARES)
        self._audio = 0.0
        self._wall = 0.0
        self._phases: dict[str, float] = {}

    def learn(self, audio_sec: float, wall_sec: float, phases: dict | None = None) -> None:
        """Fold one finished recording in. Everything measured replaces everything assumed.

        Args:
            audio_sec: The recording's length.
            wall_sec: Wall time one worker spent on it, end to end.
            phases: Its per-stage seconds, as :class:`~siarapp.runner.FileResult` records them.
        """
        if audio_sec <= 0 or wall_sec <= 0:
            return
        self._audio += float(audio_sec)
        self._wall += float(wall_sec)
        for stage, spent in (phases or {}).items():
            self._phases[stage] = self._phases.get(stage, 0.0) + float(spent)

    @property
    def measured(self) -> bool:
        """Whether this run has finished anything of its own yet."""
        return self._audio > 0 and self._wall > 0

    def cost_rate(self) -> float:
        """Wall seconds one worker spends per second of audio, or ``0.0`` when unknowable."""
        return self._wall / self._audio if self.measured else self._prior_rate

    def shares(self) -> dict:
        """How a recording's time divides between the stages, as fractions summing to one."""
        return _normalised(self._phases) or self._prior_shares

    def stage_seconds(self, stage: str, audio_sec: float, spent: dict | None = None) -> float:
        """How long ``stage`` should take on this recording. ``0.0`` when there is no telling.

        Two ways of answering, and the first is far the better one:

        * **From this recording's own completed stages.** Once ``fft`` has taken eight seconds on
          this file, and the split says ``fft`` is a fifteenth of what ``scan`` is, ``scan`` is
          about two minutes — and that holds whatever the sample rate, whatever the machine, and
          whether or not this algorithm has ever run here before. It is the ratios between the
          stages that are stable, not the seconds.
        * **From what a second of audio cost last time**, for the stages that run before anything
          has been measured — the first bar of the first recording, and nothing else.

        Args:
            stage: The stage to size.
            audio_sec: The recording's length.
            spent: Seconds this recording has already spent, by stage, from the stages it has
                finished. This is what makes the estimate self-correcting: a seed that is ten
                times out is corrected by the first stage that completes rather than being
                believed for the whole file.

        Returns:
            Seconds, or ``0.0`` when neither route can answer.
        """
        if stage not in PHASES or audio_sec <= 0:
            return 0.0
        shares = self.shares()
        share = shares.get(stage, 0.0)
        if share <= 0:
            return 0.0

        usable = [k for k in (spent or {}) if k in _CALIBRATORS and shares.get(k, 0.0) > 0]
        measured = sum(float(spent[k]) for k in usable)
        weight = sum(shares[k] for k in usable)
        if measured > 0 and weight > 0:
            return share * measured / weight

        rate = self.cost_rate()
        return audio_sec * rate * share if rate > 0 else 0.0

    def stage_fraction(self, stage: str, audio_sec: float, in_stage_sec: float,
                       spent: dict | None = None) -> float:
        """How far through its current stage a lane is, ``0.0`` when there is no way to tell.

        Args:
            stage: The stage that lane is in, ``""`` before it has reported one.
            audio_sec: The recording's length.
            in_stage_sec: Wall time since that stage began.
            spent: See :meth:`stage_seconds`.

        Returns:
            ``0.0`` to :data:`_CAP`, reaching :data:`_ON_TIME` when the stage has taken exactly
            as long as expected and creeping towards the cap after that. It never stops moving
            and it never fills: an estimate that has been overtaken should say so by slowing
            down, not by freezing, and not by claiming the work is done.
        """
        expected = self.stage_seconds(stage, audio_sec, spent)
        if expected <= 0:
            return 0.0
        over = max(0.0, in_stage_sec) / expected
        if over <= 1.0:
            return _ON_TIME * over
        return _CAP - (_CAP - _ON_TIME) / over

    def scanned_fraction(self, stage: str, audio_sec: float, in_stage_sec: float,
                         spent: dict | None = None) -> float:
        """How much of a recording in flight to count as scanned, for the corpus bar.

        Nothing before the scan starts, the scan's own progress during it, and all of it
        afterwards: by ``write`` the algorithm has seen every frame, and the audio copy that
        follows is not what the reader is waiting on.
        """
        if stage not in PHASES:
            return 0.0
        if PHASES.index(stage) < PHASES.index(MAIN_PHASE):
            return 0.0
        if stage != MAIN_PHASE:
            return _CAP
        return self.stage_fraction(stage, audio_sec, in_stage_sec, spent)


class Passage:
    """One recording's trip through one worker, as a display follows it.

    Held from ``on_start`` to the result, and the reason a lane's bar can be right about a file
    the run has never seen the like of: it remembers how long this recording's finished stages
    actually took, which is what :meth:`Throughput.stage_seconds` calibrates the rest against.

    Attributes:
        index: Its position in the run, for the ``[7/412]`` counter.
        rel_path: Path relative to the scanned folder.
        seconds: Its length in audio.
        size_bytes: What it occupies on disk.
        started_at: When the worker took it.
        stage: The stage it is in now, ``""`` before it has reported one.
        stage_at: When that stage began — the bar starts again from here.
        spent: Seconds each finished stage of this recording took.
    """

    __slots__ = ("index", "rel_path", "seconds", "size_bytes", "started_at", "stage",
                 "stage_at", "spent")

    def __init__(self, index: int, rel_path: str, seconds: float, size_bytes: int,
                 at: float) -> None:
        self.index = int(index)
        self.rel_path = rel_path
        self.seconds = float(seconds)
        self.size_bytes = int(size_bytes)
        self.started_at = self.stage_at = float(at)
        self.stage = ""
        self.spent: dict[str, float] = {}

    def advance(self, stage: str, at: float) -> None:
        """Move on to ``stage``, banking what the one before it cost."""
        if self.stage:
            self.spent[self.stage] = max(0.0, at - self.stage_at)
        self.stage = stage
        self.stage_at = float(at)

    def fraction(self, cost: Throughput, now: float) -> float:
        """How far through its current stage this recording is."""
        return cost.stage_fraction(self.stage, self.seconds, now - self.stage_at, self.spent)

    def scanned(self, cost: Throughput, now: float) -> float:
        """How much of it to count towards the corpus bar."""
        return cost.scanned_fraction(self.stage, self.seconds, now - self.stage_at, self.spent)

    def expected(self, cost: Throughput) -> float:
        """How long its current stage should take, ``0.0`` when there is no telling."""
        return cost.stage_seconds(self.stage, self.seconds, self.spent)


def _normalised(shares: dict | None) -> dict:
    """Stage weights as fractions of their own total, or ``{}`` when there is nothing to divide."""
    if not shares:
        return {}
    total = sum(float(shares.get(stage) or 0.0) for stage in PHASES)
    if total <= 0:
        return {}
    return {stage: float(shares.get(stage) or 0.0) / total for stage in PHASES}
