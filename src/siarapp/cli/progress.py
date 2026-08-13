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

__all__ = ["Throughput"]

#: No estimated bar is ever drawn full. A bar that reaches 100% and then keeps going has misled
#: the reader twice; one that waits just short of it has only ever said "nearly", which is true.
#: The same cap :data:`siarapp.cli.format.BAR_CAP` applies to what is printed, kept here as a
#: fraction because this module is where the fraction is decided.
_CAP = 0.99


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

    def stage_seconds(self, stage: str, audio_sec: float) -> float:
        """How long ``stage`` should take on a recording of this length. ``0.0`` if unknown."""
        rate = self.cost_rate()
        if rate <= 0 or audio_sec <= 0 or stage not in PHASES:
            return 0.0
        return audio_sec * rate * self.shares().get(stage, 0.0)

    def stage_fraction(self, stage: str, audio_sec: float, in_stage_sec: float) -> float:
        """How far through its current stage a lane is, ``0.0`` when there is no way to tell.

        Args:
            stage: The stage that lane is in, ``""`` before it has reported one.
            audio_sec: The recording's length.
            in_stage_sec: Wall time since that stage began.

        Returns:
            ``0.0`` to :data:`_CAP`. A stage that overruns its estimate holds at the cap rather
            than rolling over into the next one — the elapsed seconds beside the bar keep moving,
            which says "longer than expected" without the bar having to lie about it.
        """
        expected = self.stage_seconds(stage, audio_sec)
        if expected <= 0:
            return 0.0
        return min(_CAP, max(0.0, in_stage_sec) / expected)

    def scanned_fraction(self, stage: str, audio_sec: float, in_stage_sec: float) -> float:
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
        return self.stage_fraction(stage, audio_sec, in_stage_sec)


def _normalised(shares: dict | None) -> dict:
    """Stage weights as fractions of their own total, or ``{}`` when there is nothing to divide."""
    if not shares:
        return {}
    total = sum(float(shares.get(stage) or 0.0) for stage in PHASES)
    if total <= 0:
        return {}
    return {stage: float(shares.get(stage) or 0.0) / total for stage in PHASES}
