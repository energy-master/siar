# Vixen Intelligence c.2026
"""One index of everything this machine can actually run.

Two things arrive by completely different routes and end up doing the same job. A **downloaded**
model is a PyArmor bundle pulled from IDent Dynamics and unpacked under
``~/.siar-app/algorithms``; a **built** model is one the operator evolved themselves with
``siar-build``, which leaves a plain Python package on disk and a row about it in its own index at
``~/.siar-build/models.db``. Both are handed to :func:`siarapp.runner.run_folder` and neither
knows about the other.

Which is exactly why they should be listed together. The question in front of somebody with a
folder of recordings is "what can I run over this", and answering it out of two commands and one
sibling tool's database is how a model that took an afternoon to build gets forgotten about.
:func:`library` answers it once.

The siar-build side is read **read-only, defensively, and never created**. That file belongs to
another program: it is opened with ``mode=ro`` so a missing one stays missing rather than becoming
an empty database this package invented, every column is fetched by name through :func:`_get` so a
newer or older schema is a missing field rather than an exception, and any failure at all means
"nothing built here" — a listing, not an error. Deleting ``models.db`` costs a listing and no
models, and this module must behave the same way about it.

Nothing here loads an algorithm, imports a bundle or touches the network. It reads manifests and
one SQLite file, so a library screen opens instantly on a vessel with no link.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

from siarapp.loader import installed_algorithms

__all__ = [
    "BUILD_HOME_ENV",
    "Model",
    "Program",
    "build_db_path",
    "built_models",
    "downloaded_models",
    "feature_usage",
    "library",
]

#: Moves siar-build's workspace, the way ``$SIAR_APP_HOME`` moves this one. Honoured here so a
#: containerised or shared install that pointed siar-build somewhere writable is still readable
#: from the library rather than silently empty.
BUILD_HOME_ENV = "SIAR_BUILD_HOME"

#: siar-build's index file, inside that workspace.
BUILD_DB_NAME = "models.db"

#: Builds read from that index. Far more than anyone browses, far short of loading a machine's
#: whole history into a screen that shows six rows at a time.
BUILD_LIMIT = 200

#: What a model of each kind is called on screen and in a message. The distinction is not
#: cosmetic: one of them can be re-downloaded and the other exists nowhere but this disk.
DOWNLOADED = "downloaded"
BUILT = "built"


class Program:
    """One program out of a siar-build search — a bot, and what it reads.

    A build produces a population, not an answer. The champion is whichever program selection
    settled on and is the one that was calibrated, packaged and can be run; the runners-up are
    recorded in siar-build's index and nowhere else, because an uncalibrated program cannot be
    packaged or run. Both are listed, because "what else did the search find" is a question about
    the corpus that only this table can answer.

    Attributes:
        rank: 0 for the champion, then the runners-up in order.
        kind: ``"champion"`` or ``"runner_up"``.
        fitness: The objective's score for this program.
        threshold: Where the decision was calibrated, or ``None`` on a runner-up — which is
            precisely what makes it unrunnable.
        polarity: ``+1`` or ``-1``: which side of the threshold is the detection.
        n_nodes: Size of the expression, and
        depth: how deep it nests — the two numbers that say whether it is readable.
        features: The feature names the expression reads, in the order the index recorded them.
        infix: The expression itself.
        saved_path: The model document this program was written to, or ``""`` for a runner-up.
    """

    __slots__ = ("rank", "kind", "fitness", "threshold", "polarity", "n_nodes", "depth",
                 "features", "infix", "saved_path")

    def __init__(self, *, rank: int = 0, kind: str = "", fitness: float | None = None,
                 threshold: float | None = None, polarity: int = 1, n_nodes: int = 0,
                 depth: int = 0, features: tuple[str, ...] = (), infix: str = "",
                 saved_path: str = "") -> None:
        self.rank = int(rank)
        self.kind = str(kind or "")
        self.fitness = fitness
        self.threshold = threshold
        self.polarity = int(polarity or 1)
        self.n_nodes = int(n_nodes or 0)
        self.depth = int(depth or 0)
        self.features = tuple(features)
        self.infix = str(infix or "")
        self.saved_path = str(saved_path or "")

    @property
    def calibrated(self) -> bool:
        """Whether this program carries a threshold, and so could be run at all."""
        return self.threshold is not None


class Model:
    """One thing this machine can run, whichever way it got here.

    Attributes:
        source: :data:`DOWNLOADED` or :data:`BUILT`.
        slug: What the model is called — the name a sidecar files its boxes under.
        title: A human title where one is recorded, else ``""``.
        version: The bundle's version, or the siar-build that produced the model.
        platform: The build tag a downloaded bundle runs on. ``"source"`` for a built model:
            it is plain Python and runs wherever this CLI does.
        path: The directory to run — an unpacked bundle, or a generated package.
        size_bytes: What that directory occupies on disk.
        stamped_at: Epoch seconds — downloaded when, or built when.
        runnable: Whether this machine can run it as it stands.
        note: Why not, when it cannot. Empty otherwise.
        detail: Facts for the panel, by key. A bundle's manifest fields, or a build's row.
        programs: The bots behind a built model, champion first. Empty for a downloaded one,
            whose insides are deliberately not knowable from here.
    """

    __slots__ = ("source", "slug", "title", "version", "platform", "path", "size_bytes",
                 "stamped_at", "runnable", "note", "detail", "programs")

    def __init__(self, *, source: str, slug: str, title: str = "", version: str = "",
                 platform: str = "", path: str = "", size_bytes: int = 0, stamped_at: int = 0,
                 runnable: bool = True, note: str = "", detail: dict | None = None,
                 programs: list[Program] | None = None) -> None:
        self.source = source
        self.slug = slug
        self.title = title
        self.version = version
        self.platform = platform
        self.path = path
        self.size_bytes = int(size_bytes)
        self.stamped_at = int(stamped_at)
        self.runnable = bool(runnable)
        self.note = note
        self.detail = dict(detail or {})
        self.programs = list(programs or [])

    @property
    def built(self) -> bool:
        """Whether this model was built here, rather than downloaded."""
        return self.source == BUILT

    @property
    def features(self) -> tuple[str, ...]:
        """Every feature any of this model's programs reads, in first-seen order.

        Order of appearance rather than sorted: the champion is first in ``programs``, so the
        features it reads lead the list, and what a runner-up brought in follows.
        """
        seen: dict[str, None] = {}
        for program in self.programs:
            for name in program.features:
                seen.setdefault(name, None)
        return tuple(seen)


def downloaded_models() -> list[Model]:
    """The algorithm bundles unpacked on this machine, newest download first.

    Reads manifests and stats trees — :func:`siarapp.loader.installed_algorithms` does the work —
    and imports nothing. A bundle built for another platform is listed with the reason it cannot
    run here rather than hidden: "why is it not in the list" is a worse question than "why does
    this one say no".

    Returns:
        One :class:`Model` per unpacked build.
    """
    models = []
    for row in installed_algorithms():
        runnable = bool(row.get("runnable"))
        models.append(Model(
            source=DOWNLOADED,
            slug=str(row.get("slug") or ""),
            title=str(row.get("title") or ""),
            version=str(row.get("version") or ""),
            platform=str(row.get("platform") or ""),
            path=str(row.get("root") or ""),
            size_bytes=int(row.get("bytes") or 0),
            stamped_at=int(row.get("downloaded_at") or 0),
            runnable=runnable,
            note="" if runnable else f"built for {row.get('platform')}, not this machine",
            detail={"family": str(row.get("family") or "")},
        ))
    return models


def build_db_path() -> str:
    """Where siar-build keeps its index on this machine.

    Returns:
        ``$SIAR_BUILD_HOME/models.db`` if that variable is set, else
        ``~/.siar-build/models.db``. The file need not exist; nothing here creates it.
    """
    override = os.environ.get(BUILD_HOME_ENV)
    root = (os.path.expanduser(override) if override
            else os.path.join(os.path.expanduser("~"), ".siar-build"))
    return os.path.join(root, BUILD_DB_NAME)


def _connect(path: str) -> sqlite3.Connection | None:
    """Open siar-build's index read-only, or ``None`` when there is nothing to open.

    ``mode=ro`` rather than a plain connect, which would *create* an empty database at that path
    — leaving this package's footprint inside another program's workspace, and making "nothing
    built yet" indistinguishable from "built, then the file was lost".
    """
    if not os.path.isfile(path):
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _get(row: sqlite3.Row, key: str, default=None):
    """One column, or ``default`` when this schema has not got it.

    The index belongs to a program that versions separately from this one. A column added or
    dropped there must cost a blank field on a screen, never a traceback in a listing.
    """
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    return default if value is None else value


def _epoch(stamp: str) -> int:
    """An ISO-8601 stamp as epoch seconds, or ``0`` if it will not parse."""
    try:
        return int(datetime.fromisoformat(str(stamp)).timestamp())
    except (TypeError, ValueError):
        return 0


def _features(raw) -> tuple[str, ...]:
    """The ``features_used`` column — a JSON array — as names."""
    try:
        loaded = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return ()
    return tuple(str(name) for name in loaded) if isinstance(loaded, list) else ()


def _dir_bytes(root: str) -> int:
    """What a generated package occupies on disk, or ``0`` if it cannot be walked."""
    total = 0
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def _programs(conn: sqlite3.Connection, build_id: int) -> list[Program]:
    """One build's programs, champion first. An unreadable table is no programs."""
    try:
        rows = conn.execute(
            "SELECT * FROM programs WHERE build_id = ? ORDER BY rank", (build_id,)
        ).fetchall()
    except sqlite3.Error:
        return []
    return [
        Program(
            rank=_get(row, "rank", 0),
            kind=_get(row, "kind", ""),
            fitness=_get(row, "fitness"),
            threshold=_get(row, "threshold"),
            polarity=_get(row, "polarity", 1),
            n_nodes=_get(row, "n_nodes", 0),
            depth=_get(row, "depth", 0),
            features=_features(_get(row, "features_used", "[]")),
            infix=_get(row, "infix", ""),
            saved_path=_get(row, "saved_path", ""),
        )
        for row in rows
    ]


#: Columns copied from a build row into :attr:`Model.detail`. Everything the library screen says
#: about a build comes from this list, so adding a fact to the panel is adding a name here.
_BUILD_DETAIL = (
    "id", "target", "created_at", "input_dir", "output_dir", "package_dir", "objective",
    "selection", "pop_size", "generations", "seed", "sample_rate", "n_fft", "hop", "n_bins",
    "fmin_hz", "fmax_hz", "held_out_auc", "held_out_auc_recording", "in_corpus_auc", "train_auc",
    "null_auc", "suspect", "parity_ok", "parity_max_delta", "test_recordings", "seconds",
    "stopped_at", "reason", "notes",
)


def built_models(db_path: str | None = None, *, limit: int = BUILD_LIMIT) -> list[Model]:
    """The models ``siar-build`` has produced on this machine, newest first.

    A build is listed whether or not it finished: one that stopped at a gate is a row somebody
    will come looking for, and a listing of successes only is a highlight reel. What it cannot do
    is claim to be runnable — that needs a packaged directory still on disk, and
    :attr:`Model.note` says which half is missing when it is not there.

    Args:
        db_path: The index to read. Defaults to :func:`build_db_path`.
        limit: Most recent builds to read.

    Returns:
        One :class:`Model` per build, each carrying its programs. Empty when siar-build has never
        run here, when its index cannot be read, or when it is a schema this cannot make sense
        of — all three of which mean the same thing to the reader.
    """
    conn = _connect(db_path or build_db_path())
    if conn is None:
        return []
    try:
        try:
            rows = conn.execute(
                "SELECT * FROM builds ORDER BY created_at DESC, id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        except sqlite3.Error:
            return []

        models = []
        for row in rows:
            package = str(_get(row, "package_dir", "") or "")
            stopped = str(_get(row, "stopped_at", "") or "")
            runnable = bool(package) and os.path.isdir(package)
            if runnable:
                note = ""
            elif not package:
                note = (f"stopped at {stopped} — nothing packaged" if stopped
                        else "not packaged — run `siar-build package` for this build")
            else:
                note = f"packaged at {package}, which is no longer there"
            models.append(Model(
                source=BUILT,
                slug=str(_get(row, "slug", "") or _get(row, "target", "") or "model"),
                title=str(_get(row, "target", "") or ""),
                version=str(_get(row, "siarbuild_version", "") or ""),
                platform="source",
                path=package,
                size_bytes=_dir_bytes(package) if runnable else 0,
                stamped_at=_epoch(_get(row, "created_at", "")),
                runnable=runnable,
                note=note,
                detail={name: _get(row, name) for name in _BUILD_DETAIL},
                programs=_programs(conn, int(_get(row, "id", 0) or 0)),
            ))
        return models
    finally:
        conn.close()


def library(*, db_path: str | None = None) -> list[Model]:
    """Everything runnable on this machine: downloaded bundles, then models built here.

    Grouped rather than interleaved by date. They are different kinds of thing — one can be
    fetched again from the server and the other exists nowhere but this disk — and a list that
    sorted them together would make that difference invisible at exactly the moment somebody is
    choosing between them.

    Args:
        db_path: siar-build's index, for a test or a non-standard workspace.

    Returns:
        The models, downloaded first, each group newest first.
    """
    return downloaded_models() + built_models(db_path)


def feature_usage(models: list[Model]) -> list[tuple[str, int]]:
    """Which features the bots in these models read, most used first.

    Across models on purpose. A feature that four searches over four corpora all settled on is
    telling you something about the audio rather than about any one build, and per-model lists
    cannot show that.

    Args:
        models: What :func:`library` returned, or any subset.

    Returns:
        ``(feature name, how many programs read it)``, most used first, then alphabetically so
        the order is stable between two calls that tie.
    """
    counts: dict[str, int] = {}
    for model in models:
        for program in model.programs:
            for name in program.features:
                counts[name] = counts.get(name, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
