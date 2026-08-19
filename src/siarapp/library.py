# Vixen Intelligence c.2026
"""One index of everything this machine can actually run.

Four things arrive by completely different routes and end up doing the same job. A
**downloaded** model is a PyArmor bundle pulled from IDent Dynamics and unpacked under
``~/.siar-app/algorithms``; a **built** model is one the operator evolved themselves with
``siar-build``, which leaves a plain Python package on disk and a row about it in its own index at
``~/.siar-build/models.db``; an **imported** model is one built on another machine and carried
here as a bundle by :mod:`siarapp.transfer`, which unpacks it under ``~/.siar-app/models`` with
the build row that travelled with it; a **published** model is one somebody else bred and
published to the installation, fetched by :mod:`siarapp.published` into ``~/.siar-app/published``
because this account was granted it. All four are handed to :func:`siarapp.runner.run_folder`
and none of them knows about the others.

Which is exactly why they should be listed together. The question in front of somebody with a
folder of recordings is "what can I run over this", and answering it out of three commands and one
sibling tool's database is how a model that took an afternoon to build gets forgotten about.
:func:`library` answers it once.

The siar-build side is read **read-only, defensively, and never created**. That file belongs to
another program: it is opened with ``mode=ro`` so a missing one stays missing rather than becoming
an empty database this package invented, every column is fetched by name through :func:`_get` so a
newer or older schema is a missing field rather than an exception, and any failure at all means
"nothing built here" — a listing, not an error. Deleting ``models.db`` costs a listing and no
models, and this module must behave the same way about it.

Nothing here loads an algorithm, imports a bundle or touches the network — a published model is
listed from what was written beside it when it was fetched, never by asking the installation who
it belongs to now. It reads manifests and one SQLite file, so a library screen opens instantly on
a vessel with no link.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

from siarapp.loader import installed_algorithms
from siarapp.published import installed_published
from siarapp.transfer import installed_bundles

__all__ = [
    "BUILD_HOME_ENV",
    "PUBLISHED",
    "Model",
    "Program",
    "build_db_path",
    "SOCIETY",
    "built_models",
    "downloaded_models",
    "feature_usage",
    "imported_models",
    "library",
    "local_model",
    "published_models",
    "society_models",
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
#: cosmetic: one of them can be re-downloaded, and the other two exist nowhere but this disk and
#: whatever machine they were carried from.
DOWNLOADED = "downloaded"
BUILT = "built"
IMPORTED = "imported"

#: A model somebody else bred and published to the installation, which this account was granted
#: and this machine has fetched. Its own kind rather than a downloaded one, because what can be
#: re-fetched depends on a grant somebody else can withdraw, and because — unlike a bundle — it is
#: a readable package whose members can be shown.
PUBLISHED = "published"

#: A **society**: many bots for one target, voting. siar-build breeds these continuously and
#: republishes as its leaderboard moves, so unlike the others what this slug names is not a
#: fixed program — it is whichever twenty bots were best when it last published. Listed as its own
#: kind rather than as a build because it *is* a different thing to reason about: a downloaded
#: bundle can be re-fetched, a built model is one search's answer, and a society is a population
#: still being bred.
SOCIETY = "society"

#: The sources whose models are a package on this disk, loaded with
#: :func:`siarapp.loader.load_local` rather than imported out of an obfuscated bundle. Built here,
#: carried here or fetched here — from the loader's point of view those are the same thing.
LOCAL_SOURCES = (BUILT, IMPORTED, SOCIETY, PUBLISHED)

#: The sources a model may be exported from. Deliberately not :data:`LOCAL_SOURCES`: a published
#: model is a readable package like the rest, but it is somebody else's, handed to this account by
#: a grant. Passing it on as a file would be this machine deciding who else may run it.
PORTABLE_SOURCES = (BUILT, IMPORTED, SOCIETY)

#: The family every siar-build society package declares. What tells a fetched model that it is a
#: society — many bots voting — rather than one program, since the row that says so is on the
#: installation and the manifest is what travelled.
SOCIETY_FAMILY = "brahma_society"


class Program:
    """One program out of a siar-build search — a bot, and what it reads.

    A build produces a population, not an answer. The champion is whichever program selection
    settled on and is the one that was calibrated, packaged and can be run; the runners-up are
    recorded in siar-build's index and nowhere else, because an uncalibrated program cannot be
    packaged or run. Both are listed, because "what else did the search find" is a question about
    the corpus that only this table can answer.

    Attributes:
        rank: 0 for the champion, then the runners-up in order.
        name: What this bot is called — ``<target>_<tag>_vxbot``. The champion's is the model's
            name too, because they are the same thing: the champion *is* what was packaged.
            ``""`` for a bot from an index written before siar-build named them, where a rank in
            a listing was all a runner-up ever had.
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

    __slots__ = ("rank", "name", "kind", "fitness", "threshold", "polarity", "n_nodes", "depth",
                 "features", "infix", "saved_path")

    def __init__(self, *, rank: int = 0, name: str = "", kind: str = "",
                 fitness: float | None = None, threshold: float | None = None, polarity: int = 1,
                 n_nodes: int = 0, depth: int = 0, features: tuple[str, ...] = (),
                 infix: str = "", saved_path: str = "") -> None:
        self.rank = int(rank)
        self.name = str(name or "")
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
    def label(self) -> str:
        """What to call this bot on screen. ``#<rank>`` when nothing named it — which is a
        position in a listing rather than a name, and reads as one."""
        return self.name or f"#{self.rank}"

    @property
    def calibrated(self) -> bool:
        """Whether this program carries a threshold, and so could be run at all."""
        return self.threshold is not None


class Model:
    """One thing this machine can run, whichever way it got here.

    Attributes:
        source: :data:`DOWNLOADED`, :data:`BUILT`, :data:`IMPORTED`, :data:`SOCIETY` or
            :data:`PUBLISHED`.
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
    def imported(self) -> bool:
        """Whether this model was carried here from another machine."""
        return self.source == IMPORTED

    @property
    def published(self) -> bool:
        """Whether this model was published to the installation by another account.

        The question the panel asks before any other, because it changes who the model belongs
        to: everything else in the library is this machine's, and this is somebody else's, held
        here for as long as the grant behind it stands.
        """
        return self.source == PUBLISHED

    @property
    def portable(self) -> bool:
        """Whether this model may be written to a file and carried to another machine.

        Not the same question as :attr:`local`, and the difference is not about the files. A
        downloaded bundle is licensed per machine; a published model was handed to this account
        by a grant, and exporting one would be this machine deciding who else may run it. Both
        are refused by :func:`siarapp.transfer.export_model`, which says which of the two it is.
        """
        return self.source in PORTABLE_SOURCES

    @property
    def society(self) -> bool:
        """Whether this is a society of bots voting rather than one program.

        Worth asking separately from :attr:`built`, because it changes what the slug *means*: a
        built model is one search's answer and will say the same thing next month, and a society
        republishes as its leaderboard moves. Two scans of one folder a week apart under one
        society name are not necessarily two runs of the same detector.

        Asked of what the model *is* rather than of how it got here, so a society fetched from an
        installation answers the same as one bred on this machine: it is the same package, with
        the same members and the same moving leaderboard behind it.
        """
        return self.source == SOCIETY or (
            self.source == PUBLISHED
            and str(self.detail.get("family") or "") == SOCIETY_FAMILY
        )

    @property
    def local(self) -> bool:
        """Whether this model is a plain package on this disk, whoever produced it.

        The question the loader asks and the one export asks: both are answered by where the
        model *is*, not by which of the two ways it got here.
        """
        return self.source in LOCAL_SOURCES

    @property
    def shapes(self) -> tuple[str, ...]:
        """The structures this model emits — what its boxes are labelled, target first.

        Not the same question as what the model is *called*, and since a model can be named
        anything the two have to be asked separately. A downloaded bundle declares its shapes in
        its manifest; a built or imported one has the target it was evolved for, and the other
        tags that counted as that target.
        """
        declared = self.detail.get("shapes")
        if isinstance(declared, (list, tuple)) and declared:
            return tuple(str(s) for s in declared)
        target = str(self.detail.get("target") or "")
        tags = self.detail.get("positive_tags")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags or "[]")
            except ValueError:
                tags = []
        others = [str(t) for t in tags if str(t) != target] if isinstance(tags, list) else []
        if not target:
            return tuple(others)
        return (target, *others)

    @property
    def target(self) -> str:
        """The one thing this model detects — the first of its :attr:`shapes`.

        What the targets panel groups on, and what a survey is planned around. ``""`` when nothing
        here knows, which puts the model under "unknown" rather than under a guess.
        """
        shapes = self.shapes
        return shapes[0] if shapes else ""

    @property
    def run_name(self) -> str:
        """The search this model came out of, for a model built or imported here.

        Empty for a downloaded bundle, which is the honest answer: the run that produced it
        happened on somebody else's machine and its name is not in the manifest.

        A model renamed after the fact is still traceable through this, which is the whole reason
        siar-build gives a run a name of its own rather than reusing the model's.
        """
        return str(self.detail.get("run_name") or "")

    @property
    def looks_for(self) -> str:
        """What this model detects, as one cell: the first shape, and how many others there are.

        ``""`` when nothing on this machine knows — a bundle whose manifest declares no shapes,
        or a build row from before the column existed. A blank is the honest answer there; a
        guess would be a label on somebody's boxes that nothing put there.
        """
        shapes = self.shapes
        if not shapes:
            return ""
        return f"{shapes[0]} +{len(shapes) - 1}" if len(shapes) > 1 else shapes[0]

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
            detail={"family": str(row.get("family") or ""),
                    "shapes": list(row.get("shapes") or ())},
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


def _get(row, key: str, default=None):
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
            name=_get(row, "name", ""),
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
    "id", "target", "run_name", "positive_tags", "created_at", "input_dir", "output_dir",
    "package_dir", "objective",
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


#: Columns copied from a society row into :attr:`Model.detail`.
_SOCIETY_DETAIL = (
    "id", "name", "target", "positive_tags", "created_at", "input_dir", "output_dir",
    "package_dir", "sample_rate", "n_fft", "hop", "n_bins", "fmin_hz", "fmax_hz",
    "arena_recordings", "unseen_recordings", "top_n", "k", "n_members", "n_bots", "rounds",
    "evaluations", "best_arena_auc", "member_mean_auc", "unseen_auc", "stability", "parity_ok",
    "state", "seed", "last_round_at", "stopped_at", "notes",
)


def _members(conn: sqlite3.Connection, soc_id: int, limit: int = 50) -> list[Program]:
    """A society's leaderboard, best first, as programs.

    The same shape a build's bots are given, because the screen asks the same question of both —
    *what is inside this* — and the answer should not need two panels. What differs is the
    ordering: a build's programs are ranked by the search that found them, and a society's by how
    they scored on audio no search ever trained on.
    """
    # Two ``SELECT *`` queries merged here rather than one join naming columns. A join has to
    # name what it selects, and every name is a bet that this schema has that column — the exact
    # bet the rest of this module refuses to make, since the file belongs to a program that
    # versions separately. ``_get`` covers a column that is not there; a join cannot.
    try:
        ranked = conn.execute(
            "SELECT * FROM soc_ranks WHERE soc_id = ? ORDER BY rank LIMIT ?",
            (int(soc_id), int(limit)),
        ).fetchall()
    except sqlite3.Error:
        # A table this schema has not got is "no members", exactly as a missing column is a blank
        # field: a listing, never an error.
        return []
    if not ranked:
        return []

    wanted = [int(_get(row, "program_id", 0) or 0) for row in ranked]
    try:
        placeholders = ",".join("?" * len(wanted))
        found = {
            int(_get(row, "id", 0) or 0): row
            for row in conn.execute(
                f"SELECT * FROM programs WHERE id IN ({placeholders})", wanted)
        }
    except sqlite3.Error:
        found = {}

    out = []
    for row in ranked:
        program = found.get(int(_get(row, "program_id", 0) or 0))
        out.append(Program(
            # The society's rank, not the search's: these are ordered by how they scored on audio
            # no search of this society ever trained on.
            rank=_get(row, "rank", 0),
            name=_get(program, "name", "") if program is not None else "",
            kind=_get(program, "kind", "") if program is not None else "",
            # Fitness on this row means the arena score, which is the number the society ranked
            # by. The search's own fitness is about a different split and is not comparable.
            fitness=_get(row, "arena_auc"),
            threshold=_get(row, "threshold"),
            polarity=_get(program, "polarity", 1) if program is not None else 1,
            n_nodes=_get(program, "n_nodes", 0) if program is not None else 0,
            depth=_get(program, "depth", 0) if program is not None else 0,
            features=_features(_get(program, "features_used", "[]") if program is not None
                               else "[]"),
            infix=_get(program, "infix", "") if program is not None else "",
            saved_path=_get(program, "saved_path", "") if program is not None else "",
        ))
    return out


def society_models(db_path: str | None = None, *, limit: int = BUILD_LIMIT) -> list[Model]:
    """The societies ``siar-build`` is breeding on this machine, newest round first.

    A society publishes one runnable package like any other model, so from here it is a folder to
    hand :func:`siarapp.loader.load_local` and nothing more. What the listing has to carry, and a
    build's row does not, is that it **moves**: the members are whichever bots were best when it
    last published, and a society still running will publish different ones.

    A society that has not published yet, or whose package has been deleted, is listed and says so
    — the same discipline :func:`built_models` follows, and for the same reason: a row somebody
    will come looking for is worth more than a listing of successes.

    Args:
        db_path: The index to read. Defaults to :func:`build_db_path`.
        limit: Most recent societies to read.

    Returns:
        One :class:`Model` per society, each carrying the top of its leaderboard as programs.
        Empty when siar-build has never run here, when its index cannot be read, or when it is a
        schema without societies in it.
    """
    conn = _connect(db_path or build_db_path())
    if conn is None:
        return []
    try:
        try:
            rows = conn.execute(
                "SELECT * FROM socs ORDER BY COALESCE(last_round_at, created_at) DESC, id DESC"
                " LIMIT ?", (int(limit),)
            ).fetchall()
        except sqlite3.Error:
            return []

        models = []
        for row in rows:
            package = str(_get(row, "package_dir", "") or "")
            runnable = bool(package) and os.path.isdir(package)
            parity = _get(row, "parity_ok")
            members = int(_get(row, "n_members", 0) or 0)
            k = int(_get(row, "k", 0) or 0)
            if not package:
                note = "no model published yet — it has not finished a round"
            elif not runnable:
                note = f"published at {package}, which is no longer there"
            elif parity is not None and not parity:
                # The same refusal siar-build makes. Boxes from a model that does not compute what
                # was measured look exactly like boxes from one that does.
                note = "last parity gate FAILED — this model must not be scanned with"
                runnable = False
            else:
                note = ""
            models.append(Model(
                source=SOCIETY,
                slug=str(_get(row, "name", "") or _get(row, "target", "") or "society"),
                title=str(_get(row, "target", "") or ""),
                version=str(_get(row, "siarbuild_version", "") or ""),
                platform="source",
                path=package,
                size_bytes=_dir_bytes(package) if runnable else 0,
                stamped_at=_epoch(_get(row, "last_round_at", "")
                                  or _get(row, "created_at", "")),
                runnable=runnable,
                note=note,
                detail={name: _get(row, name) for name in _SOCIETY_DETAIL} | {
                    "votes": f"{k} of {members}" if members else "",
                },
                programs=_members(conn, int(_get(row, "id", 0) or 0)),
            ))
        return models
    finally:
        conn.close()


def imported_models() -> list[Model]:
    """The models carried onto this machine from another one, newest import first.

    They are read from this package's own workspace rather than from siar-build's index, because
    the machine they were built on is not this one and its database is not here. What arrived
    with them — the build row, the programs, the held-out figures — is in the bundle, so an
    imported model is as knowable as a local build and says on the row where it came from.

    Returns:
        One :class:`Model` per imported bundle.
    """
    models = []
    for row in installed_bundles():
        build = dict(row.get("build") or {})
        origin = row.get("exported_from") or "another machine"
        build["imported_from"] = origin
        build["imported_at"] = row.get("imported_at") or 0
        build["exported_at"] = row.get("exported_at") or ""
        build["uid"] = row.get("uid") or ""
        runnable = bool(row.get("runnable"))
        models.append(Model(
            source=IMPORTED,
            slug=str(row.get("slug") or "model"),
            title=str(row.get("title") or ""),
            version=str(row.get("version") or ""),
            platform="source",
            path=str(row.get("path") or ""),
            size_bytes=int(row.get("bytes") or 0),
            stamped_at=int(row.get("imported_at") or 0),
            runnable=runnable,
            note="" if runnable else "the package is missing from this import — import it again",
            detail=build,
            programs=[_program_from(p) for p in row.get("programs") or []],
        ))
    return models


def published_models() -> list[Model]:
    """The models published to the installation that this machine has fetched, newest first.

    Read from this package's own cache rather than from the installation, so the listing is the
    same on a vessel with no link as it is in the office: what is on this disk is what can be run,
    and whether the account is *still* granted a model is a question only a fetch can answer.

    The row says whose model it is and whether it has been released, because both are facts about
    somebody else's decision that this machine cannot change. An unreleased model in this list is
    not an error — it is the publisher's own upload, or a super's, and being able to run one
    before it goes out is the whole reason publishing from the build box is worth doing.

    Returns:
        One :class:`Model` per cached model.
    """
    models = []
    for row in installed_published():
        catalogue = dict(row.get("catalogue") or {})
        manifest = catalogue.pop("manifest", None)
        shapes = manifest.get("shapes") if isinstance(manifest, dict) else None
        vote = catalogue.get("vote") if isinstance(catalogue.get("vote"), dict) else {}
        detail = dict(catalogue)
        detail["fetched_at"] = row.get("fetched_at") or 0
        detail["install"] = row.get("base_url") or ""
        if isinstance(shapes, list) and shapes:
            detail["shapes"] = [str(shape) for shape in shapes]
        if vote.get("n"):
            detail["votes"] = f"{int(vote.get('k') or 0)} of {int(vote['n'])}"
        runnable = bool(row.get("runnable"))
        models.append(Model(
            source=PUBLISHED,
            slug=str(row.get("slug") or "model"),
            title=str(row.get("title") or ""),
            version=str(row.get("version") or ""),
            platform="source",
            path=str(row.get("path") or ""),
            size_bytes=int(row.get("bytes") or 0),
            stamped_at=int(row.get("fetched_at") or 0),
            runnable=runnable,
            note="" if runnable else
                 "the package is missing from this download — fetch it again with "
                 f"`siar-app published --get {row.get('slug') or ''}`",
            detail=detail,
            programs=[_program_from(p) for p in row.get("programs") or []],
        ))
    return models


def _program_from(row: dict) -> Program:
    """One program out of a bundle's manifest, ignoring fields this version does not know."""
    if not isinstance(row, dict):
        return Program()
    known = {name: row[name] for name in Program.__slots__ if name in row}
    features = known.get("features")
    if features is not None:
        known["features"] = tuple(str(name) for name in features)
    return Program(**known)


def local_model(slug: str, *, db_path: str | None = None) -> Model | None:
    """The runnable model on this disk with this name, newest first, or ``None``.

    What ``siar-app run -a <slug>`` consults before it asks the server: a model that was built
    here or carried here has no other way of being named, and having to give
    ``--algorithm-path`` for a model the library is already listing is the CLI failing to know
    what it knows.

    Args:
        slug: The name to match, case-insensitively.
        db_path: siar-build's index, for a test.

    Returns:
        The most recent runnable match. Imported models come first — an import is the most
        deliberate act of the four — then societies, then single builds, and a model published by
        somebody else last. That order is what stops a cached copy of another account's model
        shadowing one of this machine's own: two of them answering to one name is unlikely, and
        the one to run is the one this bench made. The order is fixed rather than incidental so
        that a name which somehow resolved two ways would resolve the same way every time.
    """
    wanted = str(slug or "").strip().lower()
    if not wanted:
        return None
    candidates = [m for m in (imported_models() + society_models(db_path)
                              + built_models(db_path) + published_models())
                  if m.runnable and m.slug.lower() == wanted]
    return candidates[0] if candidates else None


def library(*, db_path: str | None = None) -> list[Model]:
    """Everything runnable on this machine, grouped by how it got here.

    Grouped rather than interleaved by date. They are different kinds of thing — one can be
    fetched again from the server, one belongs to another account and is here on a grant, one is a
    population still being bred, one is a single search's answer, and one is here because somebody
    carried it — and a list that sorted them together would make that difference invisible at
    exactly the moment somebody is choosing between them.

    The two that came off the network lead, because "what did I fetch" is one question and asked
    together. Societies come before builds because a society *is* the best of many builds, so a
    machine running one has its answer at the top rather than under forty rounds of its own
    workings.

    Args:
        db_path: siar-build's index, for a test or a non-standard workspace.

    Returns:
        The models, downloaded first, each group newest first.
    """
    return (downloaded_models() + published_models() + society_models(db_path)
            + built_models(db_path) + imported_models())


#: What a model whose target nothing on this machine knows is filed under. A word rather than a
#: blank row, so the group is selectable and says why it exists.
UNKNOWN_TARGET = "unknown"


class Target:
    """One thing this machine can detect, and everything that can detect it.

    The first question about a library is not "which of these forty models" but "what can this
    machine find" — a target is what a survey is planned around, and the models under it are the
    attempts at it. Grouping is by :attr:`Model.target` rather than by name, because two models
    called ``wideband`` and ``strict`` looking for the same call are one choice to make, and forty
    rows sorted by date hide that they are.

    Attributes:
        name: The target, or :data:`UNKNOWN_TARGET`.
        models: Every model that emits it, in the order :func:`library` returned them.
    """

    __slots__ = ("name", "models")

    def __init__(self, name: str, models: list[Model]) -> None:
        self.name = str(name)
        self.models = list(models)

    @property
    def runnable(self) -> int:
        """How many of these can actually be run as they stand."""
        return sum(1 for m in self.models if m.runnable)

    @property
    def bots(self) -> int:
        """Every bot across every model here, champions and runners-up alike."""
        return sum(len(m.programs) for m in self.models)

    @property
    def newest(self) -> int:
        """Epoch seconds of the most recent of them, for ordering."""
        return max((m.stamped_at for m in self.models), default=0)


def targets(models: list[Model]) -> list[Target]:
    """Group models by what they detect, most recently added first.

    Args:
        models: What :func:`library` returned, or any subset.

    Returns:
        One :class:`Target` per distinct target. Ordering puts the most recently added first and
        breaks ties by name, so two calls a second apart agree; :data:`UNKNOWN_TARGET` sorts with
        the rest rather than being pinned anywhere, since it is a real group with real models in
        it and pinning it would say otherwise.
    """
    grouped: dict[str, list[Model]] = {}
    for model in models:
        grouped.setdefault(model.target or UNKNOWN_TARGET, []).append(model)
    out = [Target(name, rows) for name, rows in grouped.items()]
    out.sort(key=lambda t: (-t.newest, t.name))
    return out


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
