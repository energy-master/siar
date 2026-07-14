# Vixen Intelligence c.2026
"""The results database — a thin, explicit wrapper over ``sqlite3``.

No ORM. The schema is small and the queries are few, so hand-written SQL is clearer than a layer
that hides it.

**Concurrency.** WAL mode is on, so the dashboard can read while a run is writing. But SIAR
never lets the HPO worker *processes* write here: Optuna keeps its own journal storage
(:func:`siar.paths.optuna_dir`) and the parent process mirrors finished trials into
``siar_trials``. One writer, no lock contention, no ``database is locked`` at trial 40 of 100.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from siar import paths
from siar.store.schema import SCHEMA_VERSION, TABLES

__all__ = ["Store", "new_uid", "utcnow"]


def utcnow() -> str:
    """Current UTC time as an ISO-8601 string, to the second."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_uid(prefix: str) -> str:
    """Generate a stable, human-typable handle for a study, model or run.

    Args:
        prefix: A short tag, e.g. ``"run"`` or ``"study"``.

    Returns:
        Something like ``run-20260714T113900Z-3f9a``. Sortable by time, unique enough to paste.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{secrets.token_hex(2)}"


class Store:
    """A connection to the SIAR results database.

    Usable as a context manager::

        with Store() as store:
            store.migrate()
            dataset_id = store.upsert_dataset(scan)

    Attributes:
        path: The database file this Store is connected to.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        """Open (and create, if needed) the results database.

        Args:
            path: Database file. Defaults to :func:`siar.paths.db_path`.
        """
        if path is None:
            paths.ensure_workspace()
            path = paths.db_path()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 10000")

    def __enter__(self) -> "Store":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()

    @property
    def conn(self) -> sqlite3.Connection:
        """The underlying connection, for queries this class does not wrap."""
        return self._conn

    # --- schema -------------------------------------------------------------

    def migrate(self) -> None:
        """Create any missing tables and record the schema version.

        Idempotent — every statement is ``IF NOT EXISTS``, so this is safe to call on every
        startup and is how a database gets created in the first place.
        """
        with self._conn:
            for stmt in TABLES:
                self._conn.execute(stmt)
            self._conn.execute(
                "INSERT INTO siar_meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(SCHEMA_VERSION),),
            )

    def schema_version(self) -> int:
        """The schema version recorded in the database, or 0 if never migrated."""
        try:
            row = self._conn.execute(
                "SELECT value FROM siar_meta WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row["value"]) if row else 0

    # --- datasets -----------------------------------------------------------

    def upsert_dataset(self, name: str, path: str, n_files: int, total_seconds: float,
                       sample_rates: dict[int, int]) -> int:
        """Record (or refresh) a scanned corpus.

        Keyed on ``path``, so re-scanning the same folder updates the row rather than making a
        second one.

        Args:
            name: Display name.
            path: Absolute folder path.
            n_files: Number of readable recordings.
            total_seconds: Total audio duration.
            sample_rates: Map of sample rate -> file count.

        Returns:
            The dataset's row id.
        """
        rates = json.dumps({str(k): v for k, v in sorted(sample_rates.items())})
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO siar_datasets (name, path, n_files, total_seconds, sample_rates)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    name = excluded.name,
                    n_files = excluded.n_files,
                    total_seconds = excluded.total_seconds,
                    sample_rates = excluded.sample_rates
                """,
                (name, path, n_files, total_seconds, rates),
            )
        row = self._conn.execute(
            "SELECT id FROM siar_datasets WHERE path = ?", (path,)
        ).fetchone()
        return int(row["id"])

    def datasets(self) -> list[sqlite3.Row]:
        """Every registered dataset, newest first."""
        return self._conn.execute(
            "SELECT * FROM siar_datasets ORDER BY created_at DESC"
        ).fetchall()

    # --- models -------------------------------------------------------------

    def insert_model(self, model, *, study_id: int | None = None,
                     trial_id: int | None = None) -> tuple[int, str]:
        """Persist a trained model.

        Args:
            model: A :class:`siar.train.model.TrainedModel`.
            study_id: The optimisation study that produced it, if any.
            trial_id: The winning trial, if any.

        Returns:
            ``(row_id, model_uid)``.
        """
        uid = new_uid("model")
        doc = model.to_json()
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO siar_models
                    (model_uid, name, detector, format, study_id, trial_id,
                     spec_json, model_json, threshold, objective, n_params)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    uid,
                    model.name,
                    doc["detector"]["detector"],
                    doc["format"],
                    study_id,
                    trial_id,
                    json.dumps(model.spec.to_dict()),
                    json.dumps(doc),
                    float(model.threshold),
                    model.provenance.get("objective"),
                    int(doc["detector"].get("n_params", 0)),
                ),
            )
        return int(cur.lastrowid), uid

    def models(self) -> list[sqlite3.Row]:
        """Every trained model, newest first (without the weights)."""
        return self._conn.execute(
            "SELECT id, model_uid, name, detector, format, threshold, objective, n_params, "
            "spec_json, created_at FROM siar_models ORDER BY created_at DESC"
        ).fetchall()

    def model_by_uid(self, uid: str) -> sqlite3.Row | None:
        """Fetch one model row, including its full JSON document.

        Args:
            uid: The model's uid. A unique prefix is also accepted, so a user can type the first
                few characters instead of the whole handle.

        Returns:
            The row, or ``None`` if no model matches.
        """
        row = self._conn.execute(
            "SELECT * FROM siar_models WHERE model_uid = ?", (uid,)
        ).fetchone()
        if row:
            return row
        rows = self._conn.execute(
            "SELECT * FROM siar_models WHERE model_uid LIKE ?", (uid + "%",)
        ).fetchall()
        return rows[0] if len(rows) == 1 else None

    # --- runs ---------------------------------------------------------------

    def create_run(self, *, name: str, model_id: int, input_path: str,
                   threshold: float) -> tuple[int, str]:
        """Open a new inference run.

        Args:
            name: Display name.
            model_id: The model being applied.
            input_path: The folder being scored.
            threshold: The z threshold in force (may override the model's own).

        Returns:
            ``(row_id, run_uid)``.
        """
        uid = new_uid("run")
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO siar_runs (run_uid, name, model_id, input_path, threshold, started_at)
                VALUES (?,?,?,?,?,?)
                """,
                (uid, name, model_id, input_path, float(threshold), utcnow()),
            )
        return int(cur.lastrowid), uid

    def finish_run(self, run_id: int, *, status: str = "done",
                   error: str | None = None) -> None:
        """Close a run, recomputing its totals from the rows actually written.

        Counting here rather than trusting a caller-supplied total means the summary can never
        disagree with the detections in the table.

        Args:
            run_id: The run.
            status: ``done``, ``failed`` or ``cancelled``.
            error: Failure message, if any.
        """
        with self._conn:
            self._conn.execute(
                """
                UPDATE siar_runs SET
                    status = ?,
                    error_message = ?,
                    finished_at = ?,
                    n_files = (SELECT COUNT(*) FROM siar_files WHERE run_id = ?),
                    n_detections = (SELECT COUNT(*) FROM siar_detections WHERE run_id = ?)
                WHERE id = ?
                """,
                (status, error, utcnow(), run_id, run_id, run_id),
            )

    def runs(self) -> list[sqlite3.Row]:
        """Every run, newest first, with its model's name."""
        return self._conn.execute(
            """
            SELECT r.*, m.model_uid, m.name AS model_name, m.detector
            FROM siar_runs r JOIN siar_models m ON m.id = r.model_id
            ORDER BY r.created_at DESC
            """
        ).fetchall()

    def run_by_uid(self, uid: str) -> sqlite3.Row | None:
        """Fetch one run by uid (or unique prefix), with its model's details."""
        sql = """
            SELECT r.*, m.model_uid, m.name AS model_name, m.detector, m.spec_json
            FROM siar_runs r JOIN siar_models m ON m.id = r.model_id
            WHERE r.run_uid {op} ?
        """
        row = self._conn.execute(sql.format(op="="), (uid,)).fetchone()
        if row:
            return row
        rows = self._conn.execute(sql.format(op="LIKE"), (uid + "%",)).fetchall()
        return rows[0] if len(rows) == 1 else None

    # --- files and detections -----------------------------------------------

    def insert_file(self, run_id: int, *, path: str, name: str, duration_s: float,
                    sample_rate: int, frames: int, n_bins: int, t_min: float, t_max: float,
                    f_min: float, f_max: float, png_path: str | None) -> int:
        """Record one scored recording within a run.

        The ``t_*`` / ``f_*`` extents and the grid dimensions are stored so the browser can place
        a box over the PNG without re-deriving any feature geometry in JavaScript.

        Returns:
            The file's row id.
        """
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO siar_files
                    (run_id, path, name, duration_s, sample_rate, png_path,
                     frames, n_bins, t_min, t_max, f_min, f_max)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (run_id, path, name, duration_s, sample_rate, png_path,
                 frames, n_bins, t_min, t_max, f_min, f_max),
            )
        return int(cur.lastrowid)

    def insert_detections(self, run_id: int, file_id: int, detections) -> None:
        """Write a file's detections, and update the file's summary columns.

        Args:
            run_id: The run.
            file_id: The file.
            detections: An iterable of :class:`siar.detect.boxes.Detection`, strongest first.
        """
        rows = [
            (run_id, file_id, i, d.t_start, d.t_end, d.f_low, d.f_high, d.score,
             d.peak_score, d.area, d.fill, d.frame_lo, d.frame_hi, d.bin_lo, d.bin_hi)
            for i, d in enumerate(detections)
        ]
        with self._conn:
            if rows:
                self._conn.executemany(
                    """
                    INSERT INTO siar_detections
                        (run_id, file_id, box_index, t_start, t_end, f_low, f_high, score,
                         peak_score, area, fill, frame_lo, frame_hi, bin_lo, bin_hi)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    rows,
                )
            self._conn.execute(
                "UPDATE siar_files SET n_detections = ?, max_score = ? WHERE id = ?",
                (len(rows), max((d.score for d in detections), default=None), file_id),
            )

    def files_for_run(self, run_id: int) -> list[sqlite3.Row]:
        """A run's files, most anomalous first — which is the order a user wants to triage in."""
        return self._conn.execute(
            "SELECT * FROM siar_files WHERE run_id = ? "
            "ORDER BY (max_score IS NULL), max_score DESC, name",
            (run_id,),
        ).fetchall()

    def file_by_id(self, file_id: int) -> sqlite3.Row | None:
        """Fetch one scored file."""
        return self._conn.execute(
            "SELECT * FROM siar_files WHERE id = ?", (file_id,)
        ).fetchone()

    def detections_for_file(self, file_id: int) -> list[sqlite3.Row]:
        """A file's detections, strongest first."""
        return self._conn.execute(
            "SELECT * FROM siar_detections WHERE file_id = ? ORDER BY score DESC",
            (file_id,),
        ).fetchall()

    # --- logs ---------------------------------------------------------------

    def log(self, line: str, *, study_id: int | None = None, run_id: int | None = None,
            level: str = "info") -> None:
        """Append a line to a study's or run's log.

        Args:
            line: The message.
            study_id: The study it belongs to, if any.
            run_id: The run it belongs to, if any.
            level: ``"info"``, ``"warn"`` or ``"error"``.
        """
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS seq FROM siar_logs "
            "WHERE study_id IS ? AND run_id IS ?",
            (study_id, run_id),
        ).fetchone()
        with self._conn:
            self._conn.execute(
                "INSERT INTO siar_logs (study_id, run_id, seq, level, line) VALUES (?,?,?,?,?)",
                (study_id, run_id, int(row["seq"]), level, line),
            )
