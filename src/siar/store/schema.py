# Vixen Intelligence c.2026
"""The SQLite schema.

Shape of the data:

    dataset  -- a folder of audio the user pointed at
      study    -- one optimisation (HPO sweep) over that dataset
        trial    -- one Optuna trial within it
        model    -- the trained detector the winning trial produced
          run      -- applying a model to a folder (the "inference" step)
            file     -- one recording within that run
              detection -- one 2-D box: (t_start, t_end, f_low, f_high, score)

A note on ``run``: training also produces one. When ``siar optimise`` finishes it applies the
best model back over the input corpus and records that as a normal run. The dashboard therefore
has exactly *one* code path for "show me the detections", whether they came from the training
corpus or from a later ``siar run`` on new audio. Building two would have been the obvious
mistake.

PNGs are written to ``runs/<uid>/`` on disk and referenced by path, not stored as BLOBs. This
is a local single-user tool: keeping the images on the filesystem means they can be inspected,
copied and served without going through SQLite, and the database stays small enough to be
copied around.
"""
from __future__ import annotations

__all__ = ["SCHEMA_VERSION", "TABLES"]

#: Bumped whenever TABLES changes in a way that needs a migration.
SCHEMA_VERSION = 1

#: Every table, in dependency order. All statements are ``IF NOT EXISTS``, so applying them to
#: an existing database is a no-op and ``siar db migrate`` is safe to re-run.
TABLES: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS siar_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS siar_datasets (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        name           TEXT    NOT NULL,
        path           TEXT    NOT NULL,
        n_files        INTEGER NOT NULL,
        total_seconds  REAL    NOT NULL,
        sample_rates   TEXT    NOT NULL,          -- JSON: {"48000": 120, "44100": 3}
        created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
        UNIQUE (path)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS siar_studies (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        study_uid         TEXT    NOT NULL UNIQUE,
        name              TEXT    NOT NULL,
        dataset_id        INTEGER NOT NULL,
        detector          TEXT    NOT NULL,       -- key into siar.models.registry.DETECTORS
        objective         TEXT    NOT NULL,
        status            TEXT    NOT NULL DEFAULT 'running'
                          CHECK (status IN ('running','done','failed','cancelled')),
        n_trials          INTEGER NOT NULL,
        n_trials_done     INTEGER NOT NULL DEFAULT 0,
        n_trials_pruned   INTEGER NOT NULL DEFAULT 0,
        fmin_hz           REAL    NOT NULL,
        fmax_hz           REAL    NOT NULL,
        search_space_json TEXT,
        split_seed        INTEGER NOT NULL DEFAULT 0,
        best_trial_id     INTEGER,
        best_value        REAL,
        model_id          INTEGER,
        error_message     TEXT,
        started_at        TEXT,
        finished_at       TEXT,
        created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (dataset_id) REFERENCES siar_datasets(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS siar_trials (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        study_id      INTEGER NOT NULL,
        trial_number  INTEGER NOT NULL,
        state         TEXT    NOT NULL
                      CHECK (state IN ('running','complete','pruned','failed')),
        value         REAL,                       -- the objective; NULL if pruned or failed
        params_json   TEXT    NOT NULL,
        metrics_json  TEXT,                       -- secondary metrics, shown but not optimised
        n_params      INTEGER,
        epochs_run    INTEGER,
        duration_s    REAL,
        created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
        UNIQUE (study_id, trial_number),
        FOREIGN KEY (study_id) REFERENCES siar_studies(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS siar_models (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        model_uid      TEXT    NOT NULL UNIQUE,
        name           TEXT    NOT NULL,
        detector       TEXT    NOT NULL,
        format         TEXT    NOT NULL,
        study_id       INTEGER,
        trial_id       INTEGER,
        spec_json      TEXT    NOT NULL,          -- FeatureSpec
        model_json     TEXT    NOT NULL,          -- full descriptor INCLUDING weights_b64
        threshold      REAL    NOT NULL,
        objective      REAL,
        n_params       INTEGER,
        created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (study_id) REFERENCES siar_studies(id) ON DELETE SET NULL,
        FOREIGN KEY (trial_id) REFERENCES siar_trials(id)  ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS siar_runs (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        run_uid        TEXT    NOT NULL UNIQUE,
        name           TEXT    NOT NULL,
        model_id       INTEGER NOT NULL,
        input_path     TEXT    NOT NULL,
        status         TEXT    NOT NULL DEFAULT 'running'
                       CHECK (status IN ('running','done','failed','cancelled')),
        threshold      REAL    NOT NULL,          -- may override the model's own
        n_files        INTEGER NOT NULL DEFAULT 0,
        n_detections   INTEGER NOT NULL DEFAULT 0,
        error_message  TEXT,
        started_at     TEXT,
        finished_at    TEXT,
        created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (model_id) REFERENCES siar_models(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS siar_files (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        INTEGER NOT NULL,
        path          TEXT    NOT NULL,
        name          TEXT    NOT NULL,
        duration_s    REAL    NOT NULL,
        sample_rate   INTEGER NOT NULL,
        n_detections  INTEGER NOT NULL DEFAULT 0,
        max_score     REAL,
        png_path      TEXT,                       -- relative to runs/<run_uid>/
        -- The PNG's axis extents. The browser needs these to place a box over the image
        -- without re-deriving the feature geometry in JS.
        frames        INTEGER NOT NULL,
        n_bins        INTEGER NOT NULL,
        t_min         REAL    NOT NULL,
        t_max         REAL    NOT NULL,
        f_min         REAL    NOT NULL,
        f_max         REAL    NOT NULL,
        FOREIGN KEY (run_id) REFERENCES siar_runs(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS siar_detections (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id       INTEGER NOT NULL,
        file_id      INTEGER NOT NULL,
        box_index    INTEGER NOT NULL,            -- rank within the file, by score desc
        t_start      REAL    NOT NULL,            -- seconds
        t_end        REAL    NOT NULL,
        f_low        REAL    NOT NULL,            -- Hz
        f_high       REAL    NOT NULL,
        score        REAL    NOT NULL,
        peak_score   REAL    NOT NULL,
        area         INTEGER NOT NULL,            -- component pixels
        fill         REAL    NOT NULL,            -- component pixels / bbox pixels
        -- Grid coordinates too, so the dashboard can draw without floating-point drift.
        frame_lo     INTEGER NOT NULL,
        frame_hi     INTEGER NOT NULL,
        bin_lo       INTEGER NOT NULL,
        bin_hi       INTEGER NOT NULL,
        UNIQUE (run_id, file_id, box_index),
        FOREIGN KEY (run_id)  REFERENCES siar_runs(id)  ON DELETE CASCADE,
        FOREIGN KEY (file_id) REFERENCES siar_files(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS siar_logs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        study_id   INTEGER,
        run_id     INTEGER,
        seq        INTEGER NOT NULL,
        level      TEXT    NOT NULL DEFAULT 'info',
        line       TEXT    NOT NULL,
        created_at TEXT    NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (study_id) REFERENCES siar_studies(id) ON DELETE CASCADE,
        FOREIGN KEY (run_id)   REFERENCES siar_runs(id)    ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_trials_study ON siar_trials(study_id, trial_number)",
    "CREATE INDEX IF NOT EXISTS ix_trials_value ON siar_trials(study_id, value DESC)",
    "CREATE INDEX IF NOT EXISTS ix_files_run    ON siar_files(run_id)",
    "CREATE INDEX IF NOT EXISTS ix_det_run      ON siar_detections(run_id, score DESC)",
    "CREATE INDEX IF NOT EXISTS ix_det_file     ON siar_detections(file_id, t_start)",
    "CREATE INDEX IF NOT EXISTS ix_logs_study   ON siar_logs(study_id, seq)",
    "CREATE INDEX IF NOT EXISTS ix_logs_run     ON siar_logs(run_id, seq)",
)
