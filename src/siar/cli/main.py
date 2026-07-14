# Vixen Intelligence c.2026
"""The ``siar`` command line.

This module builds the argparse tree and dispatches; the work is in
:mod:`siar.cli.commands`. The console script ``siar`` (see ``pyproject.toml``) calls
:func:`main`.

Subcommands:

* ``siar version``   — print the package version.
* ``siar scan``      — walk a folder of audio, summarise it, register it as a dataset.
* ``siar detectors`` — list the registered anomaly solutions.
* ``siar train``     — learn "normal" from a folder, and save a detector.
* ``siar run``       — apply a saved detector to a folder, and record what it finds.
* ``siar models``    — list trained models.
* ``siar runs``      — list inference runs.
* ``siar export``    — write a run's results to JSON.
* ``siar dash``      — serve the local results dashboard.
* ``siar db``        — create or inspect the results database.

Still to come: ``siar optimise`` (the Optuna hyperparameter sweep).
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from siar import __version__
from siar.cli import commands

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="siar",
        description="Signal Intelligence and Reconnaissance — "
        "unsupervised acoustic anomaly detection.",
    )
    parser.add_argument("--version", action="version", version=f"siar {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("version", help="print the package version")

    p_scan = sub.add_parser("scan", help="summarise and register a folder of audio")
    p_scan.add_argument("data", metavar="FOLDER", help="folder of audio to scan")
    p_scan.add_argument("--name", help="dataset name (default: the folder's name)")
    p_scan.add_argument("--json", metavar="PATH", help="also write the full scan to a JSON file")
    p_scan.add_argument(
        "--no-register",
        action="store_true",
        help="print the summary without recording the dataset in the database",
    )

    sub.add_parser("detectors", help="list the registered anomaly solutions")

    p_train = sub.add_parser(
        "train",
        help="train a detector on a folder of normal audio",
        description="Learn what 'normal' sounds like. The folder should be predominantly free of "
        "the events you want to find — anything common in it will NOT be flagged.",
    )
    p_train.add_argument("data", metavar="FOLDER", help="folder of (assumed normal) audio")
    p_train.add_argument("--detector", default="conv_ae", help="which solution (default: conv_ae)")
    p_train.add_argument("--name", help="model name (default: the folder's name)")
    p_train.add_argument(
        "--mode",
        default="pooled_linear",
        choices=("pooled_linear", "log_mel"),
        help="frequency axis (default: pooled_linear)",
    )
    p_train.add_argument("--sample-rate", type=int, help="resample everything to this rate")
    p_train.add_argument("--fft", type=int, default=1024, help="FFT size (default: 1024)")
    p_train.add_argument("--bins", type=int, default=64, help="grid height (default: 64)")
    p_train.add_argument("--patch", type=int, default=16, help="patch width in frames (default: 16)")
    p_train.add_argument("--fmin", type=float, default=0.0, help="low edge of the band, Hz")
    p_train.add_argument("--fmax", type=float, help="high edge of the band, Hz (default: Nyquist)")
    p_train.add_argument("--epochs", type=int, help="override the training epoch count")
    p_train.add_argument(
        "--contamination",
        type=float,
        default=1e-3,
        help="fraction of pixels you accept being flagged on normal audio (default: 0.001). "
        "This IS the threshold — lower means fewer, higher-confidence detections.",
    )
    p_train.add_argument(
        "--threshold-method",
        default="evt",
        choices=("evt", "quantile", "robust_z"),
        help="how to turn contamination into a threshold (default: evt)",
    )
    p_train.add_argument("--seed", type=int, default=0, help="seed for the split and weight init")
    p_train.add_argument("--out", metavar="PATH", help="also write the model JSON here")

    p_run = sub.add_parser("run", help="apply a trained model to a folder of audio")
    p_run.add_argument("model", metavar="MODEL", help="model uid (or a unique prefix)")
    p_run.add_argument("data", metavar="FOLDER", help="folder to score")
    p_run.add_argument("--name", help="run name (default: the folder's name)")
    p_run.add_argument(
        "--threshold", type=float, help="override the model's z threshold for this run"
    )
    p_run.add_argument(
        "--no-render", action="store_true", help="skip the spectrogram PNGs (faster, headless)"
    )
    p_run.add_argument("--out", metavar="PATH", help="also write the results JSON here")

    sub.add_parser("models", help="list trained models")
    sub.add_parser("runs", help="list inference runs")

    p_export = sub.add_parser("export", help="write a run's results to JSON")
    p_export.add_argument("run", metavar="RUN", help="run uid (or a unique prefix)")
    p_export.add_argument("--out", metavar="PATH", help="output file (default: stdout)")

    p_dash = sub.add_parser("dash", help="serve the local results dashboard")
    p_dash.add_argument("--port", type=int, default=8420, help="port (default: 8420)")
    p_dash.add_argument("--open", action="store_true", help="open a browser")

    p_db = sub.add_parser("db", help="create or inspect the results database")
    p_db.add_argument(
        "db_command",
        choices=("migrate", "check"),
        metavar="{migrate,check}",
        help="migrate: create missing tables; check: report the database's state",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    handlers = {
        "version": commands.cmd_version,
        "scan": commands.cmd_scan,
        "detectors": commands.cmd_detectors,
        "train": commands.cmd_train,
        "run": commands.cmd_run,
        "models": commands.cmd_models,
        "runs": commands.cmd_runs,
        "export": commands.cmd_export,
        "dash": commands.cmd_dash,
        "db": commands.cmd_db,
    }
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"siar: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
