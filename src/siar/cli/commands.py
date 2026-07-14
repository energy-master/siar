# Vixen Intelligence c.2026
"""Implementations of the ``siar`` subcommands.

Each ``cmd_*`` takes the parsed argparse namespace and returns a process exit code. Argument
*parsing* lives in :mod:`siar.cli.main`; this module does the work.

Imports are kept lazy where they are heavy: ``siar scan`` must not pay for torch.
"""
from __future__ import annotations

import argparse
import json
import sys

from siar import __version__, paths
from siar.data.dataset import scan_folder
from siar.store.db import Store

__all__ = [
    "cmd_activate",
    "cmd_dash",
    "cmd_db",
    "cmd_detectors",
    "cmd_export",
    "cmd_export_ident",
    "cmd_models",
    "cmd_run",
    "cmd_runs",
    "cmd_scan",
    "cmd_train",
    "cmd_version",
]


def cmd_version(_args: argparse.Namespace) -> int:
    """Print the package version.

    Args:
        _args: Unused.

    Returns:
        Exit code 0.
    """
    print(f"siar {__version__}")
    return 0


def cmd_activate(args: argparse.Namespace) -> int:
    """Register an API key so the CLI is unlocked.

    Args:
        args: Needs ``key``.

    Returns:
        Exit code 0, or 1 on failure.
    """
    from siar.auth import activate

    path = activate(args.key)
    print(f"activated — key hash written to {path}")
    print("export SIAR_API_KEY=<your-key> in your shell to use SIAR")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Walk a folder of audio, summarise it, and register it as a dataset.

    Reads headers only — a multi-GB corpus is summarised in a second or two, not decoded.

    Args:
        args: Needs ``data`` (folder), ``json`` (optional output path), ``no_register``.

    Returns:
        Exit code 0, or 1 if the folder holds no readable audio.
    """
    scan = scan_folder(args.data)

    if scan.unreadable:
        print(f"warning: {len(scan.unreadable)} file(s) could not be read:", file=sys.stderr)
        for path, reason in scan.unreadable[:5]:
            print(f"  {path}: {reason}", file=sys.stderr)
        if len(scan.unreadable) > 5:
            print(f"  ... and {len(scan.unreadable) - 5} more", file=sys.stderr)

    if not scan.files:
        print(f"no readable audio found in {scan.root}", file=sys.stderr)
        return 1

    rates = scan.sample_rates
    hours = scan.total_duration / 3600.0
    print(f"{scan.root}")
    print(f"  files       {scan.n_files}")
    print(f"  duration    {scan.total_duration:.1f} s ({hours:.2f} h)")
    print(f"  sample rate {', '.join(f'{r} Hz x{n}' for r, n in sorted(rates.items()))}")

    # A corpus with several sample rates is nearly always an accident, and it is much cheaper to
    # say so now than after an hour of optimisation on a grid built from the wrong Nyquist.
    if len(rates) > 1:
        print(
            f"  note: mixed sample rates. Everything will be resampled to the spec's rate "
            f"(default {scan.dominant_sample_rate} Hz, the most common). Pass --sample-rate to "
            f"choose.",
            file=sys.stderr,
        )

    durations = sorted(f.duration for f in scan.files)
    print(f"  shortest    {durations[0]:.2f} s")
    print(f"  longest     {durations[-1]:.2f} s")
    print(f"  median      {durations[len(durations) // 2]:.2f} s")

    if not args.no_register:
        with Store() as store:
            store.migrate()
            dataset_id = store.upsert_dataset(
                name=args.name or scan.root.rstrip("/").rsplit("/", 1)[-1],
                path=scan.root,
                n_files=scan.n_files,
                total_seconds=scan.total_duration,
                sample_rates=rates,
            )
        print(f"  registered  dataset #{dataset_id} in {paths.db_path()}")

    if args.json:
        payload = {
            "root": scan.root,
            "n_files": scan.n_files,
            "total_seconds": scan.total_duration,
            "sample_rates": {str(k): v for k, v in sorted(rates.items())},
            "files": [
                {
                    "path": f.path,
                    "name": f.name,
                    "sample_rate": f.sample_rate,
                    "channels": f.channels,
                    "duration": f.duration,
                }
                for f in scan.files
            ],
            "unreadable": [{"path": p, "reason": r} for p, r in scan.unreadable],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  wrote       {args.json}")

    return 0


def cmd_detectors(_args: argparse.Namespace) -> int:
    """List the registered anomaly solutions.

    Args:
        _args: Unused.

    Returns:
        Exit code 0.
    """
    # Import the package, not the registry module — the package is what imports each detector
    # for its registration side effect.
    from siar.models import DETECTORS

    if not DETECTORS:
        print("no detectors registered")
        return 0
    width = max(len(k) for k in DETECTORS)
    for key, cls in sorted(DETECTORS.items()):
        summary = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
        print(f"{key:<{width}}  {summary}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train an anomaly detector on a folder of (assumed normal) audio.

    Args:
        args: Needs ``data``, ``detector``, ``name``, ``contamination``, ``threshold_method``,
            ``mode``, ``fft``, ``bins``, ``patch``, ``sample_rate``, ``fmin``, ``fmax``, ``seed``,
            ``out``, ``epochs``.

    Returns:
        Exit code 0, or 1 on failure.
    """
    import json as _json

    from siar.data.dataset import scan_folder
    from siar.features.spec import FeatureSpec
    from siar.models import get_detector
    from siar.train.fit import default_spec, train_from_folder

    scan = scan_folder(args.data)
    if not scan.files:
        print(f"no readable audio in {args.data}", file=sys.stderr)
        return 1

    sample_rate = args.sample_rate or scan.dominant_sample_rate
    spec = default_spec(
        sample_rate,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
        fft_size=args.fft,
        n_bins=args.bins,
        patch_frames=args.patch,
    )
    if args.mode != spec.mode:
        spec = FeatureSpec(**{**spec.to_dict(), "mode": args.mode})

    config = None
    if args.epochs:
        config = get_detector(args.detector).default_config(spec)
        config["epochs"] = args.epochs

    print(f"training on {scan.n_files} file(s), {scan.total_duration / 60:.1f} min of audio")
    print(f"  spec: {spec.mode} {spec.n_bins} bins, {spec.fft_size}-pt FFT @ {spec.sample_rate} Hz")

    result = train_from_folder(
        args.data,
        detector=args.detector,
        spec=spec,
        config=config,
        threshold_method=args.threshold_method,
        contamination=args.contamination,
        seed=args.seed,
        name=args.name,
        report=print,
    )

    with Store() as store:
        store.migrate()
        _row_id, model_uid = store.insert_model(result.model)

    print()
    print(f"model {model_uid}")
    print(f"  run it:  siar run {model_uid} <folder>")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            _json.dump(result.model.to_json(), fh, indent=2)
        print(f"  wrote    {args.out}")

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Apply a trained model to a folder of audio.

    Args:
        args: Needs ``model``, ``data``, ``threshold``, ``name``, ``no_render``, ``out``.

    Returns:
        Exit code 0, or 1 if the model is not found.
    """
    import json as _json

    from siar.infer.run import run_from_folder
    from siar.store.export import export_run
    from siar.train.model import TrainedModel

    with Store() as store:
        store.migrate()
        row = store.model_by_uid(args.model)
        if row is None:
            print(f"no model matching {args.model!r} (try `siar models`)", file=sys.stderr)
            return 1

        model = TrainedModel.from_json(_json.loads(row["model_json"]))
        print(f"model {row['model_uid']}  ({row['name']}, {row['detector']})")

        result = run_from_folder(
            model,
            int(row["id"]),
            args.data,
            store=store,
            threshold=args.threshold,
            name=args.name,
            render=not args.no_render,
            report=print,
        )

        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                _json.dump(export_run(store, result.run_uid), fh, indent=2)
            print(f"  wrote    {args.out}")

        if args.ident:
            from siar.store.export_ident import export_ident_sidecars

            n = export_ident_sidecars(store, result.run_uid, args.ident)
            print(f"  ident    {n} sidecar(s) in {args.ident}")

    print()
    print(f"run {result.run_uid}")
    print("  view it: siar dash")
    return 0


def cmd_models(_args: argparse.Namespace) -> int:
    """List trained models.

    Returns:
        Exit code 0.
    """
    with Store() as store:
        store.migrate()
        rows = store.models()

    if not rows:
        print("no models yet — train one with `siar train <folder>`")
        return 0

    print(f"{'MODEL':<28} {'NAME':<20} {'DETECTOR':<10} {'THRESHOLD':>10}  CREATED")
    for r in rows:
        print(
            f"{r['model_uid']:<28} {r['name'][:20]:<20} {r['detector']:<10} "
            f"{r['threshold']:>10.2f}  {r['created_at']}"
        )
    return 0


def cmd_runs(_args: argparse.Namespace) -> int:
    """List inference runs.

    Returns:
        Exit code 0.
    """
    with Store() as store:
        store.migrate()
        rows = store.runs()

    if not rows:
        print("no runs yet — score a folder with `siar run <model-uid> <folder>`")
        return 0

    print(f"{'RUN':<26} {'NAME':<18} {'STATUS':<8} {'FILES':>6} {'DETECTIONS':>11}  MODEL")
    for r in rows:
        print(
            f"{r['run_uid']:<26} {r['name'][:18]:<18} {r['status']:<8} "
            f"{r['n_files']:>6} {r['n_detections']:>11}  {r['model_name']}"
        )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Write a run's full results to a JSON file (or stdout).

    Args:
        args: Needs ``run`` and ``out``.

    Returns:
        Exit code 0, or 1 if the run is not found.
    """
    import json as _json

    from siar.store.export import export_run

    with Store() as store:
        store.migrate()
        try:
            payload = export_run(store, args.run)
        except KeyError as exc:
            print(f"{exc} (try `siar runs`)", file=sys.stderr)
            return 1

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, indent=2)
        n = payload["summary"]["n_detections"]
        print(f"wrote {args.out}  ({n} detections across {payload['dataset']['n_files']} files)")
    else:
        print(_json.dumps(payload, indent=2))
    return 0


def cmd_export_ident(args: argparse.Namespace) -> int:
    """Write IDent Dynamics decision sidecars for a run.

    Produces one ``<recording>.json`` per scored file that had detections. Copy or symlink the
    audio into the same folder, open it in IDent Dynamics, and the boxes appear.

    Args:
        args: Needs ``run`` and ``out``.

    Returns:
        Exit code 0, or 1 if the run is not found.
    """
    from siar.store.export_ident import export_ident_sidecars

    with Store() as store:
        store.migrate()
        try:
            n = export_ident_sidecars(store, args.run, args.out)
        except KeyError as exc:
            print(f"{exc} (try `siar runs`)", file=sys.stderr)
            return 1

    print(f"wrote {n} sidecar(s) to {args.out}")
    return 0


def cmd_dash(args: argparse.Namespace) -> int:
    """Serve the local dashboard.

    Args:
        args: Needs ``port`` and ``open``.

    Returns:
        Exit code 0.
    """
    from siar.web.server import serve

    with Store() as store:
        store.migrate()

    serve(port=args.port, open_browser=args.open)
    return 0


def cmd_db(args: argparse.Namespace) -> int:
    """Create or inspect the results database.

    Args:
        args: Needs ``db_command`` (``migrate`` or ``check``).

    Returns:
        Exit code 0.
    """
    with Store() as store:
        if args.db_command == "migrate":
            store.migrate()
            print(f"schema v{store.schema_version()} at {store.path}")
            return 0

        version = store.schema_version()
        if version == 0:
            print(f"{store.path}: not initialised (run `siar db migrate`)")
            return 1
        n_datasets = len(store.datasets())
        print(f"{store.path}")
        print(f"  schema v{version}")
        print(f"  datasets {n_datasets}")
        return 0
