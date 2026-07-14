# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

**SIAR** — Signal Intelligence and Reconnaissance. A standalone, pip-installable Python library
for **unsupervised** acoustic anomaly detection. Point it at a folder of audio, run an
optimisation, get a trained detector that draws **2-D boxes** (time × frequency) around whatever
does not look like the rest of the corpus. Results land in a local SQLite database and a local
web dashboard; everything is exportable as JSON.

The pipeline, end to end:

```
audio -> STFT -> band grid -> patches -> detector -> per-pixel error map -> z-map -> boxes
```

## Stack

- Python ≥ 3.13, setuptools, `src/` layout. Console script `siar`.
- Core deps: numpy, scipy, soundfile, **torch**, optuna. Extras: `mel` (librosa), `viz`
  (matplotlib), `dev` (pytest).
- **All models are torch.** So is the signal frontend — `torch.stft`, not a hand-rolled FFT.
- Dashboard: stdlib `http.server` + vanilla JS/HTML/CSS, served from `web/static/`. **No Flask, no
  PHP, no build step, no CDN.** A pip-installed tool must not require a PHP runtime. Binds to
  127.0.0.1 only — the API streams local files and has no auth because it is not reachable.
- Spectrogram PNGs are written by `viz/png.py` (stdlib `zlib`). **Not Pillow, not matplotlib** —
  the library must not pull an imaging stack in just to save a picture.
- CPU-first. There is no GPU on the target box; the 40 cores are used by running Optuna trials in
  parallel processes, not by making one trial faster.

**SIAR is standalone.** It has no dependency on `brahma_framework` or any other Vixen project,
and must not acquire one.

## The three things that will silently break this

Read these before changing anything in `models/` or `detect/`.

1. **The autoencoder bottleneck must be narrow AND local.**
   - *Too wide* → it learns the identity map, reconstructs anomalies perfectly, the error map goes
     flat, and it detects **nothing** — while the training loss gets *better*. Guarded by
     `MAX_LATENT_FRACTION` + `validate_config()` in `models/conv_ae.py`.
   - *Too global* (e.g. a `Linear` bottleneck) → it still detects, but reconstruction error smears
     across every frequency and boxes come back spanning the whole spectrum. The bottleneck is a
     **1×1 conv** for exactly this reason. Do not "simplify" it to a Linear layer.

2. **Per-bin normalisation is not optional** (`detect/normalise.py`). Raw reconstruction error is
   systematically larger in loud bands. Skip the whitening and every box lands in the loudest part
   of the spectrum regardless of what is actually anomalous. The MAD floor
   (`MIN_MAD_FRACTION`) matters too: an unfloored quiet bin produces z-scores of ~500,000 and
   outranks every real detection.

3. **Calibration must be held out.** The threshold is fitted on files the model never trained on.
   Fit it on training data and the reported false-positive rate is fiction.

## The objective (when HPO lands)

Optimising held-out **reconstruction loss is an anti-objective** — it selects the
highest-capacity autoencoder, i.e. the one closest to the identity map, i.e. the one that detects
nothing. The plan is to optimise **detection of synthetically injected anomalies** on a held-out
split, which directly punishes the degenerate solution. See the project plan.

## Layout

```
src/siar/
  features/   spec.py (FeatureSpec — the contract, stored ON the model)
              frontend.py (torch: audio -> grid; bin_edges_hz vs row_support_hz)
              patches.py (patchify / overlap_add — exact inverses)
  models/     base.py (Detector protocol), registry.py (@register_detector), conv_ae.py
  detect/     normalise.py (per-bin z), threshold.py (evt/quantile/robust_z), boxes.py
  data/       audio.py (load/mono), dataset.py (scan_folder, split_files — split by FILE)
  train/      model.py (TrainedModel bundle), fit.py (train_from_folder)
  infer/      run.py (run_from_folder — score a folder, persist detections + PNGs)
  store/      schema.py (SQLite DDL), db.py (Store), export.py (siar-detections-v1)
  viz/        png.py (stdlib PNG encoder), colormap.py, spectrogram.py
  web/        server.py (stdlib http.server) + static/ (vanilla JS, no build step)
  cli/        main.py (argparse + dispatch), commands.py (cmd_*)
tutorials/    01 (the method, in Python) and 02 (the product: train/run/dash)
tests/        pytest — 50 tests
```

**The spectrogram PNG is exactly `frames` x `n_bins` pixels — one image pixel per grid cell** —
and the dashboard's SVG overlay uses that grid as its `viewBox`. So a box is drawn straight from
its grid coordinates with no scaling maths in JS and no drift. Render the PNG "nicely" (margins,
axes, fitted aspect ratio) and every box creeps off its event. Don't.

**The database is stdlib `sqlite3`.** It is not a dependency and must never become one. It
self-creates via `Store.migrate()`; there is no migration tool and no server.

`bin_edges_hz` (monotone rendering axis) and `row_support_hz` (per-row band, may overlap) are
**different functions for different jobs**. Boxes use `row_support_hz`. Using the edges array for
a mel box reports a 6 kHz tone as 5573–5926 Hz — a band that does not contain it.

## Conventions

- Header `# Vixen Intelligence c.2026`; `from __future__ import annotations`; explicit `__all__`.
- Google-style docstrings (Args/Returns/Raises) on every module, class and function. Module
  docstrings state the **data contract** and explain *why*, not just what.
- Frozen `@dataclass(slots=True)` for value types. `float32` for all signal arrays.
- **Torch imported lazily inside functions**, so `import siar` stays cheap and torch-free.
- **Never pickle a model.** Models are declarative JSON with base64 weights, loaded with
  `torch.load(..., weights_only=True)` so opening one can never execute code.
- Registries are explicit decorators into plain dicts. No `eval()`, no entry-point scanning, no
  `import *`. `siar/models/__init__.py` *is* the plugin manifest.

## Adding a detector

1. Write `src/siar/models/<name>.py` implementing the `Detector` protocol (`models/base.py`).
   The only method that matters is `error_map(grid) -> (frames, n_bins)` — **per-pixel**.
2. Decorate with `@register_detector("<name>")`.
3. Add one import line to `src/siar/models/__init__.py`.

If step 3 requires touching anything else, the seam is wrong — fix the seam.

## Testing

```bash
pytest                                       # 50 tests, ~5s
python tutorials/01_first_detector.py        # the method, end to end
python tutorials/02_train_run_dashboard.py   # then: siar train / run / dash
```

`tests/test_e2e.py` is the one that matters: it trains on pink noise, plants a chirp at a known
time and frequency, and requires SIAR to box it and to find nothing on clean audio. There is no
real ground truth in unsupervised detection, so we manufacture it. **Any change to `models/` or
`detect/` must keep that test green.**

## Commits

Git identity is **Rahul Tandon (Vixen Intelligence)** — `rahul@vixenintelligence.com`.
