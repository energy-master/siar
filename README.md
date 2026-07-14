# SIAR — Signal Intelligence and Reconnaissance

Unsupervised acoustic anomaly detection. Point it at a folder of audio, and it learns what
"normal" sounds like — then draws boxes around whatever isn't.

No labels required. Nothing to annotate.

```
audio -> STFT -> band grid -> patches -> autoencoder -> per-pixel error -> boxes
```

Detections are **two-dimensional**: a start and end time *and* a low and high frequency, so a
result is "a 3–4 kHz chirp between 2.0 and 3.0 seconds", not just "something happened around 2.9
seconds".

## Install

```bash
pip install git+https://github.com/<org>/siar.git
```

Requires Python ≥ 3.13. Runs on CPU.

## Quick start

Three commands. Train on audio you believe is normal, score audio you haven't checked, look at
what came back.

```bash
siar train /path/to/normal-audio --name baseline   # learn "normal"
siar run <model-uid> /path/to/audio-to-check       # find what isn't
siar dash --open                                   # look at it
```

Work through **[tutorials/02-train-run-dashboard.md](tutorials/02-train-run-dashboard.md)** — it
builds a corpus with anomalies planted at known times and frequencies, so you can check the
answer:

```
[1/6] chirp_at_2s: 3 detection(s)  top z=3,636,353
[2/6] clicks_at_1.5_2.5s: 2 detection(s)  top z=2,495,716
[3/6] quiet_00: 0 detection(s)
...
```

| planted | detected |
|---|---|
| chirp — 2.00–3.00 s, 3000–3500 Hz | **1.97–3.01 s, 2500–3625 Hz** |
| click — 1.5 s, ~5500 Hz | **1.46–1.55 s, 4500–5625 Hz** |
| click — 2.5 s, ~5500 Hz | **2.43–2.56 s, 5000–5625 Hz** |

**[tutorials/01-first-detector.md](tutorials/01-first-detector.md)** does the same thing in
Python, one step at a time, and explains *why* each step is there. Start there if you want to
understand the method rather than just use it.

## Where the results go

```
~/.siar/
  siar.db          SQLite — models, runs, detections
  runs/<uid>/      spectrogram PNGs
```

SQLite ships with Python, so there is **no database to install and nothing to configure**. Set
`$SIAR_HOME` to keep a project's results beside the project.

The dashboard is served by Python's standard library — no Flask, no PHP, no build step, no CDN.
It binds to localhost.

## Status

Under active development. **Train → run → results → dashboard works end to end.**

Working:

- `siar scan` / `train` / `run` / `models` / `runs` / `export` / `dash` / `db`
- feature frontend (`torch.stft`), in `pooled_linear` and `log_mel` modes
- the `conv_ae` detector — a PyTorch denoising convolutional autoencoder with the anti-identity
  bottleneck guard
- per-pixel error maps, per-bin robust normalisation, EVT / quantile / robust-z thresholds
- 2-D box extraction, dynamic-range gate, raggedness filter
- portable models: one JSON document carrying the feature recipe, weights, normalisation and
  threshold, so a model re-run months later means the same thing
- SQLite results store, spectrogram PNGs, local dashboard, JSON export
- 50 tests, including an end-to-end proof against planted anomalies

Not done yet:

- **`siar optimise`** — the Optuna hyperparameter sweep, and the dashboard views for it
- the feature cache (every run currently recomputes its STFTs)
- audio playback of a detection in the dashboard

## The one thing to understand

An autoencoder detects anomalies **only because it cannot represent everything**. Its bottleneck
has to be narrow enough that it cannot learn to simply copy its input — otherwise it reconstructs
anomalies perfectly, its error map goes flat, and it finds nothing, all while its training loss
gets *better*. It fails silently, and it looks like it's working.

SIAR refuses to build such a model (`ValueError`, not a warning), and its bottleneck is
convolutional rather than fully-connected so that reconstruction error stays *where the anomaly
is* instead of smearing across the whole spectrum.

That is most of the engineering. The rest is bookkeeping.

## Licence

Proprietary. © Vixen Intelligence.
