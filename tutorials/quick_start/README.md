# SIAR quick start

**Train a detector, score audio it has never heard, and look at what it found — in five commands.**

Everything here is synthetic and self-contained. The anomalies are planted at times and
frequencies *we choose*, so at the end you can check SIAR's boxes against the truth instead of
squinting at them and hoping. Do exactly this on your own audio before you trust a single box it
draws.

---

## 0. Install

From a clean clone, on Linux or macOS:

```bash
git clone <this-repo> siar
cd siar

python3.13 -m venv .venv          # Python 3.13 or newer is required
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -e .                  # numpy, scipy, soundfile, torch, optuna
siar version                      # siar 0.1.0
```

CPU only, and that is deliberate — there is no GPU in the design. Everything below runs in about
a minute on a laptop.

> **macOS.** The `pip install -e .` above is all you need: the `torch` and `soundfile` wheels are
> prebuilt for both Apple Silicon and Intel, and `soundfile` bundles libsndfile, so there is no
> Homebrew step. If you are on an older Python, `brew install python@3.13` first — SIAR needs
> ≥ 3.13 and will not install on 3.12.

---

## 1. Build the sample corpus

```bash
python tutorials/quick_start/make_corpus.py
```

```
wrote tutorials/quick_start/corpus/normal  24 files, background only  (train on this)
wrote tutorials/quick_start/corpus/check   6 clean + 3 with planted events  (score this)

planted, for you to check SIAR's boxes against:
  chirp     3.0–4.0 s               2000–2800 Hz (sweeping)
  impacts   1.0 s, 2.5 s, 4.0 s     ~6000 Hz (broadband)
  squeal    1.5–3.5 s               5000 Hz (steady)
```

Two folders, both 16 kHz mono WAV:

```
corpus/normal/   24 files — a machine room: broadband floor + a steady electrical hum
corpus/check/     6 files — same machine room, nothing wrong      <- must come back clean
                  1 file  — chirp    (a sweep: something moving)
                  1 file  — impacts  (transients: something struck)
                  1 file  — squeal   (a steady tone: a bearing going)
```

The background is a **hum, not white noise**, on purpose. A detector that only works on featureless
noise has told you nothing; this one has to learn that 100/200/300 Hz is *normal* and a 5 kHz tone
is not.

The detector is told nothing about any of this. It never sees a label, and it never sees the
`check` folder during training.

---

## 2. Train on normal

```bash
siar train tutorials/quick_start/corpus/normal --name machine-room --epochs 60
```

```
training on 24 file(s), 2.0 min of audio
  spec: pooled_linear 64 bins, 1024-pt FFT @ 16000 Hz
  split: 16 train / 4 val / 4 calib
  building grids...
  training conv_ae...
  epoch   1/60  loss 5479.12819  val 5405.62675
  epoch  30/60  loss 2780.39010  val 2685.22942
  epoch  60/60  loss 1015.21999  val  991.57515
  trained: 12,211 parameters
  calibrating on held-out files...
  threshold: z = 4.35 (evt, contamination 0.001)

model model-20260714T151534Z-287e
  run it:  siar run model-20260714T151534Z-287e <folder>
```

**Copy that model id — the next command needs it.** (A unique prefix works too: `model-20260714T15`.)

Three things in that output are worth more than the rest:

- **`split: 16 train / 4 val / 4 calib`.** Of 24 files, four are held back to *calibrate the
  threshold*. They are never trained on. Fit a threshold on audio the network has already been
  optimised to reconstruct and it comes out far too low — the detector then fires on everything the
  moment it meets a recording it has not seen.
- **`--epochs 60`, not the default 15.** This background has structure (the hum), and at 15 epochs
  the loss is still falling steeply — an undertrained model reconstructs the hum badly, that error
  drowns the real events, and **the chirp is missed entirely**. Try it: drop `--epochs` and watch
  `chirp: 0 detection(s)` come back. Watch the val loss flatten out; that is when you have trained
  enough.
- **`threshold: z = 4.35`.** Chosen for you, from the held-out files, at the false-positive budget
  you asked for.

The one knob that matters is `--contamination` (default `0.001`). It is not a quality dial, it is
a **budget**: *what fraction of pixels am I willing to have flagged on audio I believe is normal?*
Lower it for fewer, higher-confidence detections; raise it to catch more and tolerate more noise.
That is the whole trade — there is no labelled data here, so there is no "correct" value to find.

The model is now in `~/.siar/siar.db`. Add `--out model.json` to also get it as a file you can
send someone.

---

## 3. Run it on audio it has never heard

```bash
siar run model-20260714T151534Z-287e tutorials/quick_start/corpus/check
```

```
model model-20260714T151534Z-287e  (machine-room, conv_ae)
  run run-20260714T151539Z-cae7  (threshold z = 4.35)
  [1/9] chirp: 1 detection(s)  top z=138,619
  [2/9] impacts: 3 detection(s)  top z=248,906
  [3/9] quiet_00: 0 detection(s)
  [4/9] quiet_01: 0 detection(s)
  [5/9] quiet_02: 0 detection(s)
  [6/9] quiet_03: 0 detection(s)
  [7/9] quiet_04: 0 detection(s)
  [8/9] quiet_05: 0 detection(s)
  [9/9] squeal: 1 detection(s)  top z=65,534
  5 detection(s) over 0.01 h  (400.0 per hour)

run run-20260714T151539Z-cae7
  view it: siar dash
```

**All three planted files found. All six clean files clean.** Now check *where* the boxes landed —
this is the only honest test there is:

| planted | detected |
|---|---|
| chirp — 3.0–4.0 s, 2000–2800 Hz | **2.96–4.00 s, 1750–3000 Hz** |
| impact — 1.0 s, ~6000 Hz | **0.96–1.01 s, 5875–6125 Hz** |
| impact — 2.5 s, ~6000 Hz | **2.46–2.51 s, 5875–6125 Hz** |
| impact — 4.0 s, ~6000 Hz | **3.97–4.02 s, 5875–6125 Hz** |
| squeal — 1.5–3.5 s, 5000 Hz | **1.50–3.44 s, 4750–5250 Hz** |

Every box is on its event, in time *and* in frequency. Nothing was hallucinated in the quiet files.
That is what a working detector looks like.

**`detections per hour` is the number to watch.** If a run reports 40,000/hour on audio you believe
is normal, the model is wrong, and no amount of staring at boxes will fix it. It is the fastest
sanity check there is.

Useful flags: `--threshold Z` overrides the model's threshold for one run without retraining;
`--no-render` skips the PNGs for a headless bulk job; `--out results.json` writes the full export
immediately.

---

## 4. Look at it

```bash
siar dash --open
```

```
siar dashboard on http://127.0.0.1:8420/
  database /home/you/.siar/siar.db
  ctrl-c to stop
```

`--open` launches your default browser (it works on macOS too). If port 8420 is busy, use
`siar dash --port 8765`. It runs in the foreground until you **ctrl-c** it, and binds to
**127.0.0.1 only** — there is no auth because it is not reachable from anywhere else. Don't
"helpfully" move it to `0.0.0.0`.

In the browser:

- **Runs**, on the left — pick the one you just made.
- **Recordings**, sorted **most anomalous first**. `impacts` is at the top, the six `quiet_*` files
  at the bottom with a `0` badge. That is the order to triage in.
- **The spectrogram** with the boxes drawn on it. Hover a box to see its time, frequency and score;
  hover a table row to light up its box.
- **The score slider.** Drag it up and the weak boxes vanish. Real events outscore artefacts by
  orders of magnitude, so this is the fastest way to find your operating point — much faster than
  retraining with a different `--contamination`.
- **Download JSON** — the whole run, every box, plus the model's full config.

> **Nothing in the dashboard?** It reads the *database*, not your audio folder. If the run list is
> empty, you have not done step 3 — `siar dash` can only ever show you a `siar run` you have
> already done. Check with `siar runs`.

[Tutorial 03](../03-view-results-in-the-browser.md) covers the dashboard properly.

---

## 5. Take the results with you

```bash
siar export run-20260714T151539Z-cae7 --out results.json
```

Identical to the browser's **Download JSON** button — same function, called from both places, so a
result saved from the browser can never quietly disagree with one written from the terminal.

```json
{
  "format": "siar-detections-v1",
  "model": { "model_uid": "...", "spec": {...}, "threshold": 4.35, "config": {...} },
  "summary": { "n_detections": 5, "detections_per_hour": 400.0, "files_with_detections": 3 },
  "files": [
    { "name": "impacts",
      "detections": [
        { "t_start": 2.46, "t_end": 2.51, "f_low": 5875.0, "f_high": 6125.0, "score": 248906.0 }
      ] }
  ]
}
```

The model's whole configuration rides along with the results, so any detection is traceable back to
the exact thing that produced it.

---

## The five commands, together

```bash
pip install -e .
python tutorials/quick_start/make_corpus.py
siar train tutorials/quick_start/corpus/normal --name machine-room --epochs 60
siar run <model-uid> tutorials/quick_start/corpus/check
siar dash --open
```

## Where it all lives

```
~/.siar/
  siar.db          SQLite — models, runs, files, detections
  runs/<uid>/      the spectrogram PNGs
```

SQLite ships with Python: no database to install, no server, no config. Set `$SIAR_HOME` to keep a
project's results beside the project instead:

```bash
export SIAR_HOME=./workspace     # then train/run/dash all use ./workspace
```

Inspect it any time:

```bash
siar models      # every model
siar runs        # every run
siar db check    # what's in the database
```

## Starting over

```bash
rm -rf ~/.siar tutorials/quick_start/corpus     # wipes models, runs, PNGs and the demo audio
```

---

## What to take away

1. **Train on normal, run on unknown.** Anything *common* in the training folder will never be
   flagged — that is the definition of the method, not a limitation to work around.
2. **Calibration is held out**, which is why the false-positive rate is honest.
3. **Undertraining looks like a detection failure, not a training failure.** The loss was still
   falling and the chirp went missing. Train until the val loss flattens.
4. **`--contamination` is a budget, not a quality dial.**
5. **Watch detections-per-hour.** It catches a broken model faster than any other number.
6. **Plant something you can check.** There is no ground truth in unsupervised detection, so
   manufacture some — that is what this corpus is, and you should do the same to a copy of your own
   audio before trusting a single box.

## Next

- [Tutorial 01](../01-first-detector.md) — the same pipeline by hand, in Python, one stage at a time.
- [Tutorial 02](../02-train-run-dashboard.md) — the CLI in more depth.
- [Tutorial 03](../03-view-results-in-the-browser.md) — the dashboard on its own.
- Then point it at your **own** audio.
