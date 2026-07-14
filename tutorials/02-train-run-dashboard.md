# Tutorial 02 — Train, run, and see the results

## Quick start

```bash
python tutorials/02_train_run_dashboard.py          # build the demo corpus
siar train siar-demo/normal --name harbour-baseline # -> prints a model id
siar run <model-id> siar-demo/check                 # score a different folder
siar dash --open                                    # look at what it found
```

The rest of this page walks through those four commands and what comes back from each.

---

Tutorial 01 drove the pipeline by hand, in Python, to show you what it does. This one uses the
actual product: three commands, a database, and a dashboard.

**Goal:** train a model on a folder of normal audio, use it to score a *different* folder, and
look at what it found in a browser.

**Runnable version:** [`02_train_run_dashboard.py`](02_train_run_dashboard.py) builds the demo
corpus. Then you type three commands.

**Time:** about a minute.

---

## The shape of the thing

```
siar train  <folder-of-normal-audio>     ->  a model, saved in the database
siar run    <model> <folder-to-check>    ->  detections, saved in the database
siar dash                                ->  a browser, showing them
```

That is the whole product. Everything else is detail.

The important part is the **middle step's folder is not the first step's folder**. You train once,
on audio you believe is unremarkable, and then re-run that model on new material for as long as
it stays relevant. A model is a portable artefact — it carries its own feature recipe, its own
normalisation, and its own threshold, so scoring a folder next month gives results comparable to
scoring one today.

---

## Step 0 — build the demo corpus

```bash
python tutorials/02_train_run_dashboard.py
```

That writes two folders under `./siar-demo/`:

```
normal/   20 files of pink noise                        <- what "normal" means
check/     4 files of pink noise                        <- should come back clean
           1 file with a chirp   at t=2.0-3.0s, 3-3.5 kHz
           1 file with 2 clicks  at t=1.5s and 2.5s, ~5.5 kHz
```

Again: we plant the anomalies so we can *check the answer*. The detector is told nothing about
them.

---

## Step 1 — train

```bash
siar train siar-demo/normal --name harbour-baseline
```

```
training on 20 file(s), 1.3 min of audio
  spec: pooled_linear 64 bins, 1024-pt FFT @ 16000 Hz
  split: 14 train / 3 val / 3 calib
  building grids...
  training conv_ae...
  epoch   1/15  loss 5.43272  val 5.73712
  epoch  15/15  loss 0.51676  val 0.51644
  trained: 12,211 parameters
  calibrating on held-out files...
  threshold: z = 18.79 (evt, contamination 0.001)

model model-20260714T123818Z-54bf
  run it:  siar run model-20260714T123818Z-54bf <folder>
```

Read the `split` line. Of 20 files, **14** train the network, **3** give a validation loss, and
**3** are reserved for calibration — measuring what normal reconstruction error looks like, and
choosing the threshold. Those 3 are never seen during training, and that is not a detail:

> Fit the threshold on audio the network trained on and you measure error it has already been
> optimised to minimise. The threshold comes out far too low, and the detector then fires on
> everything the moment it meets a recording it has not seen.

The one knob that matters is `--contamination` (default `0.001`). It is not a quality setting; it
is a **budget**:

> *What fraction of pixels am I willing to have flagged on audio I believe is normal?*

Lower it for fewer, higher-confidence detections. Raise it to catch more and tolerate more noise.
That is the entire trade, stated honestly — there is no labelled data here to tune against, so
there is no "correct" value to discover.

The model is now in `~/.siar/siar.db`. Add `--out model.json` if you also want it as a file you
can send someone.

```bash
siar models
```
```
MODEL                        NAME                 DETECTOR    THRESHOLD  CREATED
model-20260714T123818Z-54bf  harbour-baseline     conv_ae         18.79  2026-07-14 12:38:18
```

---

## Step 2 — run it on audio it has never seen

```bash
siar run model-20260714T123818Z-54bf siar-demo/check
```

(A unique prefix works too — `siar run model-2026 …`.)

```
model model-20260714T123818Z-54bf  (harbour-baseline, conv_ae)
  run run-20260714T123829Z-3b52  (threshold z = 18.79)
  [1/6] chirp_at_2s: 3 detection(s)  top z=3,636,353
  [2/6] clicks_at_1.5_2.5s: 2 detection(s)  top z=2,495,716
  [3/6] quiet_00: 0 detection(s)
  [4/6] quiet_01: 0 detection(s)
  [5/6] quiet_02: 0 detection(s)
  [6/6] quiet_03: 0 detection(s)
  5 detection(s) over 0.01 h  (750.0 per hour)
```

Both planted files found. All four clean files clean.

Check where the detections actually landed:

| planted | detected |
|---|---|
| chirp — 2.00–3.00 s, 3000–3500 Hz | **1.97–3.01 s, 2500–3625 Hz** |
| click — 1.5 s, ~5500 Hz | **1.46–1.55 s, 4500–5625 Hz** |
| click — 2.5 s, ~5500 Hz | **2.43–2.56 s, 5000–5625 Hz** |

Note the two extra detections on the chirp file, scoring 182 and 104 against the real one's
3,636,353. They are the sharp onset and offset transients of the chirp — arguably real, arguably
noise. They are **four orders of magnitude weaker**, which is why the dashboard has a score
slider: in practice you triage from the top and stop when the scores go quiet.

**`detections per hour` is the number to watch.** If a run reports 40,000/hour on data you believe
is normal, the model is wrong and no amount of staring at boxes will fix it. It is the fastest
sanity check there is.

Useful flags: `--threshold Z` overrides the model's threshold for one run without retraining;
`--no-render` skips the PNGs for a headless bulk job; `--out results.json` writes the export
immediately.

---

## Step 3 — look at it

```bash
siar dash --open
```

```
siar dashboard on http://127.0.0.1:8420/
```

The dashboard is a plain HTML page served by Python's standard library — no Flask, no PHP, no
build step, nothing to install. It binds to localhost only.

You get:

- **Runs**, down the left. Pick one.
- **Recordings**, sorted **most anomalous first** — the order you actually want to triage in. The
  badge is the detection count; it turns red when there is something to look at.
- **The spectrogram**, with the detections drawn on it. Hover a box to see its time, frequency and
  score; hover a table row to light up its box.
- **A score slider.** Drag it up and the weak transients disappear, leaving the real event. This
  is the fastest way to find the operating point you actually want.
- **Download JSON** — the complete run, every box, the model's full configuration.

### The one design decision worth knowing

The spectrogram PNG is **exactly `frames` × `n_bins` pixels — one image pixel per grid cell.** No
axes, no padding, no fitted aspect ratio. The SVG overlay uses that same grid as its `viewBox`, so
a detection is drawn straight from its grid coordinates:

```js
rect.x      = detection.frame_lo
rect.y      = n_bins - 1 - detection.bin_hi   // SVG y grows down; the image is flipped
rect.width  = detection.frame_hi - detection.frame_lo + 1
rect.height = detection.bin_hi - detection.bin_lo + 1
```

No scaling maths, no unit conversion, no drift when the image is resized — and the browser never
has to know what an FFT is. Render the image "nicely" with margins and that guarantee is gone and
the boxes creep off the events.

---

## Step 4 — take the results with you

```bash
siar export run-20260714T123829Z-3b52 --out results.json
```

Or click **Download JSON** in the dashboard. They produce the identical file — same function,
called from both places, so a result saved from the browser can never quietly disagree with one
written by the CLI.

```json
{
  "format": "siar-detections-v1",
  "model": { "model_uid": "...", "spec": {...}, "threshold": 18.79, "config": {...} },
  "dataset": { "path": "siar-demo/check", "n_files": 6 },
  "summary": { "n_detections": 5, "detections_per_hour": 750.0, "files_with_detections": 2 },
  "files": [
    { "name": "chirp_at_2s",
      "detections": [
        { "t_start": 1.97, "t_end": 3.01, "f_low": 2500.0, "f_high": 3625.0,
          "score": 3636353.0, "peak_score": 4712001.5, "area": 402, "fill": 0.69 }
      ] }
  ]
}
```

The model's whole configuration rides along with the results, so a detection is always traceable
back to the exact thing that produced it.

---

## Where the data lives

```
~/.siar/
  siar.db          SQLite. Models, runs, files, detections.
  runs/<uid>/      the spectrogram PNGs
```

SQLite ships with Python, so there is no database to install, no server to start, and no
configuration. Point `$SIAR_HOME` somewhere else to keep a project's results beside the project.

```bash
siar db check     # what's in there?
siar runs         # every run
siar models       # every model
```

---

## What to take away

1. **Train on normal, run on unknown.** Anything *common* in your training folder will not be
   flagged — that is the definition of the method, not a limitation to work around.
2. **Calibration is held out.** It is why the false-positive rate is honest.
3. **`--contamination` is a budget, not a quality dial.** You are choosing a false-positive rate.
4. **Watch detections-per-hour.** It catches a broken model faster than any metric.
5. **Triage from the top.** Real events outscore artefacts by orders of magnitude. Use the slider.
6. **A model is portable.** It carries its feature recipe, normalisation and threshold, so it can
   be re-run on new audio months later and mean the same thing.

---

## Next

- Run the model against the *training* folder (`siar run <model> siar-demo/normal`). It should be
  nearly silent. If it isn't, the model has overfitted and the threshold is too low.
- Score the same folder at a lower threshold: `siar run <model> siar-demo/check --threshold 100`.
  Watch the false-positive count climb, and decide what you can live with.
- Try `siar train --mode log_mel`. A different frequency axis — finer at the bottom, coarser at
  the top.
- Point it at your **own** audio. Then plant something in a copy of it and confirm SIAR finds it,
  before you trust a single box it draws.
