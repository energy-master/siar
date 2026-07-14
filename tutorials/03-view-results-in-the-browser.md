# Tutorial 03 — View your results in the browser

## Quick start

```bash
siar dash --open              # http://127.0.0.1:8420 — ctrl-c to stop
```

That is the whole command. The rest of this page is what you are looking at once it opens, and
what to do when it opens empty.

---

**Goal:** take the detections from tutorial 02 and actually look at them.

**Prerequisite:** at least one completed run. Check with `siar runs`. If that says *"no runs
yet"*, do [**quick_start/**](quick_start/) — it builds a sample corpus, trains on it and scores it
in five commands, and leaves you with exactly what this page wants to show you. See
[Nothing to see](#nothing-to-see) below.

**Time:** two minutes.

---

## The dashboard reads the database, not your audio folder

This is the one thing to understand, and it is the source of the only confusing failure:

```
siar train ...  ->  writes a model      ->  ~/.siar/siar.db
siar run   ...  ->  writes detections   ->  ~/.siar/siar.db  + PNGs in ~/.siar/runs/<uid>/
siar dash       ->  reads all of it back
```

`siar dash` never touches your audio. It serves what is already in the database. So the dashboard
can only ever show you the result of a **`siar run` you have already done** — pointing it at a
folder of interesting audio does nothing, because there is nothing to point.

---

## Nothing to see

You started the dashboard, it loaded, and the run list says **"no runs yet"**. The server is fine.
The database is empty. Confirm it:

```bash
siar runs
```
```
no runs yet — score a folder with `siar run <model-uid> <folder>`
```

Generating the demo audio does not train anything. The tutorial scripts only write WAV files — the
training and scoring are the shell commands that follow. Do those:

```bash
python tutorials/quick_start/make_corpus.py                                    # writes the audio
siar train tutorials/quick_start/corpus/normal --name machine-room --epochs 60 # -> a model id
siar run <model-id> tutorials/quick_start/corpus/check                         # -> a run id
```

([quick_start/](quick_start/) explains each of those. Tutorial 02's `siar-demo` corpus works just
as well if you already have it.)

Now reload the page. The run is there — no need to restart the server, it re-reads the database on
every request.

---

## Starting it

```bash
siar dash                     # serve on http://127.0.0.1:8420
siar dash --open              # ...and open a browser at it
siar dash --port 8765         # if something already has 8420
```

```
siar dashboard on http://127.0.0.1:8420/
  database /home/you/.siar/siar.db
  ctrl-c to stop
```

It runs in the foreground until you **ctrl-c** it. If the port is taken you get
`OSError: [Errno 98] Address already in use` — either stop the other dashboard, or pass
`--port`.

It binds to **127.0.0.1 and nothing else**. The API streams local files straight off disk and has
no authentication, because it is not reachable from anywhere but this machine. That is the
security model. Do not change the bind address to `0.0.0.0` to "share it with the team".

---

## What you are looking at

The page has three columns, left to right, and they are a triage workflow:

**1. Runs.** Every `siar run` you have ever done, newest first. Pick one and you get its model,
corpus, threshold, and detections-per-hour across the top. The newest run is selected for you on
load.

**2. Recordings — sorted most anomalous first.** This is the order you want to work in, not
alphabetical. The badge is the detection count and turns red when there is something in the file;
`top` is the highest score in it. The dashboard jumps straight to the first recording with a
detection, because that is what you came for.

**3. The spectrogram**, with the detections drawn on it. Time runs left→right, frequency
bottom→top, and the axis labels under the image tell you the extent of each.

Below the image is the detection table — one row per box, with its time span, frequency band,
score and fill. **Hover a box to light up its row; hover a row to light up its box.** That is how
you tell which of five boxes is the 3.6-million-scoring one.

### The score slider — the control that matters

Real events outscore artefacts by *orders of magnitude*. On the tutorial-02 chirp file you get
three boxes: the chirp at a score around 3,600,000, and its onset and offset transients at 182 and
104. All three are above threshold. Only one is the thing you were looking for.

Drag the slider up and the weak ones vanish, leaving the event. The caption under the table tells
you what you are hiding (`1 of 3 detections shown`), so you always know the cost of where you have
set it. This is the fastest way to find the operating point you actually want — far faster than
retraining with a different `--contamination`.

**Show boxes** unticks the overlay entirely, for when you want to look at the raw spectrogram and
judge it with your own eyes before the model tells you what to think.

### Download JSON

The **Download JSON** link gives you the complete run — every box, plus the model's full
configuration. It is byte-for-byte what the CLI writes:

```bash
siar export <run-id> --out results.json
```

Same function, called from both places, so a result saved from the browser can never quietly
disagree with one written from the terminal.

---

## Why the boxes always land on the event

The spectrogram PNG is **exactly `frames` × `n_bins` pixels — one image pixel per grid cell.** No
axes, no margins, no fitted aspect ratio. The SVG overlay uses that same grid as its `viewBox`, so
a detection is drawn straight from its grid coordinates:

```js
rect.x = detection.frame_lo
rect.y = n_bins - 1 - detection.bin_hi   // SVG y grows down; the image is flipped
```

No scaling maths, no unit conversion, and no drift when CSS resizes the image. It is also why the
PNG looks so austere: render it "nicely" with margins and axes, and every box creeps off its
event.

---

## The API, if you want it

The front end is vanilla JS with no build step, talking to a small JSON API. You can use it
directly — it is just HTTP:

```bash
curl -s http://127.0.0.1:8420/api/runs                        # every run
curl -s http://127.0.0.1:8420/api/runs/<run-id>/files         # its recordings, worst first
curl -s http://127.0.0.1:8420/api/files/<file-id>/detections  # the boxes
curl -s http://127.0.0.1:8420/api/runs/<run-id>/export.json   # the whole run
```

`/api/files/<file-id>/spectrogram.png` serves the image itself.

---

## What to take away

1. **The dashboard shows the database, not a folder.** No `siar run`, nothing to see.
2. **Triage from the top** — worst recording first, then the slider, then read down.
3. **Scores span orders of magnitude.** Above threshold does not mean interesting.
4. **The browser and the CLI export the same file.** Use whichever is to hand.
5. **Localhost only.** No auth, by design.

---

## Next

- Score a second folder with the same model and compare the two runs side by side in the run list.
- Re-run at a deliberately loose threshold (`siar run <model> siar-demo/check --threshold 100`) and
  use the slider to find where the real events stop and the noise starts.
- `$SIAR_HOME=./results siar dash` — keep a project's database, PNGs and dashboard beside the
  project instead of in your home directory.
