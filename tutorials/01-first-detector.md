# Tutorial 01 — Your first detector

## Quick start

```bash
python tutorials/01_first_detector.py
```

Builds the corpus, trains the detector, and prints the box it found. The rest of this page
explains what that script did and why.

---

**Goal:** train an anomaly detector on audio that contains no anomalies, then show it one file
that does, and have it draw a box around the thing you hid.

**Runnable version:** [`01_first_detector.py`](01_first_detector.py) — `python tutorials/01_first_detector.py`

**Time:** about 30 seconds on CPU.

---

## The idea in one paragraph

An autoencoder is a network that is trained to copy its input to its output through a
deliberately narrow middle. Because the middle is too small to carry everything, it cannot learn
to copy *anything* — it has to spend its limited capacity on whatever it sees most often. Show it
ten thousand patches of ordinary sea noise and it becomes excellent at rebuilding ordinary sea
noise, and stays hopeless at rebuilding a dolphin whistle it has never seen. So we do not ask the
network "is this an anomaly?". We ask it to rebuild the spectrogram, and then we look at **where
it failed**. That map of failure is the detection.

---

## Step 0 — the ground truth

We are going to build our own corpus, so that we know the right answer.

```
NORMAL     = pink noise                       # 16 files, 4 seconds each
ANOMALOUS  = pink noise + a chirp we plant:
                 from t = 2.0 s to t = 3.0 s
                 sweeping 3000 Hz -> 3500 Hz
```

Pink noise is a reasonable stand-in for a natural background: energy falling off with frequency,
no structure. The chirp is the needle. **The detector is never told about it.**

We split the 16 normal files three ways, and the reason for the third split is the part people
usually get wrong:

```
train  (12 files)  the autoencoder learns "normal" from these
calib   (4 files)  used to measure what normal error looks like, and to set the threshold
test    (1 file)   the one with the chirp in it
```

`calib` must be **held out from training**. If you measure "normal error" on the same files the
network was fitted to, you measure error the network has already minimised — it looks far too
small, your threshold ends up far too low, and the detector fires on everything the moment it
sees a file it wasn't trained on. Held-out calibration is what makes the false-positive rate you
report the one you actually get.

---

## Step 1 — decide what the model will look at

```
spec = FeatureSpec(
    mode         = "pooled_linear",   # linear frequency bins, mean-pooled
    sample_rate  = 16000,
    fft_size     = 512,
    hop_size     = 128,               # -> 8 ms per frame
    fmin_hz      = 0, fmax_hz = 8000,
    n_bins       = 32,                # grid is 32 rows tall
    patch_frames = 16,                # the AE sees 16 frames (128 ms) at a time
    stride_frames= 8,                 # patches overlap by half
)
```

This object is the *entire* recipe for turning a waveform into the picture the model sees, and it
gets stored **inside the trained model**. That is what lets you re-run the model on a new folder
next month and get comparable results: the model doesn't carry "a spectrogram", it carries the
instructions for building the one it was trained on.

```
grid = build_grid(waveform, sample_rate, spec)   #  (frames, 32) float32
```

Each row of the grid is one 8 ms slice of time; each column is a frequency band.

---

## Step 2 — train on normal audio only

```
detector = ConvAE.fit(train_grids, spec, config)
```

Under the hood:

```
for each grid:
    cut it into overlapping 16-frame patches       # patchify()
normalise all patches by their median and MAD      # robust to outliers, unlike mean/std
for each epoch:
    for each batch of patches:
        add a little noise to the input            # denoising: makes copying unprofitable
        reconstruct it
        loss = mean squared error vs the CLEAN patch
        step
```

Two details in there matter more than they look.

**Median and MAD, not mean and standard deviation.** The corpus contains the very outliers we are
hunting. A mean gets dragged around by them; a median doesn't.

**The noise.** We corrupt the input but ask for the clean target. A network that has learned to
copy its input cannot do this — copying now reproduces the noise. The only way to win is to learn
what the data actually looks like. This is the cheapest insurance against the failure mode below.

### The failure mode you must know about

> An autoencoder detects things **only because it cannot represent everything.**

Make the middle too wide and the optimal solution is to just copy the input. Such a network
reconstructs anomalies *perfectly*, its error map goes flat, and it detects nothing — while
scoring better and better on reconstruction loss as it becomes more useless. This is not
hypothetical; it is what you get by default if you are not careful, and it fails **silently**.

SIAR refuses to build one:

```
latent units  =  latent_channels x (patch_frames / 2^depth) x (n_bins / 2^depth)
                 must be  <=  patch pixels / 8      else raise ValueError
```

In this tutorial: `2 x 4 x 8 = 64` latent units for a `16 x 32 = 512`-pixel patch. Exactly the
budget, eight times smaller than the input. Try setting `latent_channels` higher and SIAR will
stop you.

There is a second, subtler trap. A *narrow* bottleneck is not enough — it must also be **local**.
If the bottleneck is a fully-connected layer, every latent unit sees every pixel, so an anomaly
anywhere corrupts the whole reconstruction and the error smears across the entire frequency
axis. You still detect the event, but the box spans 250 Hz to 7750 Hz and tells you nothing.
SIAR's bottleneck is a 1x1 **convolution**: narrow, but still local, so error stays where the
anomaly is.

---

## Step 3 — score the test file, per pixel

```
error_map = detector.error_map(test_grid)      # (frames, 32) — same shape as the grid
```

The critical word is **per-pixel**. It would be easy — and it is what most implementations do —
to reduce each patch's error to a single number. Do that and the best you can ever say is
"something odd happens around 2.9 seconds". Keep the full residual and you can say "a 3–3.5 kHz
chirp, from 2.0 to 3.0 seconds". That difference *is* SIAR.

```
for each patch:
    residual = (reconstruction - input) ** 2    # keep ALL of it. Do not average.
overlap-add the residuals back onto the grid    # averaging where patches overlap
```

---

## Step 4 — make the error comparable across frequency

Raw error is not comparable between a loud band and a quiet one. Low frequencies in pink noise
carry far more energy, reconstruct less accurately in absolute terms, and would win every time. If
you skip this step, every box you ever draw lands at the bottom of the spectrogram.

So we learn, from the **calibration** files, what error is normal *for each frequency bin
individually*, and re-express every pixel as a robust z-score against its own bin:

```
baseline = fit_baseline([error_map(g) for g in calib_grids])
    per bin:  median  = median(error in this bin)
              scale   = MAD(error in this bin)  ... floored, see below

z_map = baseline.apply(error_map)
    z = (error - median) / scale
```

**The floor matters.** In a near-silent bin the MAD can be ~1e-9, and then a pixel a hair above
the median scores a z of half a million. It is not anomalous; it is just in a quiet bin — but it
will outrank every real detection. SIAR floors each bin's scale at 5% of the map's global scale,
so no bin can claim to be arbitrarily more certain than the map as a whole.

---

## Step 5 — pick the threshold

There are no labels, so the threshold is not a fact to be discovered. It is a **budget you set**:

> *What fraction of my data am I willing to have flagged?*

That is `contamination`, and it defaults to 0.001.

```
threshold = fit_threshold(calib_z_maps, method="evt", contamination=1e-3)
```

The default method is `evt` — Peaks-Over-Threshold. Rather than reading a far-tail quantile off a
handful of order statistics (noisy), it fits a Generalised Pareto distribution to the exceedances
above a high anchor and extrapolates. It is the right tool for "how big is a
one-in-a-hundred-thousand pixel". `quantile` and `robust_z` are also available.

Note the threshold is fitted on **calibration** data — held out from both training and the test
file. That is why the false-positive count it implies is honest.

In this run it comes out around **z = 18**.

---

## Step 6 — turn the hot pixels into boxes

```
mask = z_map > threshold
mask = binary_opening(mask)      # delete isolated pixels: noise, not events
mask = binary_closing(mask)      # rejoin an event the threshold fragmented
components = label(dilate(mask)) # group neighbouring fragments into one event

for each component:
    gate to within 10% of its own peak z      # <-- see below
    drop it if too small / too brief / too narrow / too ragged
    box = bounding box of what survives
    convert frames -> seconds, bins -> Hz
merge boxes that overlap heavily; keep the strongest
```

Two of those lines are doing almost all the work.

**The dynamic-range gate** (`peak_fraction`, default 0.1). The threshold answers *"is anything
here?"* and is deliberately sensitive. But a real anomaly can exceed it by **four orders of
magnitude** — in this tutorial the peak z is over 2,000,000 against a threshold of 18. The chirp's
error *halo* therefore also clears 18, the component floods outward, and the box swells to cover
the whole spectrum. You detected the event and learned nothing about where it is. Gating each
component relative to **its own peak** draws the box around the event instead of around its
shadow. Without this line, the box comes back as 750–7750 Hz. With it, 2000–4000 Hz.

**The raggedness filter** (`min_fill`, default 0.25). A component that snakes diagonally across
the map has a huge bounding box and hardly any area inside it. It is nearly always noise — and
because its box is huge, it would visually swamp every real detection in the dashboard. If fewer
than a quarter of the pixels in a box are actually above threshold, the box is thrown away.

---

## Step 7 — the result

```
PLANTED : t = 2.00 - 3.00 s    f = 3000 - 3500 Hz

DETECTED: t = 2.02 - 3.01 s    f = 2000 - 4000 Hz    score = 2,170,070

false positives on the 4 clean calibration files: 0
```

It found it. Timing is essentially exact. The frequency box is about four times wider than the
true event — it *contains* the chirp, and correctly excludes the 87% of the spectrum where
nothing happened, but it is not tight.

**That looseness is real, and worth understanding rather than hiding.** A patch-based
convolutional autoencoder reconstructs a whole patch at once, and its convolutions have a
receptive field, so reconstruction error inevitably bleeds outward from an event by a few bins.
Time resolution stays sharp because the patch strides in time and errors get averaged down
between overlapping patches; frequency resolution does not get that help. Expect SIAR's boxes to
be tight in time and generous in frequency. If you need surgical frequency edges, the box tells
you where to look — you refine within it.

---

## What to take away

1. **Held-out calibration is not optional.** Fit the threshold on data the model trained on and
   your false-positive rate is fiction.
2. **The bottleneck must be narrow *and* local.** Too wide, and the detector silently detects
   nothing. Too global, and it detects everything but cannot tell you where.
3. **Per-pixel error is the whole product.** Averaging it away costs you the frequency axis.
4. **Normalise per frequency bin**, or every box lands in the loudest band.
5. **The threshold is a budget, not a truth.** You are choosing a false-positive rate.
6. **Verify on planted anomalies.** It is the only way to know it works, because unsupervised
   detectors always produce output — plausible-looking output — whether or not they are broken.

---

## Next

Try changing things and watch them break. It is more instructive than reading:

- Set `latent_channels` to 16. SIAR raises `ValueError` — the anti-identity guard.
- Comment out the guard, train anyway, and watch the error map go flat and the detections vanish
  while the training loss gets *better*.
- Set `peak_fraction=0.0`. Watch the box swell to the whole spectrum.
- Move the chirp to 500 Hz, where pink noise is loudest. It gets much harder — and that is what
  the per-bin normalisation in Step 4 is fighting.
- Set `mode="log_mel"`. Different frequency axis, finer at the bottom, coarser at the top.
