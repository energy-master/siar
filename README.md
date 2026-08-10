# siar-scanner

Run the IDent Dynamics structure scanners over a folder of recordings, on your own machine.

Point it at a root folder of WAV or FLAC. It scans every recording and writes an output folder
you drag into the IDent Dynamics web app — every box the scanner found, on every file, with a
spectrogram preview on every lane.

```
audio folder -> STFT -> scanning algorithm -> boxes + thumbnails -> a folder the app opens
```

Detections are **two-dimensional**: a start and end time *and* a low and high frequency, so a
result is "a down-sweep from 3.2 to 2.1 kHz between 4.0 and 4.6 seconds", not just "something
happened around 4 seconds".

## Install

```bash
pip install git+https://github.com/energy-master/siar-scanner.git
```

Requires Python ≥ 3.11. Runs on CPU. Pulls in numpy and soundfile, and nothing else.

## Quick start

Three commands. Sign in, see what you can run, run it.

```bash
siar-scanner login                                    # your IDent Dynamics account
siar-scanner algorithms                               # what your account can download
siar-scanner run ~/survey-audio -a all_structures -o ~/survey-scan
```

Then open `~/survey-scan` in IDent Dynamics — **Open folder** in the app — and work through
what came back.

---

# Tutorial

This walks the whole thing end to end on a folder you already have. Nothing here is synthetic:
if you have a directory of survey recordings, use it.

## 1. Sign in

The scanning algorithms live in your IDent Dynamics installation, not in this package. You get
them with the same username and password you use for the web app.

```bash
$ siar-scanner login
IDent Dynamics username or email: rahul@vixenintelligence.com
Password:
Signed in to https://goident.ai as rahul.
Token saved to /home/you/.siar-scanner/credentials.json
```

The token is cached at mode 0600 and is what every later command uses. In a script or a
container, skip the prompt entirely:

```bash
export SIAR_SCANNER_URL=https://goident.ai
export SIAR_SCANNER_TOKEN=...      # from a previous login, or your account page
```

`siar-scanner whoami` says who you are signed in as; `siar-scanner logout` forgets the token
locally. Revoking it properly is done from your account page in the app.

## 2. See what you can run

```bash
$ siar-scanner algorithms
SLUG                      FINDS                  RUNS HERE  DESCRIPTION
------------------------  ---------------------  ---------  ------------------------------------
all_structures            sweep, tonal, click,   yes        The survey. Boxes every significant
                          patch, blob                       structure in the recording — tonals,
                                                            sweeps, clicks, bursts — with no
                                                            target class. Reach for this when the
                                                            question is "what is in here".
all_structures_sensitive  sweep, tonal, click,   yes        Lower bar, for faint structure in a
                          patch, blob                       quiet recording.
fuzzy_hp_alpha            click, click_train     yes        Harbour-porpoise click candidates,
                                                            90-150 kHz. High recall by design:
                                                            it finds clicks and groups them into
                                                            trains, and leaves the rejecting to
                                                            you.
```

The descriptions are written by your installation's super user in the admin panel, so they say
what each algorithm is actually being used for, not what it was called when it was built.

Add `--params` to see what each one lets you tune:

```bash
$ siar-scanner algorithms --params
...
all_structures — parameters (--param name=value):
NAME          TYPE    DEFAULT  MEANING
------------  ------  -------  ---------------------------------------------------------
minSigma      number  2.5      How far above its own local background a cell must stand.
maxRegions    number  400      Cap on boxes returned, keeping the strongest by rank.
fmin          number  0        Low edge of the band to scan, in Hz.
fmax          number  null     High edge; null means up to Nyquist.
```

**"RUNS HERE" says no?** An obfuscated bundle is pinned to an operating system, a CPU
architecture and a Python minor version. `siar-scanner version` prints the tag your machine
reports; ask whoever publishes the algorithms for a build with that tag.

## 3. Look at the folder before you scan it

```bash
$ siar-scanner scan ~/survey-audio
412 recording(s) under /home/you/survey-audio
27.40 h of audio

SAMPLE RATE  FILES
-----------  -----
96000 Hz       412
```

Headers only, so this takes about a second even on a corpus that will take hours to scan. It is
worth the second: if that table shows more than one sample rate, every frequency band maps to a
different bin in each group, and scanning the folder as one thing is really scanning several.

## 4. Run a scan

```bash
$ siar-scanner run ~/survey-audio --algorithm all_structures --out ~/survey-scan
algorithm  all_structures (linux-x86_64-cp313)
scanning   /home/you/survey-audio
output     /home/you/survey-scan
[412/412] station-c/2025-09-08T1400.wav: 37 structures

412 file(s), 27.40 h of audio, in 1841.2s
9,043 structures: 122 click_train, 4188 click, 219 sweep, 3901 tonal, 613 patch
1 file(s) error

Output folder: /home/you/survey-scan
Open it in IDent Dynamics (Open folder) to see the boxes on the spectrogram.
```

The first run downloads the algorithm and caches it under `~/.siar-scanner/algorithms/`. Every
run after that is completely offline — useful on a vessel, required in a lab with no network.

Useful flags:

| Flag | Why |
|---|---|
| `--resume` | Carry on where an interrupted run stopped. Safe to pass always. |
| `--limit 20` | Trial the algorithm on twenty files before committing to a corpus. |
| `--link` | Hardlink the audio into the output folder instead of copying it. Same filesystem only; falls back to a copy. |
| `--param minSigma=3.5` | Tune the algorithm. Repeatable. |
| `--fmin 2000 --fmax 20000` | Restrict the band. |
| `--fft 2048 --hop 512` | Override the analysis grid. Only if you know why — each algorithm carries the grid it was tuned at. |
| `--channel left` | Which channel, on a multi-channel recording. Default is a mix. |
| `--no-thumbnails` | Skip the lane previews. Saves a few milliseconds per file and makes the folder less useful. |

Interrupt it with Ctrl-C at any point; nothing is half-written, and `--resume` picks it up.

## 5. Open the result in IDent Dynamics

The output folder mirrors your input folder's layout:

```
~/survey-scan/
  station-c/
    2025-09-08T1400.wav                copy of your recording
    2025-09-08T1400.structures.json    what the scanner found
    2025-09-08T1400.png                the lane thumbnail
  siar-scanner-run.json                what was run, and what came back
```

In the web app, use **Open folder** and pick `~/survey-scan`. Every recording appears as a lane
with its spectrogram preview; click one and its boxes are drawn over the surface, counted by
shape in the **Structures** panel and enumerated one row per box in the **Structure list**.
Hovering a row lights its box and vice versa, and clicking a box promotes it to a label.

Save it as a **work project** and the whole thing — folder, boxes and any labels you have made —
comes back in one click next time.

## Command reference

| Command | What it does |
|---|---|
| `siar-scanner version` | the package version and this machine's build tag |
| `siar-scanner login` | sign in and cache a bearer token |
| `siar-scanner logout` | forget the cached token on this machine |
| `siar-scanner whoami` | who the cached token belongs to |
| `siar-scanner algorithms` | the algorithms your account can download |
| `siar-scanner scan` | summarise a folder from headers alone |
| `siar-scanner run` | scan a folder and build the output folder |
| `siar-scanner runs` | what has been run from this machine |

`--help` on any of them prints the full flag list.

## What is in the output

`<recording>.structures.json` is one document per recording:

```json
{
  "format": "siar-scanner-structures-v1",
  "filename": "2025-09-08T1400.wav",
  "duration_sec": 300.0,
  "algorithm": "all_structures",
  "algorithm_version": "1",
  "params": {},
  "stft": { "fft": 1024, "hop": 256, "window": "hann", "sample_rate": 96000 },
  "count": 37,
  "structures": [
    {
      "tmin": 4.0107, "tmax": 4.6293,
      "fmin": 2109.4, "fmax": 3234.4,
      "peakHz": 2812.5, "cells": 214,
      "peakSigma": 6.42, "confidence": 0.89,
      "shape": "sweep"
    }
  ]
}
```

Nine fields are the contract and can be relied on: `tmin`, `tmax`, `fmin`, `fmax`, `peakHz`,
`cells`, `peakSigma`, `confidence`, `shape`. An algorithm may add its own diagnostics alongside
them, and those are written down as it gave them.

`siar-scanner-run.json` records the algorithm, the grid, the parameters, and one row per
recording with its status, its structure count and its per-shape breakdown — including the rows
that failed and why.

## Notes

**Sample rate.** Recordings are read at their native rate. The web app's browser decode
resamples to the tab's audio-context rate, typically 48 kHz, which throws away everything above
24 kHz before a scanner sees it. Running here is how a 130 kHz porpoise click survives to be
found.

**Memory.** One recording is resident at a time, so a 10,000-file corpus costs what its largest
single recording costs. A long recording at a fine grid is the case to watch: the CLI prints the
grid size before starting and suggests raising `--hop` when it is large.

**The algorithms are closed source.** They download as obfuscated bundles and this package never
sees inside them. That is a deterrent paired with your licence terms, not a cryptographic
guarantee. Your audio never leaves your machine either way — the only thing this tool sends is
your login.

---

© Vixen Intelligence, 2026. Proprietary.
