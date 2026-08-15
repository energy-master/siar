# siar-app
## One library for edge and high performance computing

**SIaR — Signal Information and Reconnaissance.** 

Run our models locally against your dataset securely and offline. SIaR allows you to point our model(s) against your dataset and build resulting datasets in your own secure and offline environment. Download the models you wish to use and run them. All output can be viewed by dropping your output folder into IDent dynamics at [goident.ai] or view locally in our slimmed down version of the viewer.

SIar is optimised for both high performance computing (HPC) setup and edge processing. Simply add --parallel to your run an ditribute your compute across your hardware. 

The data pipeline:
```
audio folder -> STFT -> scanning algorithm -> boxes + thumbnails -> a folder the app opens
```

Detections / Structures are **two-dimensional**: 1. a start and end time *and*  2. a low and high frequency. So for example, a
result is "a down-sweep from 3.2 to 2.1 kHz between 4.0 and 4.6 seconds", not just "something
happened around 4 seconds".

## Install

```bash
uv tool install --python 3.13 git+https://github.com/energy-master/siar.git
```

Runs on your CPU. Pulls in numpy and soundfile, and nothing else.

**Use `uv`, and let it choose the Python.** The algorithms are downloaded as prebuilt
bundles tied to CPython **3.13 exactly**. `uv tool install` fetches a
private 3.13 for this tool and keeps it isolated, so the CLI works whatever Python your system
happens to have and whatever you upgrade to later. Installing with `pip` under the wrong
interpreter produces a CLI that runs, signs in, and then finds no algorithm it can load.

If you do not have `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
```

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

To upgrade later: `uv tool upgrade siar-app`.

<details>
<summary>Installing with pip instead</summary>

Works, but the 3.13 is then your responsibility:

```bash
python3.13 -m pip install git+https://github.com/energy-master/siar.git
```

`pip` will refuse any other version — `requires-python` is pinned to `>=3.13,<3.14` precisely so
that this fails at install time rather than at the first scan.

</details>

The first command you run shows the MIT licence and asks you to accept it. See
[Licence](#licence). Run the commands below to get started.

```bash
siar-app quick-start      # the illustrated walkthrough, sixteen steps
siar-app readme           # this manual
```

## Quick start

Three commands. Sign in, see what you can run, run it.

```bash
siar-app login                                    # your IDent Dynamics account
siar-app algorithms                               # what your account can download
siar-app run ~/survey-audio -a all_structures -o ~/survey-scan
```

No IDent Dynamics account yet? `siar-app signup` makes one from here.

Then open `~/survey-scan` in IDent Dynamics — **Open folder** in the app — and work through
what came back.

Or type `siar-app` on its own and do the same thing on one screen.

## The library — `siar-app` with no command

`siar-app` with no command opens the **library** (also `siar-app lib`): everything this machine
can run, and the form that runs it.

```
siar-app  library   5 downloaded · 1 built here · 6 bots · 6 features
────────────────────────────────────────────────────────────────────────────────────────────────
   NAME                      WHERE FROM  VERSION  RUNS ON                  SIZE  WHEN         BOTS
   all_structures            downloaded  1.0.0    linux-x86_64-cp313    1.0 MiB  2026-08-10      —
 ▸ recall                    built here  0.1.0    this machine        140.6 KiB  2026-08-15      6
────────────────────────────────────────────────────────────────────────────────────────────────
   #  KIND       FITNESS  THRESHOLD  NODES  DEPTH  READS
 ▸ 0  champion    0.9899    -0.3674     15      8  ratio_5350_5700hz_over_total, contrast_5700…
   1  runner-up   0.9897          —     13      7  band_6750_7100hz, ratio_5350_5700hz_over_to…
────────────────────────────────────────────────────────────────────────────────────────────────
MODEL — recall
  target recall   built 2026-08-15 21:05 by siar-build 0.1.0
  corpus /home/you/survey-audio
  audio  96000 Hz · fft 8192 · hop 2048 · band 5000-7800 Hz · 128 bins
  result held-out 0.779 window / 0.859 recording · null 0.515 · parity ok
  reads  ratio_5350_5700hz_over_total×6, contrast_5700_6050hz, peak_band_share×6
────────────────────────────────────────────────────────────────────────────────────────────────
RUN
   scan      ~/survey-audio   412 recordings
   write to  ~/survey-scan
   parallel  auto — one worker per core, as many as this machine's memory will hold
 ▸ ▶ start   run recall now
────────────────────────────────────────────────────────────────────────────────────────────────
 ↑↓ select   tab pane   enter choose/run   i scan   o write to   p parallel   R reload   q quit
```

Two kinds of model, one list. The **downloaded** ones are the bundles `siar-app run` fetches from
IDent Dynamics. The ones **built here** are yours, from
[`siar-build`](https://github.com/energy-master/siar-build) — read straight out of its own index
at `~/.siar-build/models.db`, so a model you evolved last week is in the list the moment it is
packaged, with the **bots** it came out of (the champion and the runners-up of that search), what
each of them **reads**, and the run they came from. Nothing is copied and nothing is written back;
delete that database and you lose a listing, not a model.

A downloaded bundle has no bots to show, and says so. What is inside it is licensed separately and
stays inside it — that split is the whole design, and the library does not pretend otherwise.

**Paths are pointed at, not typed.** `i` and `o` open a browser over the folder that row is
already pointing at: folders and recordings, <kbd>enter</kbd> to open one, <kbd>←</kbd> to go back
up, letters to jump, and a row at the top that chooses the folder you are standing in. The output
browser adds **new folder here…**, which takes a name — or a whole path, if you would rather type.
The input may be **one recording** as readily as a folder, which is how you try a model on a single
file before committing a survey drive to it.

`parallel` is off or auto: one recording at a time, or every core this machine's memory will hold.
Anything more precise is a flag on `siar-app run`.

Then <kbd>enter</kbd> on **▶ start**, and the run takes the screen — the same live panel as
[`siar-app run --tui`](#watching-a-long-run---tui), with the bar, the stage breakdown, a row per
worker and what is being found. <kbd>Ctrl-Q</kbd> closes it and hands the library back, with the
result on the form and the run in the history at the bottom of the screen.

| Key | What it does |
|---|---|
| <kbd>↑</kbd> <kbd>↓</kbd> | move in the pane that has the focus |
| <kbd>tab</kbd> | models → bots → run |
| <kbd>enter</kbd> | open the browser on a path row, toggle `parallel`, or start the run |
| `i` / `o` | choose what to scan / where to write |
| `p` | parallel off or auto |
| `R` | re-read the algorithm cache, siar-build's index and the run history |
| `q` | quit |

It needs a terminal. In a pipe, a log or a CI job `siar-app` prints its help as it always did.

---

# Tutorial

This walks the whole thing end to end on a folder you already have. Nothing here is synthetic:
if you have a directory of survey recordings, use it.

## 1. Sign in

The algorithms live on Vixen Intelligence servers, not in this package. Sign in in order to create a valid token for communicating with Vixen server's.

No account yet? `siar-app signup` creates one without opening a browser — it is the same
self-service signup as the web form, and lands you in the same place:

```bash
$ siar-app signup
Email: you@example.com
Username (3-64 chars, letters digits . _ -): you
Display name (optional): You
Password (8+ characters):
Confirm password:
Account created on https://goident.ai as you.
We've emailed a verification link to you@example.com.
Click it, then run `siar-app login`.
```

This does not sign you in as a new account has to confirm its email address. Click the link, then:

```bash
$ siar-app login
IDent Dynamics username or email: rahul@vixenintelligence.com
Password:
Signed in to https://goident.ai as rahul.
Token saved to /home/you/.siar-app/credentials.json
```

The token is cached at mode 0600 and is what every later command uses. In a script or a
container, skip the prompt entirely:

```bash
export SIAR_APP_URL=https://goident.ai
export SIAR_APP_TOKEN=...      # from a previous login, or your account page
```

`siar-app whoami` says who you are signed in as; `siar-app logout` forgets the token
locally. Revoking it properly is done from your account page in the app.

## 2. See what you can run

```bash
$ siar-app algorithms
NAME                      FINDS                  RUNS HERE  WHAT IT IS
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
$ siar-app algorithms --params
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
architecture and a Python minor version. `siar-app version` prints the tag your machine
reports; ask whoever publishes the algorithms for a build with that tag.

## 3. Look at the folder before you scan it

```bash
$ siar-app scan ~/survey-audio
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
$ siar-app run ~/survey-audio --algorithm all_structures --out ~/survey-scan
algorithm  all_structures (linux-x86_64-cp313)
scanning   /home/you/survey-audio
output     /home/you/survey-scan
[412/412] station-c/2025-09-08T1400.wav: 37 structures

METRIC            VALUE  SHARE
-------------  --------  -----
recordings          412
scanned             411
errors                1
audio scanned   27.40 h
wall time      30.7 min
  decode        2.1 min     7%
  fft           8.0 min    26%
  scan         17.6 min    57%
  write            24 s     1%
  thumbnail     1.9 min     6%
  overhead         41 s     2%
realtime          53.6x
workers               1
structures        9,043

STRUCTURE    FOUND
-----------  -----
click        4,188
click_train    122
patch          613
sweep          219
tonal        3,901

Output folder: /home/you/survey-scan
Open it in IDent Dynamics (Open folder) to see the boxes on the spectrogram.
```

While a recording is being worked on, that line is live and says what is happening to it:

```
[287/412] station-c/1400.wav  1.9 GiB · 30.0 min  [scan]  ████████░░░ 61%  38s of ~62s  ·  41s on the file
```

The size and the length come from the file itself, and the tag in the middle is the stage it is
in right now — one of the same five the table below charges the time to.

**The bar is the stage's, and it starts again at each one.** That matters more than it sounds:
the five stages are nothing like equal, and `scan` is typically 90% of a recording. A bar drawn
against the whole file is therefore full before the scan has even begun, and then sits at 99% for
the hour that the work actually takes. Per stage, `[decode]` and `[fft]` each get their own short
bar and `[scan]` gets a long one that means something.

The estimate behind it comes from **this recording's own earlier stages**. The transform and the
scan walk the same magnitude grid, so once `[fft]` has taken eight seconds on this file, the scan
is about whatever multiple of eight the algorithm's split says — and that holds whatever the
sample rate, whatever the machine, and whether or not the algorithm has ever run here before. The
stages that wait on a disk are left out of it: how fast a file was read says more about the page
cache than about the work ahead.

Only the first stage of the first recording has nothing to go on, and there the last run of the
same algorithm on this machine fills in (`~/.siar-app/runs.json`, recorded automatically). A
first run of a new algorithm reports elapsed time alone for a few seconds, then has bars like any
other. A bar reaches 90% when a stage has taken exactly as long as expected and creeps towards
99% after that without ever filling: an estimate that has been overtaken should say so by slowing
down, not by stopping — a frozen bar and a hung run look identical.

`realtime` is the row to plan with: 53.6x means the scan ran 53.6 times faster than the audio it
scanned, so an hour of recording cost about a minute. It counts only audio this run actually
scanned, so a `--resume` run is rated on the work it did rather than on the files it skipped, and
on `--parallel` a second row gives the same figure per worker.

The rows indented under `wall time` say where that time went, and each has a different cure:

| Stage | What it is | If it dominates |
|---|---|---|
| `decode` | Reading and mixing the audio | The disk is the limit — a faster one, or fewer channels |
| `fft` | Building the Fourier grid | Raise `--hop`, or lower `--fft` |
| `scan` | The algorithm itself | Where the time should go. `--parallel` is the only lever left |
| `write` | Copying the audio and writing the sidecar | Try `--link` |
| `thumbnail` | The 200x64 lane preview | `--no-thumbnails` gives it back |
| `overhead` | Header probing, manifest rewrites, pool startup | Rarely worth chasing |

On a `--parallel` run those stages were spent by several workers at once, so they are totalled
under `worker time` — their own heading — and add up to more than the wall clock, which is the
point of running them at once. The same numbers, per file, are in `siar-app-performance.json` at
the root of the output folder.

The first run downloads the algorithm and caches it under `~/.siar-app/algorithms/`. Every
run after that is completely offline — useful on a vessel, required in a lab with no network.

Useful flags:

| Flag | Why |
|---|---|
| `--parallel` | Scan several recordings at once, one process per core. The single biggest lever on a large corpus. |
| `--tui` | Draw the whole run in one live panel: progress, where the time is going, what is being found, a row per worker. Holds the finished run on screen until Ctrl-Q. |
| `--resume` | Carry on where an interrupted run stopped. Safe to pass always. |
| `--limit 20` | Trial the algorithm on twenty files before committing to a corpus. |
| `--max-size 2GB` | Raise (or with `0`, remove) the ceiling on how big a recording may be. Default 550MB — see [Memory](#notes). |
| `--link` | Hardlink the audio into the output folder instead of copying it. Same filesystem only; falls back to a copy. |
| `--param minSigma=3.5` | Tune the algorithm. Repeatable. |
| `--fmin 2000 --fmax 20000` | Restrict the band. |
| `--fft 2048 --hop 512` | Override the analysis grid. Only if you know why — each algorithm carries the grid it was tuned at. |
| `--channel left` | Which channel, on a multi-channel recording. Default is a mix. |
| `--no-thumbnails` | Skip the lane previews. Saves a few milliseconds per file and makes the folder less useful. |

Interrupt it with Ctrl-C at any point; nothing is half-written, and `--resume` picks it up.

### Scanning on every core

Recordings are independent of each other, so a corpus is the easiest kind of thing to scan in
parallel. `--parallel` runs one worker process per recording in flight:

```bash
siar-app run ~/three-week-stream --algorithm all_structures --out ~/stream-scan --parallel
```

Bare `--parallel` uses as many workers as the machine will hold; `--parallel 8` fixes the number.
Twelve hours of work on one core is an hour on twelve, and the output folder is identical either
way — the same sidecars, byte for byte, because a worker is handed the same grid and the same
parameters wherever it runs.

The display becomes one row per worker, so a stalled lane is visible rather than buried:

```
[████████████░░░░░░░░░░░░]  48%  12043/25318 files  ~201.4 h of 418.2 h audio  ~38.1x realtime
12 workers  ·  5:24:11 elapsed  ·  5:41:03 left  ·  91043 structures
  1  ████████░░░░░░  61%    38s   40.2x  [scan]    station-a/0410.wav  3.3 GiB · 40.0 min
  2  ██░░░░░░░░░░░░  17%    11s   35.9x  [decode]  station-a/0420.wav  238.4 MiB · 10.0 min
  3  ··············    —      idle
```

Each row says how far through its **current stage** that worker is, how long it has been on the
recording, **its own realtime factor**, which stage it is in, and how big the file is in bytes
and in audio. The per-worker factor is that lane's own measurement, not the run's divided by the
pool — which is the point of having a row per worker at all: a lane running at half the speed of
its neighbours is a number that says so.

The top line is the corpus, and its bar counts **scan progress**: a recording still being decoded
has not been scanned in any sense worth drawing, one half-way through its scan is half done, and
one that has reached `write` is finished as far as the algorithm is concerned. Counting whole
files only is how a bar reads 2% while every row under it reads 40%. The `~` marks the figures
that include work still in flight — they are estimates, and they converge on the closing table's
exact numbers as the recordings land.

Two things to know before turning it up:

* **Memory, not cores, is usually the limit.** Each worker holds one recording's magnitude grid,
  so the pool costs the largest recording in the corpus times the number of workers. The count
  chosen by bare `--parallel` is capped to fit; an explicit `--parallel N` is obeyed with a
  warning if it will not. Raising `--hop` shrinks every grid and buys workers, and `--max-size`
  keeps one outsized recording from sizing the whole pool: files above the ceiling are gone
  before the pool is measured.
* **The manifest lists recordings in the order they finished**, which on a parallel run is not
  folder order — a ten-second clip submitted after a forty-minute drift recording is written
  first. Nothing reads the manifest in order; the sidecars are what the app pairs to lanes.

### Watching a long run: `--tui`

`--tui` replaces the per-file lines with one panel displaying key metrics in real-time:

```bash
siar-app run ~/three-week-stream -a all_structures --out ~/stream-scan --parallel --tui
```

```
╭─ all_structures · 12 workers ─────────────────────────────────── 5:24:11 elapsed ─╮
│ ██████████████░░░░░░░░░░░░░░░  48%  12043/25318 files  ~201.4 h of 418.2 h  ~38.1x│
│ 5:41:03 left  ·  91,043 structures  ·  3 errors                                   │
├─ time by stage ─────────────────────────┬─ structures found ──────────────────────┤
│ scan         48.90 h  ██████████  91%   │ click        41,882  ██████████         │
│ fft           2.71 h  █·········   5%   │ tonal        39,014  █████████·         │
│ thumbnail     1.09 h  ··········   2%   │ patch         6,131  █·········         │
│ decode        0.85 h  ··········   2%   │ sweep         2,194  ··········         │
│ write         0.16 h  ··········   0%   │ click_train   1,822  ··········         │
├─ workers ─────────────────────────────────────────────────────────────────────────┤
│   1  ████████░░░░  61%    38s   40.2x  [scan]    …0410.wav   3.3 GiB · 40.0 min   │
│   2  ██░░░░░░░░░░  17%    11s   35.9x  [decode]  …0420.wav  238.4 MiB · 10.0 min  │
│   3  ············   idle                                                          │
├─ problems ────────────────────────────────────────────────────────────────────────┤
│ ! station-b/2026-06-28/1130.wav: could not decode: Format not recognised.         │
├─ just finished ───────────────────────────────────────────────────────────────────┤
│ ✓ station-a/2026-07-03/0400.wav                             37 structures         │
│ ✓ station-c/2026-07-02/2350.wav                             12 structures         │
│ Ctrl-C leaves a usable folder — --resume picks it up where it stopped.            │
╰───────────────────────────────────────────────────────────────────────────────────╯
```

The stage duration data allows you to fine tune and understand where compute is spent.

## 5. Open the result in IDent Dynamics

The output folder mirrors your input folder's layout:

```
~/survey-scan/
  station-c/
    2025-09-08T1400.wav                copy of your recording
    2025-09-08T1400.structures.json    what the algorithm found
    2025-09-08T1400.png                the lane thumbnail
  siar-app-run.json                what was run, and what came back
  siar-app-performance.json        what it cost — see below
```

Every structure declaration declares the **family** of the model that wrote it, so IDent dynamics automatically structures itself around it.

In the web app, use **Open folder** and pick `~/survey-scan`. Every recording appears as a lane
with its spectrogram preview; click one and its boxes are drawn over the surface, counted by
shape in the **Structures** panel and enumerated one row per box in the **Structure list**.
Hovering a row lights its box and vice versa, and clicking a box promotes it to a label.

Save it as a **work project** for quick retrieval in the future.

## 6. Look at a scan that is on another machine: `siar-app serve` (HPC)

A survey is often scanned where the cores are, namely a remote server. Rather than copying the results folder over, simply serve the folder to your local machine and view the results on your local browser. 

```bash
# on the machine that did the scanning
$ siar-app serve ~/survey-scan

FIELD       VALUE
----------  ------------------------------------------------------
folder      /home/you/survey-scan
recordings  12,481  (12,469 scanned, 12 errors)
audio       27.40 h
structures  3,117,204
state       complete
url         http://127.0.0.1:8420/?t=jLhljXDeneOuLwuEqn5w__x7…

On your laptop, run:
    ssh -N -L 8420:localhost:8420 you@survey-box
then open:
    http://localhost:8420/?t=jLhljXDeneOuLwuEqn5w__x7…

Read-only: nothing this serves can change the folder. Ctrl-C to stop.
```

Copy the `ssh` line, run it on your laptop, open the URL. You get a page showing every recording as
a lane with its thumbnail, its outcome and its structure count; click one and its boxes are drawn
over a spectrogram, with the run's performance table and the other runs on that box a click away.

**It sends a picture, not the audio.** Opening a lane fetches a reduced spectrogram computed on the
remote machine.  The recordings themselves are downloaded only if you press play or download,
and `--no-audio` refuses even that.


**Serve a run that is still going.** The daemon reads the run manifest, which `siar-app run`
rewrites after every recording, so a scan started this morning can be watched from your laptop this
afternoon. 

| Flag | Meaning |
|---|---|
| `--port N` | port to listen on (default 8420; `0` picks a free one) |
| `--bind ADDR` | address to listen on (default `127.0.0.1`, the `ssh -L` end) |
| `--allow-remote` | permit a non-loopback `--bind`, over plain HTTP |
| `--token VALUE` | use this token instead of a fresh one. Persists in shell history — prefer the minted one |
| `--open` | also open the page in a browser on the serving machine |
| `--no-audio` | refuse to serve the recordings themselves; pictures still work |
| `--allow-origin URL` | let a web origin read the daemon cross-origin. Repeatable, none by default |
| `--verbose` | one line per request, with the token stripped |

With no folder argument it serves the most recent run from `siar-app runs`.


## Command reference

Fifteen commands. Seven of them (`lib`, `version`, `quick-start`, `readme`, `installed`, `scan`,
and `run --algorithm-path`) work without an account; the rest need a login.

| Command | What it does | Needs a login |
|---|---|---|
| [`lib`](#siar-app-lib) | browse what this machine can run, and scan from it — also what bare `siar-app` opens | no |
| [`version`](#siar-app-version) | the package version and this machine's build tag | no |
| [`license`](#siar-app-license) | show the licence, or accept it without a prompt | no |
| [`quick-start`](#siar-app-quick-start) | open the illustrated quickstart in a browser | no |
| [`readme`](#siar-app-readme) | open this manual in a browser | no |
| [`signup`](#siar-app-signup) | create an IDent Dynamics account | no — it makes one |
| [`login`](#siar-app-login-username) | sign in and cache a bearer token | — |
| [`logout`](#siar-app-logout) | forget the cached token on this machine | no |
| [`whoami`](#siar-app-whoami) | who the cached token belongs to | reads the cache |
| [`algorithms`](#siar-app-algorithms) | the algorithms your account can download | yes |
| [`installed`](#siar-app-installed) | the algorithms on **this machine**, and their versions | no |
| [`scan`](#siar-app-scan-folder) | summarise a folder from headers alone | no |
| [`run`](#siar-app-run-folder---out-dir) | scan a folder and build the output folder | first run only |
| [`runs`](#siar-app-runs) | what has been run from this machine | no |
| [`feedback`](#siar-app-feedback-name) | rate how well an algorithm performed, 0-9 | yes |

`--server URL` is accepted by every command that talks to the server, on **either** side of
the subcommand — `siar-app --server … run` and `siar-app run --server …` both work. You
normally never pass it: the install you logged in to is remembered.

`--help` on any command prints its flags; `siar-app --version` prints the version and exits.
With no command at all it opens [the library](#the-library--siar-app-with-no-command) — or, off a
terminal, prints the help and exits 0.

Every command prints a two-line banner first:

```
SIaR · Signal Information and Reconnaissance · goident.ai
siar-app 0.6.0 · © Vixen Intelligence 2026
```

It goes to **stderr**, never stdout — `algorithms --json`, `installed --json` and `runs --json`
exist to be piped into something, and a banner on stdout would make every one of them emit
invalid JSON. `$SIAR_APP_NO_BANNER` turns it off for a script that logs stderr.

### `siar-app lib`

The library, described in full [above](#the-library--siar-app-with-no-command): every model on
this machine — downloaded bundles and anything `siar-build` has packaged here — the bots and
features behind each one, and a form that scans a folder or a single recording with it. `library`
is accepted as an alias, and bare `siar-app` opens the same screen.

No flags. Every choice it offers is one it can also show you the options for, which is the point
of it. It needs a terminal, and off one it says so and points at `siar-app installed` and
`siar-app run` instead.

Nothing here is a second implementation of anything: the run is
[`siar-app run --tui`](#watching-a-long-run---tui), the model list is what
[`siar-app installed`](#siar-app-installed) reports, and the models built here are read
**read-only** out of siar-build's own index (`~/.siar-build/models.db`, or
`$SIAR_BUILD_HOME/models.db`). A run started from the library is the same row in the same history
as one started from a shell.

### `siar-app version`

```bash
$ siar-app version
siar-app 0.6.0
platform     linux-x86_64-cp313
licence      MIT
© Vixen Intelligence 2026
```

The platform line is the build tag this machine reports — operating system, CPU architecture and
Python minor version. It is the first thing to check when a download is refused or an algorithm
says it does not run here. No flags, no network, no login.

### `siar-app license`

Prints the terms — on stdout, unlike the first-run prompt, since someone who typed `license`
may want to redirect it — and says whether they have been accepted on this machine.

| Flag | Meaning |
|---|---|
| `--accept` | record acceptance and exit, for a script or a container |

See [Licence](#licence) below.

### `siar-app quick-start`

Opens the illustrated quickstart in whatever browser this machine has — sixteen steps, from
installing `uv` through to rating an algorithm, each one a terminal window showing the real
command and its real output.

```bash
$ siar-app quick-start
Opened the quickstart in your browser.
  /home/you/.local/share/uv/tools/siar-app/lib/python3.13/site-packages/siarapp/local_web/quickstart.html
```

It ships **inside the package** and pulls nothing from the internet, so it works on a vessel
with no signal. No flags, no login, and it runs before the licence has been accepted — a
prompt in front of the manual would be a poor greeting.

On a headless box, where there is no browser to open, it prints that path instead of
pretending it worked. Copy it to any machine that has one.

To change what it says, edit `page-text.js` beside it: the whole deck is plain text with a
one-character marker per line, and the numbering and step list follow whatever is in the file.

### `siar-app readme`

Open README in your browser.

```bash
$ siar-app readme
Opened the manual in your browser.
  /tmp/siar-app-readme-xxxxxxxx/siar-app-readme.html
```

| Flag | Meaning |
|---|---|
| `--text` | print it as Markdown on stdout instead of opening a browser |

Also offline. Nothing extra was packaged to make this work: `pyproject.toml` names this file as
the project's long description, so every wheel already carries the whole of it in its metadata,
and that is what gets rendered. A source checkout reads `README.md` from the repository root
instead — the file wins over the metadata, so editing it and re-running shows the edit rather
than whatever was current when you last installed.

### `siar-app signup`

Creates an account on an IDent Dynamics install — the headless half of the web app's signup
form, sharing its validation, its per-IP hourly limit and the account it produces (a plain
user, 50 MB of storage). Every field is prompted for if not given.

| Flag | Meaning |
|---|---|
| `--email ADDRESS` | where the verification link is sent; prompted for if omitted |
| `--username NAME` | 3-64 characters: letters, digits, and `.` `_` `-` |
| `--display-name NAME` | how your name appears in the app (default: your username). Pass `''` to take the default without a prompt |
| `--server URL` | which install to create the account on (default: `https://goident.ai`) |

The password is read from `$SIAR_APP_PASSWORD` if set, and prompted for twice otherwise.

It does **not** log you in afterwards, and that is not an omission: the account is created
unverified, and `login` would refuse it until the emailed link has been used. So the command
ends by telling you which address to go and check. If the email never arrives, the account
still exists — ask for another link from the sign-in page in the web app.

Signup can be switched off installation-wide (`app.allow_signup = false`), in which case this
command reports that rather than failing obscurely. On an install that is not `goident.ai`,
pass `--server` — it is not remembered, since nothing is cached until you log in.

### `siar-app login [USERNAME]`

Exchanges your IDent Dynamics username (or email) and password for a bearer token, cached at
`~/.siar-app/credentials.json` (mode 0600). The install you sign in to becomes the default
for every later command, so `--server` is a once-only argument.

| Flag | Meaning |
|---|---|
| `USERNAME` | username or email; prompted for if omitted |
| `--device LABEL` | how this machine is labelled in your account's token list |
| `--server URL` | which install to sign in to (default: the last one, else `https://goident.ai`) |

The password is read from `$SIAR_APP_PASSWORD` if set, and prompted for otherwise. A shell
with no interactive input — CI, a pipe, an agent — cannot prompt, so pass the username as an
argument and set that variable, or skip logging in entirely with `$SIAR_APP_TOKEN`.

### `siar-app logout`

Deletes the cached token from this machine and says so. It does **not** revoke the token
server-side — a lost laptop is dealt with from your account page in the web app, not here. No
flags.

### `siar-app whoami`

Prints the user, the server and this machine's platform tag. Exits 1 if there is no cached
token, so it doubles as a "am I signed in" test in a script.

### `siar-app algorithms`

The catalogue your installation's super user has published. Columns are the name (what you pass
to `run -a`), what shapes it finds, whether a build exists for this machine, how users have
[rated](#siar-app-feedback-name) it, and model description.

| Flag | Meaning |
|---|---|
| `--family NAME` | only models in one family (name or title) |
| `--params` | also print each algorithm's tunable parameters, with types and defaults |
| `--json` | the raw catalogue, including the `platforms` build tags and `params_schema` |
| `--server URL` | check a different install |

`RATED` shows the mean score and how many people gave one; `—` means nobody has rated it yet,
which is deliberately not the same as a low score.

Models are printed **one table per family**. Every SIaR model belongs to exactly one family. 

Two footnotes may appear under the table. **"no build for `<tag>`"** means the algorithm exists
but not for your OS/architecture/Python. **A `*` beside
a name** means that model is unpublished and only a super user can see it. Ordinary accounts
are not offered it at all. 

### `siar-app installed`

Which models and what version are installed on this machine. `algorithms` lets you have a peek at which models are available to you.
```bash
$ siar-app installed
NAME                      VERSION  PLATFORM               SIZE  DOWNLOADED        RUNS HERE
------------------------  -------  ------------------  -------  ----------------  ---------
all_structures            1.0.0    linux-x86_64-cp313  1.0 MiB  2026-08-10 12:18  yes

1 bundle(s), 1.0 MiB in /home/you/.siar-app/algorithms
```

| Flag | Meaning |
|---|---|
| `--check` | also ask the server whether a newer version is published |
| `--json` | the raw list, including each bundle's path on disk |
| `--server URL` | which install `--check` asks |


With `--check` a `SERVER` column appears:

```bash
$ siar-app installed --check
NAME            VERSION  PLATFORM               SIZE  DOWNLOADED        RUNS HERE  SERVER
--------------  -------  ------------------  -------  ----------------  ---------  ---------------
all_structures  1.0.0    linux-x86_64-cp313  1.0 MiB  2026-08-10 12:18  yes        1.1.0 available

`siar-app run --refresh -a <name>` replaces a cached bundle with the newest.
```

`up to date`, `X.Y.Z available`, `ahead (X.Y.Z published)` — you are running a build newer than
the catalogue, which happens to whoever publishes them — or `not offered`, meaning the algorithm
has been withdrawn or your account can no longer see it. Nothing will replace what is cached
either way; the cache is yours until you refresh it. A `--check` that cannot reach the server
prints a warning and still gives you the local answer.

### `siar-app scan FOLDER`

Read all the headers and grab relevant data for display before starting a run.

| Flag | Meaning |
|---|---|
| `--no-recursive` | only the top level of the folder |

Worth the second before a long run as if the table shows more than one sample rate, every
frequency band maps to a different bin in each group, so the compute time for the run is significantly increased. Recommended to group datasets by samplerates, too.

### `siar-app run FOLDER --out DIR`

The main run command. Runs one algorithm over every recording under `FOLDER` and writes an output
folder holding the audio, one structures datafile per recording, and a lane thumbnail. `--out` is
required and must not be the folder being accessed for input data.

`FOLDER` may also be **a single recording** — `siar-app run ~/survey-audio/station-c/0410.wav
-a all_structures -o /tmp/try` — which is how you try an algorithm on one file before committing a
survey drive to it. The output folder is laid out exactly as it would have been had you scanned
the folder that file sits in, so `--resume` over the whole folder later picks up where it left
off.

**Choosing the algorithm**

| Flag | Meaning |
|---|---|
| `--algorithm`, `-a NAME` | which algorithm (see `siar-app algorithms`) |
| `--algorithm-path DIR` | run an unobfuscated algorithm package straight off disk — development only, needs no login |
| `--platform TAG` | download the build for another platform tag instead of this machine's |
| `--refresh` | re-download even if the bundle is already cached |
| `--server URL` | which install to download from |

**Analysis grid** — defaults come from the algorithm which carries the grid it was tuned at.
Override only if you know why.

| Flag | Meaning |
|---|---|
| `--fft N` | FFT size, a power of two |
| `--hop N` | hop in samples (default: `fft/4`) |
| `--window` | `hann` (default), `hamming`, `blackman`, `rectangular` |
| `--channel SEL` | `mix` (default), `left`, `right`, or a channel index |

**Algorithm parameters**

| Flag | Meaning |
|---|---|
| `--param NAME=VALUE` | set one parameter; repeatable |
| `--fmin HZ` / `--fmax HZ` | shorthand for the band to scan |

`--param` values are typed by what they look like: `true`/`yes`/`on` and `false`/`no`/`off`
become booleans, `null`/`none`/empty becomes null, digits become an int or a float, and anything
else stays a string. An algorithm that wanted a float should not receive `"2.5"`.

**Output**

| Flag | Meaning |
|---|---|
| `--resume` | skip recordings already written to the output folder |
| `--link` | hardlink the audio instead of copying it (same filesystem only) |
| `--no-thumbnails` | skip the per-recording lane previews |
| `--limit N` | stop after N recordings — a trial run over a big corpus |
| `--max-size SIZE` | skip recordings larger than this (default `550MB`; `0` for no ceiling). Takes `KB`, `MB`, `GB` or a plain byte count |
| `--parallel [N]` | scan N recordings at once, one process each; bare `--parallel` uses every core the machine's memory will hold |
| `--no-recursive` | only the top level of the folder |
| `--tui` | draw the whole run in one live panel, held at the end until Ctrl-Q — [see above](#watching-a-long-run---tui). Needs a terminal |
| `--quiet`, `-q` | no per-file progress |

The first run of an algorithm downloads it and displays a progress bar:

```
downloading all_structures  [██████████████████░░░░░░]  73%  256.0 KiB / 349.4 KiB
```

`--quiet` suppresses terminal output.

Exits 1 if any recording errored, so a scripted survey can tell a partial run from a clean one.
Ctrl-C exits 130 and leaves a usable folder — `--resume` picks it up, and a fully-resumed run
says "nothing to do" rather than reporting zero structures like a failure.

### `siar-app runs`

Every run from this machine, newest first. Shows time of run, algorithm, number of files, number of structures flagged, and the output folder. Read from `~/.siar-app/runs.json`; local history, not an
account-wide one.

```bash
$ siar-app runs --limit 3
WHEN                  ALGORITHM       FILES  FOUND  OUTPUT
--------------------  --------------  -----  -----  --------------------
2026-08-10T11:26:00Z  all_structures      3     11  /home/you/survey-scan
```

| Flag | Meaning |
|---|---|
| `--limit N` | how many to show (default 20) |
| `--json` | the raw history |

### `siar-app serve [DIR]`

Serve one output folder read-only over HTTP so it can be browsed from another machine through an
ssh tunnel — see [the tutorial section](#6-look-at-a-scan-that-is-on-another-machine-siar-app-serve-hpc)
for what the page shows and why it sends a picture rather than the audio. `DIR` defaults to the most
recent run in `siar-app runs`.

```bash
$ siar-app serve ~/survey-scan --port 8420
```

Binds `127.0.0.1`, mints a token, and prints the `ssh -L` line to copy. Runs in the foreground until
Ctrl-C, which exits 0 — leave it under `tmux` or `nohup` on a box you log out of. There is no route
that writes: `GET`, `HEAD` and `OPTIONS` are the only methods answered.

| Flag | Meaning |
|---|---|
| `--port N` | port to listen on (default 8420; `0` picks a free one) |
| `--bind ADDR` | address to listen on (default `127.0.0.1`) |
| `--allow-remote` | required for a non-loopback `--bind` |
| `--token VALUE` | use this token instead of a fresh one |
| `--open` | also open the page on the serving machine |
| `--no-audio` | refuse to serve the recordings themselves |
| `--allow-origin URL` | allow a web origin to read it cross-origin; repeatable |
| `--verbose` | one line per request |

### `siar-app feedback NAME`

Rate how well an algorithm did on *your* recordingg[ 0 .. 9]. 

```bash
$ siar-app feedback all_structures --score 7 -m "found the sweeps, missed two faint tonals"
Thanks — all_structures 1.1.0 rated 7/9.
That build now averages 6.7/9 from 3 ratings.
```

| Flag | Meaning |
|---|---|
| `--score`, `-s 0-9` | the rating; prompted for, with a scale, if omitted |
| `--comment`, `-m TEXT` | a sentence on what it did well or badly |
| `--mine` | list the ratings you have given instead of adding one |
| `--server URL` | which install to send it to |

The scale, as the prompt describes it:

| | |
|---|---|
| **0-2** | found nothing useful, or buried the recording in false boxes |
| **3-5** | usable, but needed a lot of sorting through |
| **6-7** | found what was there, with some noise |
| **8-9** | found what was there and little else |



## Environment and files

Everything lives under one directory.

| Variable | Effect |
|---|---|
| `SIAR_APP_HOME` | where everything is kept (default `~/.siar-app`) |
| `SIAR_APP_TOKEN` | use this bearer token, ignoring the credentials file |
| `SIAR_APP_URL` | talk to this install, ignoring the credentials file |
| `SIAR_APP_PASSWORD` | read by `login` instead of prompting |
| `SIAR_APP_ACCEPT_LICENSE` | accept the licence without a prompt (containers, CI) |
| `SIAR_APP_NO_BANNER` | suppress the two-line banner on stderr |
| `SIAR_BUILD_HOME` | where the library looks for `siar-build`'s index (default `~/.siar-build`) |

```
~/.siar-app/
  credentials.json              server, username and token
  license.json                  that you accepted the licence, and when
  runs.json                     what `siar-app runs` lists
  algorithms/<name>/<platform>/ unpacked bundles, one tree per build
```

The one file outside that directory is `~/.siar-build/models.db`, and it belongs to `siar-build`.
The library reads it and never writes it: no row is added, and the file is never created by being
looked for.


**Exit codes.** 0 on success, 1 on a handled failure — a bad folder, an expired token, a run
with errors in it and 130 on Ctrl-C. Expected failures print one sentence to stderr, not a
traceback.

## Updating and removing it

### Update

```bash
uv tool upgrade siar-app
```


```bash
uv tool install --force --python 3.13 git+https://github.com/energy-master/siar.git
```


**To update an algorithm rather than the CLI**, see [`installed --check`](#siar-app-installed)
for which builds have a newer version published, then `siar-app run --refresh -a <name>` to
replace a cached bundle on the next run.

Installed with `pip` instead? `python3.13 -m pip install --upgrade git+https://github.com/energy-master/siar.git`.

### Remove

```bash
uv tool uninstall siar-app        # the CLI and the private 3.13 that ran it
rm -rf ~/.siar-app                # token, licence, run history, cached algorithms
```

Or `python3.13 -m pip uninstall siar-app` for a `pip` install. Three things worth knowing:

- **Revoke the token first if the machine is leaving your hands.** `siar-app logout` is local,
  and so is deleting `~/.siar-app` — the token stays valid on the server either way. Kill it
  from your account page in the web app.
- **Deleting the workspace deletes the algorithm cache**, which is the only expensive part to
  rebuild. To keep your login and clear only the bundles, remove
  `~/.siar-app/algorithms` and leave the rest.
- **A pre-rename workspace normally isn't left behind.** The first run after the rename *moves*
  `~/.siar-scanner` to `~/.siar-app` rather than copying it. Worth a look only if that move ever
  failed — the tool says so on stderr when it does.

Nothing is written outside `~/.siar-app` and `uv`'s own tool directory, so those two removals
are the whole of it.

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

Nine fields are the data contract and can be relied on: `tmin`, `tmax`, `fmin`, `fmax`, `peakHz`,
`cells`, `peakSigma`, `confidence`, `shape`. An algorithm may add its own diagnostics alongside
them. IDent dynamics is able to recognise data types and load appropriate data views.

`siar-app-run.json` records the algorithm, the grid, the parameters, and one row per
recording with its status, its structure count and its per-shape breakdown including the rows
that failed and why.

`siar-app-performance.json` records what the run cost: per file and in total, the audio
duration, the compute wall time, and the **realtime factor** — 3.7 means the scan ran 3.7 times faster than the audio it scanned. It also records the
machine because a factor with no machine attached is close to meaningless, e.g. the same scan is 8x
on a workstation and 0.9x on a field laptop. The worker count is recorded beside the machine for
the same reason and matters more: one flag makes the same machine sixteen times faster. Two times
are kept and the gap between them is the answer to "why is this slower than the sum of its
parts" — `wall_sec` includes decoding, thumbnails and writing while `scan_sec` is the algorithm
alone. On a `--parallel` run `scan_sec` is the larger of the two, because it sums time several
workers spent at once, and dividing it by `wall_sec` says how much of the pool was working.

## Notes

**Sample rate.** Recordings are read at their native rate. 

**Memory.** One recording is resident at a time per worker, so a 10,000-file corpus costs what
its largest single recording costs — times `--parallel`. A long recording at a fine grid is the
case to watch: the CLI prints the grid size before starting and suggests raising the hop size,
`--hop` ,when it is ltoo arge.

Because that cost is set by the biggest file in the folder, a run refuses the outliers rather
than dying on them. Anything over `--max-size` — **550MB** by default, roughly an hour at 96 kHz
stereo — is left out of the run before the headers are read, listed in the manifest as
`too_large`, and reported on the terminal as one line:

```
3 recording(s) are over the --max-size ceiling of 550 MB (largest 41293 MB) and will not be
scanned. They are in the manifest as `too_large`.
```

The file size is not the memory it costs: the magnitude grid built from it is several times
larger. Raise the ceiling (`--max-size 4GB`) or remove it (`--max-size 0`) when the machine can
take it, and raise `--hop` alongside it if the grid is what is tight. Nothing is deleted or
altered — a skipped recording is simply not in the output folder.

**The algorithms are closed source.** They download as obfuscated bundles and this package never
sees inside them. That is a deterrent paired with your licence terms, not a 
guarantee. Your audio never leaves your machine. The only thing this tool sends is
your login ( you can go offline after ).

## Licence

This command line is **MIT** — see [`LICENSE`](LICENSE).

The first command that does any work shows the terms and asks you to accept them. **Once.** The
answer is recorded and you are never asked again. A shell that cannot prompt — CI, a container, an agent — is not accepted silently. It is told to run `siar-app license --accept` once, or to set
`$SIAR_APP_ACCEPT_LICENSE`. Declining exits 1 and runs nothing.

`version` and `license` are the only commands that work before acceptance, because gating either
would be a licence you have to accept before you may read it.

**What MIT does and does not cover.** It covers this package: the CLI, the decoder, the STFT,
the output-folder format. It does **not** cover the algorithms as those are proprietary,
are not distributed with this package, and are licensed separately by the IDent Dynamics
installation you sign in to. 

---

SIaR — Signal Information and Reconnaissance · [goident.ai](https://goident.ai), [vixenintelligence.com](https://www.vixenintelligence.com)

© Vixen Intelligence, 2026.
