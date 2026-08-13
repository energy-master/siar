# siar-app

**SIaR — Signal Information and Reconnaissance.** 

Run our models locally against your dataset securely and offline.

SIaR allows you to point our model(s) against your dataset and build resulting datasets in your own secure and offline environment. Download the models you wish to use and run them. All output can be viewed by dropping your output folder into IDent dynamics at [goident.ai] or view locally in our slimmed down version of the viewer.

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
siar-app quick-start      # the illustrated walkthrough, thirteen steps
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

412 file(s), 27.40 h of audio, in 1841.2s
9,043 structures: 122 click_train, 4188 click, 219 sweep, 3901 tonal, 613 patch
1 file(s) error

Output folder: /home/you/survey-scan
Open it in IDent Dynamics (Open folder) to see the boxes on the spectrogram.
```

The first run downloads the algorithm and caches it under `~/.siar-app/algorithms/`. Every
run after that is completely offline — useful on a vessel, required in a lab with no network.

Useful flags:

| Flag | Why |
|---|---|
| `--parallel` | Scan several recordings at once, one process per core. The single biggest lever on a large corpus. |
| `--resume` | Carry on where an interrupted run stopped. Safe to pass always. |
| `--limit 20` | Trial the algorithm on twenty files before committing to a corpus. |
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
[████████████░░░░░░░░░░░░]  48%  12043/25318 files  201.4 h of 418.2 h audio
12 workers  ·  38.1x realtime  ·  5:24:11 elapsed  ·  5:41:03 left  ·  91043 structures
  1  ████████░░░░░░  61%    38s  station-a/2026-07-03/0410.wav
  2  ██░░░░░░░░░░░░  17%    11s  station-a/2026-07-03/0420.wav
  3  ··············    —      idle
```

Two things to know before turning it up:

* **Memory, not cores, is usually the limit.** Each worker holds one recording's magnitude grid,
  so the pool costs the largest recording in the corpus times the number of workers. The count
  chosen by bare `--parallel` is capped to fit; an explicit `--parallel N` is obeyed with a
  warning if it will not. Raising `--hop` shrinks every grid and buys workers.
* **The manifest lists recordings in the order they finished**, which on a parallel run is not
  folder order — a ten-second clip submitted after a forty-minute drift recording is written
  first. Nothing reads the manifest in order; the sidecars are what the app pairs to lanes.

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

Every sidecar declares the **family** of the model that wrote it, so the app can tell what kind
of results it is opening rather than guessing from which fields are present. A structure seeker
and a click detector both emit boxes with the same nine fields; only the family says which is
which.

In the web app, use **Open folder** and pick `~/survey-scan`. Every recording appears as a lane
with its spectrogram preview; click one and its boxes are drawn over the surface, counted by
shape in the **Structures** panel and enumerated one row per box in the **Structure list**.
Hovering a row lights its box and vice versa, and clicking a box promotes it to a label.

Save it as a **work project** and the whole thing — folder, boxes and any labels you have made —
comes back in one click next time.

## Command reference

Fourteen commands. Six of them (`version`, `quick-start`, `readme`, `installed`, `scan`, and
`run --algorithm-path`) work without an account; the rest need a login.

| Command | What it does | Needs a login |
|---|---|---|
| [`version`](#siar-app-version) | the package version and this machine's build tag | no |
| [`license`](#siar-app-license) | show the licence, or accept it without a prompt | no |
| [`quick-start`](#siar-app-quick-start) | open the illustrated quickstart in a browser | no |
| [`readme`](#siar-app-readme) | open this manual in a browser | no |
| [`signup`](#siar-app-signup) | create an IDent Dynamics account | no — it makes one |
| [`login`](#siar-app-login-username) | sign in and cache a bearer token | — |
| [`logout`](#siar-app-logout) | forget the cached token on this machine | no |
| [`whoami`](#siar-app-whoami) | who the cached token belongs to | reads the cache |
| [`algorithms`](#siar-lib-algorithms) | the algorithms your account can download | yes |
| [`installed`](#siar-app-installed) | the algorithms on **this machine**, and their versions | no |
| [`scan`](#siar-app-scan-folder) | summarise a folder from headers alone | no |
| [`run`](#siar-app-run-folder---out-dir) | scan a folder and build the output folder | first run only |
| [`runs`](#siar-app-runs) | what has been run from this machine | no |
| [`feedback`](#siar-app-feedback-name) | rate how well an algorithm performed, 0-9 | yes |

`--server URL` is accepted by every command that talks to the server, on **either** side of
the subcommand — `siar-app --server … run` and `siar-app run --server …` both work. You
normally never pass it: the install you logged in to is remembered.

`--help` on any command prints its flags; `siar-app --version` prints the version and exits.
With no command at all, it prints the help and exits 0.

Every command prints a two-line banner first:

```
SIaR · Signal Information and Reconnaissance · goident.ai
siar-app 0.2.0 · © Vixen Intelligence 2026
```

It goes to **stderr**, never stdout — `algorithms --json`, `installed --json` and `runs --json`
exist to be piped into something, and a banner on stdout would make every one of them emit
invalid JSON. `$SIAR_APP_NO_BANNER` turns it off for a script that logs stderr.

### `siar-app version`

```bash
$ siar-app version
siar-app 0.2.0
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

Opens the illustrated quickstart in whatever browser this machine has — thirteen steps, from
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

This document, rendered and opened in your browser.

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
[rated](#siar-app-feedback-name) it, and the description — which is written in the admin
panel, so it says what the algorithm is actually being used for.

| Flag | Meaning |
|---|---|
| `--family NAME` | only models in one family (name or title) |
| `--params` | also print each algorithm's tunable parameters, with types and defaults |
| `--json` | the raw catalogue, including the `platforms` build tags and `params_schema` |
| `--server URL` | check a different install |

`RATED` shows the mean score and how many people gave one; `—` means nobody has rated it yet,
which is deliberately not the same as a low score.

Models are printed **one table per family**. Every SIaR model belongs to exactly one family. The families and their order are set by your installation's super user, not by this application.

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

Reads headers only — no decode, no algorithm, no login — so a multi-GB corpus is summarised in
about a second. Prints the recording count, the total duration, and a sample-rate breakdown.

| Flag | Meaning |
|---|---|
| `--no-recursive` | only the top level of the folder |

Worth the second before a long run as if the table shows more than one sample rate, every
frequency band maps to a different bin in each group, so the compute time for the run is significantly increased. Recommended to group datasets by samplerates, too.

### `siar-app run FOLDER --out DIR`

The main run command. Runs one algorithm over every recording under `FOLDER` and writes an output
folder holding the audio, one structures datafile per recording, and a lane thumbnail. `--out` is
required and must not be the folder being accessed for input data.

**Choosing the algorithm**

| Flag | Meaning |
|---|---|
| `--algorithm`, `-a NAME` | which algorithm (see `siar-app algorithms`) |
| `--algorithm-path DIR` | run an unobfuscated algorithm package straight off disk — development only, needs no login |
| `--platform TAG` | download the build for another platform tag instead of this machine's |
| `--refresh` | re-download even if the bundle is already cached |
| `--server URL` | which install to download from |

**Analysis grid** — defaults come from the algorithm, which carries the grid it was tuned at.
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
| `--parallel [N]` | scan N recordings at once, one process each; bare `--parallel` uses every core the machine's memory will hold |
| `--no-recursive` | only the top level of the folder |
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

```
~/.siar-app/
  credentials.json              server, username and token
  license.json                  that you accepted the licence, and when
  runs.json                     what `siar-app runs` lists
  algorithms/<name>/<platform>/ unpacked bundles, one tree per build
```


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
