/* The words on the quickstart page. Edit the text below, save, reload the browser.

   It is a .js rather than a .txt because a browser will not read a .txt off local disk when
   the page was opened with File > Open. Two typing rules follow from that: write a backtick
   as \` , and never write a dollar sign immediately followed by { .

   A slide starts with  --- slide ---  and carries three fields: title, rail (its short name
   in the left-hand list) and note (the paragraph under the terminal, where **stars** make
   bold and ~tildes~ make code). A line reading  term:  begins the transcript.

   In a transcript the first two characters set the colour:
     $   a command typed at the prompt      !   a figure worth noticing
     >   a question, as  Label: :: answer   +   it worked
     .   quiet text: notes, table headings  -   a table rule
   Anything else is ordinary output, and its leading spaces are kept, which is what holds the
   table columns in line. A blank line stays blank.

   Reorder slides by moving their blocks, delete one by deleting its block; the numbering,
   the step count and the left-hand list all follow. */

window.QUICKSTART_TEXT = `

eyebrow: SIaR · Signal Information and Reconnaissance
headline: Build datasets from your data. 
intro: Sixteen steps with ~siar-app~ — the command line that runs the IDent Dynamics structure scanners over a folder of recordings and writes an output folder you drag straight into the web app.
prompt: user@vi:~$
window: user@vi: ~
railtitle: Steps
footer:  Taken from the siar-app README available at siar-app readme. Paths assume a Linux box; on macOS the workspace is the same ~/.siar-app.

--- slide ---
title: Get uv 
rail: Get uv
note: **Use uv and select the correct Python version.** The scanners are prebuilt bundles tied to CPython **3.13 exactly** — not 3.12, not 3.14. This installs uv alone and allows for the correct version of Python to be used.
term:
$ curl -LsSf https://astral.sh/uv/install.sh | sh
. installing to /home/user/.local/bin
.   uv
.   uvx
+ everything's installed!

. # Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

--- slide ---
title: Install siar-app
rail: Install siar-app
note: One command: ~uv tool install~ fetches a private 3.13 for this tool alone so it works whatever Python your system has and whatever you upgrade to later.
term:
$ uv tool install --python 3.13 git+https://github.com/energy-master/siar.git
. Resolved 4 packages in 1.21s
. Installed 4 packages in 34ms
.  + numpy==2.1.3
.  + soundfile==0.12.1
+ Installed 1 executable: siar-app

. # numpy and soundfile, and nothing else. Runs on CPU.

--- slide ---
title: Check the build platform tag
rail: Check the build platform tag
note: The **platform** line is the tag this machine reports — OS, CPU architecture, Python minor version. It is the first thing to check if a download is ever refused. The licence is shown once and accepted once.
term:
$ siar-app version
siar-app 0.5.0
! platform     linux-x86_64-cp313
python       3.13.1
licence      MIT (not yet accepted)
. © Vixen Intelligence 2026

. # The first command that does real work asks you to accept the licence.

--- slide ---
title: Create an account (if you don't already have one)
rail: Create an account 
note: No account yet? ~signup~ is the same self-service signup as the Ident dyanmics @goident.ai. This step does not sign you in as a new account has to confirm its email address first.
term:
$ siar-app signup
> Email: :: you@example.com
> Username (3-64 chars, letters digits . _ -): :: you
> Display name (optional): :: Survey Team
Password (8+ characters):
Confirm password:
+ Account created on https://goident.ai as you.
We've emailed a verification link to you@example.com.
Click it, then run \`siar-app login\`.

--- slide ---
title: Sign in
rail: Sign in
note: The models live on Vixen Intelligence servers, not this package, and therefore requrie a Vixen token.
term:
$ siar-app login
> IDent Dynamics username or email: :: you@example.com
Password:
+ Signed in to https://goident.ai as you.
. Token saved to /home/user/.siar-app/credentials.json
. (delete it, or run \`siar-app logout\`, to sign out).

. # siar-app whoami says who you are; siar-app logout forgets the token.

--- slide ---
title: See which models are available to you
rail:  Availablemodels
note: Users have access to different models. Use this function to see which models are available to you.
term:
$ siar-app algorithms
. NAME                      FINDS                  RUNS HERE  WHAT IT IS
- ------------------------  ---------------------  ---------  --------------------------
all_structures            sweep, tonal, click,   yes        The survey. Boxes every
                          patch, blob                       significant structure.
all_structures_sensitive  sweep, tonal, click,   yes        Lower bar, for faint
                          patch, blob                       structure in a quiet file.
fuzzy_hp_alpha            click, click_train     yes        Harbour-porpoise click
                                                            candidates, 90-150 kHz.

. # --params prints what each one lets you tune.

--- slide ---
title: Which models are installed on this machine
rail:  My models
note: Which models are installed on this machine. Works offline. Add ~--check~ to ask whether a newer build is published.
term:
$ siar-app installed
. NAME            VERSION  PLATFORM               SIZE  DOWNLOADED        RUNS HERE
- --------------  -------  ------------------  -------  ----------------  ---------
all_structures  1.0.0    linux-x86_64-cp313  1.0 MiB  2026-08-10 12:18  yes

. 1 bundle(s), 1.0 MiB in /home/user/.siar-app/algorithms

. # The cache is yours until you refresh it: run --refresh replaces a bundle.

--- slide ---
title: Peek into your input dataset before you run any models
rail: Peek into your input dataset before you run any models
note: Headers only, so this takes about a second on a corpus that will take hours to run. Worth the second! If the table shows **more than one sample rate**, every frequency band maps to a different bin per group which adds considerable time. Consider splitting your input data by samplrates.
term:
$ siar-app scan ~/survey-audio
412 recording(s) under /home/user/survey-audio
! 27.40 h of audio

. SAMPLE RATE  FILES
- -----------  -----
96000 Hz       412

. # No decode, no algorithm, no login needed.

--- slide ---
title: Run the model
rail: Run the model
note: The first run downloads the algorithm and caches it. Every run after that is **completely offline**.Ctrl-C at any point leaves a usable folder, and ~--resume~ picks it up.
term:
$ siar-app run ~/survey-audio --algorithm all_structures --out ~/survey-scan
. algorithm  all_structures (linux-x86_64-cp313)
. scanning   /home/user/survey-audio
. output     /home/user/survey-scan
[412/412] station-c/2025-09-08T1400.wav: 37 structures

. METRIC            VALUE  SHARE
- -------------  --------  -----
recordings          412
. scanned             411
. errors                1
audio scanned   27.40 h
wall time      30.7 min
.   decode        2.1 min     7%
.   fft           8.0 min    26%
!   scan         17.6 min    57%
.   write            24 s     1%
.   thumbnail     1.9 min     6%
.   overhead         41 s     2%
! realtime          53.6x
. workers               1
! structures        9,043

. # The rows under wall time say where it went: fft wants a bigger --hop, scan wants --parallel.

--- slide ---
title: Scan on every core
rail: Use every core
note: Recordings are independent, so a corpus scans in parallel. Bare ~--parallel~ takes every core the machine's memory will hold; ~--parallel 8~ fixes the number. The output folder is **identical either way**, and one row per worker means a stalled lane is visible rather than buried.
term:
$ siar-app run ~/three-week-stream -a all_structures --out ~/stream-scan --parallel
! [████████████░░░░░░░░░░░░]  48%  12043/25318 files  201.4 h of 418.2 h audio
. 12 workers  ·  38.1x realtime  ·  5:24:11 elapsed  ·  5:41:03 left  ·  91043 structures
  1  ████████░░░░░░  61%    38s  station-a/2026-07-03/0410.wav
  2  ██░░░░░░░░░░░░  17%    11s  station-a/2026-07-03/0420.wav
  3  ··············    —      idle

. # Each worker holds one recording's grid: memory, not cores, is usually the limit.

--- slide ---
title: Watch a long run
rail: Watch a long run
note: ~--tui~ gives the run **the whole screen** — the alternate buffer, like ~vim~ — cleared and repainted every quarter second: how far through it is, where the time is going, what it is finding, a row per worker, and any failures kept on screen rather than scrolling past. The stage block is the reason to use it: an hour in it says whether the time is going into the algorithm, where it should, or into the thumbnails. When the run finishes the panel **stays up** with the closing metrics in it and waits for **Ctrl-Q**, so the answer does not vanish with the last recording. Quitting hands your shell back exactly as it was, with the same summary in the scrollback.
term:
$ siar-app run ~/three-week-stream -a all_structures --out ~/stream-scan --parallel --tui
. ╭─ all_structures · 12 workers ──────────────────── 5:24:11 elapsed ─╮
! │ ██████████████░░░░░░░░░░░░░░░  48%  12043/25318 files  201.4 h     │
! │ 38.1x realtime  ·  5:41:03 left  ·  91,043 structures  ·  3 errors │
. ├─ time by stage ──────────────┬─ structures found ──────────────────┤
! │ scan     48.90 h ██████ 91%  │ click     41,882  ██████████        │
. │ fft       2.71 h █·····  5%  │ tonal     39,014  █████████·        │
. │ thumbnail 1.09 h ······  2%  │ patch      6,131  █·········        │
. ├─ workers ──────────────────────────────────────────────────────────┤
  │   1  ████████░░░░  61%   38s  station-a/2026-07-03/0410.wav        │
  │   2  ██░░░░░░░░░░  17%   11s  station-a/2026-07-03/0420.wav        │
. │   3  ············   idle                                           │
. ╰────────────────────────────────────────────────────────────────────╯

. # Ctrl-Q closes the finished panel. Piped into a log it falls back to one line per recording.

--- slide ---
title: Look at a scan on another machine
rail: View it remotely
note: Scanned on a server? The output folder holds a copy of every recording, so it is the one thing you cannot conveniently move. ~siar-app serve~ serves it **where it is**, read-only, and prints the ~ssh -L~ line to copy. Opening a lane sends a **reduced spectrogram** computed on that box — about 350 KB whatever the recording's length — not the audio.
term:
$ siar-app serve ~/survey-scan
. FIELD       VALUE
- ----------  ----------------------------------------------
. folder      /home/user/survey-scan
recordings  412  (411 scanned, 1 errors)
. audio       27.40 h
! structures  9,043
. state       complete
! url         http://127.0.0.1:8420/?t=jLhljXDeneOuLwuEqn5w

. On your laptop, run:
$ ssh -N -L 8420:localhost:8420 you@survey-box
. then open the URL above.

+ Read-only: nothing this serves can change the folder.

. # Serve a run that is still going — the page fills as recordings land.

--- slide ---
title: Peek at the result
rail: Peek at the result
note: The output folder mirrors your input folder's layout. In the web app use **Open folder** and pick it or drop the folder into the Ident dynamics webapp.
term:
$ tree ~/survey-scan
/home/user/survey-scan/
  station-c/
    2025-09-08T1400.wav              copy of your recording
!     2025-09-08T1400.structures.json  what the scanner found
    2025-09-08T1400.png              the lane thumbnail
.   siar-app-run.json                  what was run, and what came back
.   siar-app-performance.json          what it cost

. # Every sidecar declares the family of the model that wrote it.

--- slide ---
title: View past runs
rail: View past runs
note: Local history and not an account-wide one. Read from ~/.siar-app/runs.json. Provides data for when, algorithm name, number of files, number ofstructures found, and output folder.
term:
$ siar-app runs --limit 3
. WHEN                  ALGORITHM       FILES  FOUND  OUTPUT
- --------------------  --------------  -----  -----  ---------------------
2026-08-10T11:26:00Z  all_structures    412   9043  /home/user/survey-scan
2026-08-09T08:02:11Z  fuzzy_hp_alpha     20    311  /home/user/hp-trial

. # --json gives the raw history.

--- slide ---
title: Rate the algorithm
rail: Rate the algorithm
note: **Important** Help us build better models and user select more accurate ones. Provide model performance feedback.
term:
$ siar-app feedback all_structures -s 7 -m "found the sweeps, missed two faint tonals"
+ Thanks — all_structures 1.1.0 rated 7/9.
That build now averages 6.7/9 from 3 ratings.

. # 0-2 nothing useful   3-5 needed sorting
. # 6-7 found it, some noise   8-9 found it and little else

--- slide ---
title: Update / Delete application
rail: Update / Delete application
note: The token, licence, history and cached algorithms all survive an app upgrade.
term:
$ uv tool upgrade siar-app
. Updated siar-app v0.4.0 -> v0.5.0

. # Installed from git, so uv compares version numbers. Same number, newer commit?
$ uv tool install --force --python 3.13 git+https://github.com/energy-master/siar.git
+ Installed 1 executable: siar-app

. # Taking it off: the CLI and its private 3.13, then the workspace
$ uv tool uninstall siar-app
$ rm -rf ~/.siar-app

! siar-app logout is local. Revoke the token from your account page.

`;
