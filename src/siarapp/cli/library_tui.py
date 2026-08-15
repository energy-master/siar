# Vixen Intelligence c.2026
"""``siar-app lib`` — what this machine can run, and running it.

The flags are the right shape for a script and the wrong shape for the ten minutes before a
survey. That is when the questions are *what have I got, which of it runs here, what did the
model I built last week actually key on, and what happens if I point it at this folder* — four
questions currently answered by three commands and one sibling tool's database, in an order
nobody remembers. This screen puts them on one page and then runs the scan from the same page,
because the answer to the fourth question is a run and everything above it was chosen in order to
start one.

Three panes, top to bottom, and the order is the decision being made:

* **Models** — the downloaded bundles and the models built here with ``siar-build``, together.
  :mod:`siarapp.library` is what merges them; nothing here knows how either got onto the disk.
* **Bots** — the programs a built model came out of, champion first, and what each one reads. A
  downloaded bundle has none to show and says so: what is inside it is the closed side of the
  seam, and a screen that implied otherwise would be lying about the product.
* **Run** — what to read, where to write, and whether to use the whole machine. Then Enter, and
  the run takes the screen.

**Paths are pointed at, not typed.** Both path rows open a browser
(:class:`Picker`) over the folder they are currently pointing at: directories and recordings,
Enter to open one, ``←`` to go back up, and a row at the top of the list that chooses the folder
you are standing in. A survey drive's paths are long, and typing one from memory into a field with
no shell behind it is where the evening goes. Somebody who would rather type still can — ``n``
names a folder outright, and takes a whole path as readily as a name.

The run itself is handed to :class:`~siarapp.cli.tui.TuiDisplay`, exactly as ``siar-app run
--tui`` hands it over: ``--tui`` already answers "is it finishing, where is the time going, is it
finding anything" better than a second panel would, and one definition of what a run looks like is
worth more than a bespoke one here. The library takes the screen back when the reader dismisses
it.

Everything that builds a line is a pure function of its arguments — :func:`render` takes state and
a size and returns a list of strings — so the layout is testable without a pty anywhere near it.
The terminal itself is :mod:`siarapp.cli.screen`'s job.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable

from siarapp.cli import screen
from siarapp.cli.format import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    RED,
    YELLOW,
    duration,
    fit,
    fit_path,
    human_bytes,
    paint,
    spin,
    stamp,
)
from siarapp.cli.progress import Throughput
from siarapp.cli.table import render_table
from siarapp.cli.tui import TuiDisplay
from siarapp.config import recent_cost, run_history
from siarapp.grid import ScannerError
from siarapp.io.audio import count_recordings, find_recordings, is_audio, probe
from siarapp.library import Model, feature_usage, library
from siarapp.loader import load_cached, load_local
from siarapp.runner import DEFAULT_MAX_BYTES, RunOptions

__all__ = ["Form", "Library", "Picker", "complete_path", "render", "run"]

#: How long a frame waits for a keystroke before drawing itself again. Nothing on this screen
#: moves on its own, so this is only how quickly a resized window catches up.
_TICK_SEC = 0.2

#: The panes, in Tab order. Also the order of the decision: which model, which of its bots, and
#: then the run.
_PANES = ("models", "bots", "run")

#: The run pane's rows. ``start`` is a row rather than a bare key binding so that "how do I
#: actually start it" is answered by the screen instead of by the footer alone.
_RUN_ROWS = ("input", "output", "parallel", "start")

#: ``--parallel`` as this screen offers it: off, or every core the machine's memory will hold.
#: The number in between is a real flag and a rare decision — somebody sizing a pool by hand has
#: read the corpus and is at a shell — and a screen that offered 1, 2, 4, 8, 16 would be asking a
#: question whose right answer is nearly always the same one.
_PARALLEL_OFF = 1
_PARALLEL_AUTO = 0

#: Rows kept for each list before the panels below start losing lines. A list of one is not a
#: list, and the detail under it is what the screen is for.
_MIN_MODEL_ROWS = 3
_MIN_BOT_ROWS = 2

#: Features named on a bot's own row before the rest become "+N more". The full list is in the
#: detail block; this is the glance.
_FEATURE_PREVIEW = 4


def _rule(width: int) -> str:
    """A full-width rule."""
    return paint("─" * max(0, width), DIM)


def _heading(text: str, width: int) -> str:
    """A section heading: the title, then a rule to the right edge."""
    name = fit_path(text, max(0, width - 2))
    return paint(f"{name} " + "─" * max(0, width - len(name) - 1), DIM)


def _table_lines(headers, rows, align=None) -> tuple[str, list[str]]:
    """A listing as ``(header line, one line per row)``, widths measured over **every** row.

    Every listing in this CLI goes through :func:`~siarapp.cli.table.render_table`, and a
    scrolling one is no exception — hand-counted format strings drift the moment a slug outgrows
    its column. Measuring over all the rows rather than over the visible window is what stops the
    columns jumping sideways as the cursor scrolls.
    """
    if not rows:
        return render_table(headers, [], align=align, rule=False), []
    lines = render_table(headers, rows, align=align, rule=False).split("\n")
    return lines[0], lines[1:]


def _window(cursor: int, count: int, rows: int) -> int:
    """First index to draw so that ``cursor`` is visible and roughly centred."""
    if count <= rows:
        return 0
    return max(0, min(cursor - rows // 2, count - rows))


def _short(path: str) -> str:
    """A path with the home directory written as ``~``. Shorter, and easier to recognise."""
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if home and path.startswith(home) else path


def complete_path(text: str) -> str:
    """Complete a partly-typed path as far as it is unambiguous.

    The typed way in, behind ``n``: a prefix matching one entry is completed to it, a prefix
    matching several is completed to their common start, and one matching nothing comes back
    untouched. Nothing is guessed.

    Args:
        text: What has been typed, as typed — ``~`` included.

    Returns:
        The completion, with a trailing separator when it names a directory, so a second Tab
        carries on inside it.
    """
    raw = os.path.expanduser(text or "")
    directory, _, prefix = raw.rpartition(os.sep)
    directory = directory or ("." if not raw.startswith(os.sep) else os.sep)
    try:
        names = sorted(n for n in os.listdir(directory) if n.startswith(prefix))
    except OSError:
        return text
    if not names:
        return text
    common = names[0]
    for name in names[1:]:
        while not name.startswith(common):
            common = common[:-1]
    if not common:
        return text
    completed = os.path.join(directory, common)
    if len(names) == 1 and os.path.isdir(completed):
        completed += os.sep
    # Given back in the spelling it was typed in: a field that silently expanded ``~`` on the
    # first Tab would rewrite the line under the cursor for no reason the typist asked for.
    home = os.path.expanduser("~")
    if text.startswith("~") and completed.startswith(home):
        return "~" + completed[len(home):]
    return completed


# -- choosing a path ---------------------------------------------------------------------------


#: What a row of the browser is. The first three are actions, the last two are the folder.
_UP, _USE, _NEW, _DIR, _FILE = "up", "use", "new", "dir", "file"


class Picker:
    """Choosing a path by pointing at it: one folder on screen, its contents under the cursor.

    Opened over whichever path the row already holds, so "somewhere near where I was last time"
    costs no navigation at all. Only directories and recordings are listed — a survey drive is
    full of spreadsheets and photographs, and a browser that offered them would be asking the
    reader to do the filtering.

    Attributes:
        kind: ``"input"`` or ``"output"`` — which row is being answered. The output browser
            offers a folder that does not exist yet; the input one offers recordings.
        directory: The folder being shown.
        entries: ``(kind, label, path, note)`` per row, in :data:`_UP`-first order.
        cursor: Which row is selected.
        hidden: Files in this folder that are not recordings, counted rather than listed.
        naming: Whether a name is being typed instead of pointed at.
        buffer: What has been typed, while it is.
        error: Why the last folder would not open, or ``""``.
    """

    __slots__ = ("kind", "directory", "entries", "cursor", "hidden", "naming", "buffer", "error")

    def __init__(self, kind: str, start: str = "") -> None:
        self.kind = kind
        self.directory = _nearest_directory(start)
        self.entries: list[tuple[str, str, str, str]] = []
        self.cursor = 0
        self.hidden = 0
        self.naming = False
        self.buffer = ""
        self.error = ""
        self.load()

    @property
    def title(self) -> str:
        """What this browser is answering, for the top of the frame."""
        return ("CHOOSE WHAT TO SCAN — a folder of recordings, or one recording"
                if self.kind == "input" else "CHOOSE WHERE TO WRITE — a folder, new or existing")

    def load(self, keep: str = "") -> None:
        """Read the folder into rows.

        Args:
            keep: A path to leave the cursor on — the folder just stepped out of, so going up and
                then down again lands where it started rather than at the top of the list.
        """
        self.entries = []
        self.hidden = 0
        parent = os.path.dirname(self.directory)
        if parent and parent != self.directory:
            self.entries.append((_UP, "..", parent, "up to " + _short(parent)))
        self.entries.append((_USE, f"use this folder — {_short(self.directory)}",
                             self.directory, ""))
        if self.kind == "output":
            self.entries.append((_NEW, "new folder here…", self.directory,
                                 "type a name, or a whole path"))

        try:
            names = sorted(os.listdir(self.directory), key=str.lower)
        except OSError as e:
            self.error = f"cannot read {self.directory}: {e.strerror or e}"
            names = []
        else:
            self.error = ""

        directories, files = [], []
        for name in names:
            if name.startswith("."):
                continue
            path = os.path.join(self.directory, name)
            if os.path.isdir(path):
                directories.append((_DIR, name + os.sep, path, ""))
            elif is_audio(name):
                files.append((_FILE, name, path, human_bytes(_size(path))))
            else:
                self.hidden += 1
        # Folders before recordings: a corpus is nearly always one level down, and a station
        # folder holding four hundred wav files should not push its sibling folders off the
        # screen.
        self.entries += directories
        if self.kind == "input":
            self.entries += files
        else:
            self.hidden += len(files)

        # On the row that chooses this folder, unless we have just stepped up out of one. The
        # browser opens *in* the folder the row already points at, so that row is the answer
        # "keep what I had" — and it should take one keystroke, not two.
        self.cursor = next((i for i, e in enumerate(self.entries) if e[0] == _USE), 0)
        for index, (_kind, _label, path, _note) in enumerate(self.entries):
            if keep and os.path.abspath(path) == os.path.abspath(keep):
                self.cursor = index
                break

    def move(self, delta: int) -> None:
        """Move the cursor, stopping at the ends rather than wrapping."""
        if self.entries:
            self.cursor = max(0, min(len(self.entries) - 1, self.cursor + delta))

    def jump(self, char: str) -> None:
        """Type-ahead: the next row starting with ``char``, wrapping past the end.

        The one thing a list of four hundred stations needs that arrow keys cannot give it.
        """
        char = char.lower()
        order = list(range(self.cursor + 1, len(self.entries))) + list(range(self.cursor + 1))
        for index in order:
            if self.entries[index][1].lower().startswith(char):
                self.cursor = index
                return

    def open_directory(self, path: str, keep: str = "") -> None:
        """Show another folder, keeping the cursor on ``keep`` if it is in there."""
        self.directory = os.path.abspath(path)
        self.load(keep)

    def up(self) -> None:
        """Go to the parent folder, with the cursor on the one just left."""
        parent = os.path.dirname(self.directory)
        if parent and parent != self.directory:
            self.open_directory(parent, keep=self.directory)

    def activate(self) -> str:
        """Act on the selected row.

        Returns:
            The chosen path when the row was a choice, and ``""`` when it was a step — opening a
            folder, or starting to type a name. The caller cannot tell the two apart from the
            row alone, and should not have to.
        """
        if not self.entries:
            return ""
        kind, _label, path, _note = self.entries[self.cursor]
        if kind == _UP:
            self.up()
            return ""
        if kind == _DIR:
            self.open_directory(path)
            return ""
        if kind == _NEW:
            self.naming = True
            self.buffer = ""
            return ""
        return path

    # -- typing a name ---------------------------------------------------------------------

    def type(self, char: str) -> None:
        """One character into the name being typed."""
        if self.naming and char.isprintable():
            self.buffer += char

    def backspace(self) -> None:
        """Delete the character before the cursor. On an empty field, stop naming."""
        if not self.naming:
            return
        if self.buffer:
            self.buffer = self.buffer[:-1]
        else:
            self.naming = False

    def complete(self) -> None:
        """Tab-complete what is being typed, when it is a path rather than a name."""
        if self.naming and self.buffer.startswith(("/", "~", ".")):
            self.buffer = complete_path(self.buffer)

    def cancel_name(self) -> None:
        """Abandon the name and go back to the list."""
        self.naming = False
        self.buffer = ""

    def named(self) -> str:
        """What was typed, as a path, or ``""`` if it was nothing.

        A bare name is a folder inside the one on screen; anything that looks like a path is
        taken as one, because somebody who typed ``~/surveys/2026/out`` has said exactly what
        they want and re-rooting it under the current folder would be a surprise.
        """
        text = self.buffer.strip()
        if not text:
            return ""
        if text.startswith(("/", "~")):
            return os.path.abspath(os.path.expanduser(text))
        return os.path.abspath(os.path.join(self.directory, text))


def _nearest_directory(start: str) -> str:
    """The folder a browser should open at, given whatever the row currently holds.

    A path that exists opens itself (or its parent, for a file); one that does not — the usual
    case for an output folder that has not been created yet — opens the nearest ancestor that
    does, so naming ``~/surveys/2026/out`` and then browsing lands in ``~/surveys``, not at the
    root.
    """
    path = os.path.abspath(os.path.expanduser(start or os.getcwd()))
    while path and not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path if os.path.isdir(path) else os.getcwd()


def _size(path: str) -> int:
    """A file's size, or ``0`` when it cannot be stat'ed."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


#: The browser's two columns. The note carries what the row means — where ``..`` goes, how big a
#: recording is — rather than repeating the name.
_PICK_HEADERS = ("NAME", "")
_PICK_ALIGN = ("<", "<")

#: Keys under the browser, and under it while a name is being typed.
_PICK_KEYS = "↑↓ move   enter open/choose   ← up   ~ home   letters jump   esc cancel"
_PICK_NEW_KEYS = "type a name or a path   tab complete   enter create   esc back to the list"


def _picker_lines(picker: Picker, width: int, height: int) -> list[str]:
    """The browser, as a whole frame. Modal on purpose: one question, one screen."""
    lines = [
        fit(f"{paint('siar-app', BOLD)}  {picker.title}", width),
        _rule(width),
        paint(f" {_short(picker.directory)}", BOLD),
    ]
    if picker.error:
        lines.append(paint(f" {picker.error}", RED))

    footer = [_rule(width),
              paint(" " + (_PICK_NEW_KEYS if picker.naming else _PICK_KEYS), DIM)]
    if picker.naming:
        footer.insert(0, paint(f" new folder:  {picker.buffer}" + paint("▏", BOLD), CYAN))
    if picker.hidden:
        footer.insert(0, paint(f" {picker.hidden} other file(s) here, not recordings", DIM))

    rows = [[label, note] for _kind, label, _path, note in picker.entries]
    header, body = _table_lines(_PICK_HEADERS, rows, align=_PICK_ALIGN)
    room = max(1, height - len(lines) - len(footer))
    start = _window(picker.cursor, len(body), room)
    for offset, line in enumerate(body[start:start + room], start=start):
        kind = picker.entries[offset][0]
        selected = offset == picker.cursor
        text = f" {'▸' if selected else ' '} {line}"
        if selected:
            text = paint(text, CYAN)
        elif kind in (_USE, _NEW):
            text = paint(text, GREEN)
        elif kind == _UP:
            text = paint(text, DIM)
        lines.append(text)

    while len(lines) < height - len(footer):
        lines.append("")
    return [fit(line, width) for line in (lines[: height - len(footer)] + footer)[:height]]


# -- the run form ------------------------------------------------------------------------------


class Form:
    """The three answers a run needs beyond which model.

    Values only: what a path *is* comes from the browser, so there is no editing state here and
    no second place where a half-typed path could be the truth.

    Attributes:
        input: Folder of recordings, or one recording.
        out: Output folder to build.
        parallel: :data:`_PARALLEL_OFF` or :data:`_PARALLEL_AUTO`.
        cursor: Which of :data:`_RUN_ROWS` is selected.
        count: Recordings under the input path, as counted when it was chosen.
        counted: Whether that count is the whole of it. False when the walk hit a bound and the
            count is a floor — which is a different thing from zero, and must not be read as
            "there is nothing here".
        note: What the input path turned out to be — the recording count, or why there is none.
        out_touched: Whether the output path has been chosen. Until it has, selecting a different
            model renames it, because an output folder named after a model nobody chose is worse
            than one that follows the choice.
    """

    __slots__ = ("input", "out", "parallel", "cursor", "count", "counted", "note", "out_touched")

    def __init__(self, input: str = "", out: str = "", parallel: int = _PARALLEL_OFF) -> None:
        self.input = input
        self.out = out
        self.parallel = parallel
        self.cursor = 0
        self.count = 0
        self.counted = True
        self.note = ""
        self.out_touched = False

    @property
    def row(self) -> str:
        """The selected row's name."""
        return _RUN_ROWS[self.cursor]

    def move(self, delta: int) -> None:
        """Move between the rows."""
        self.cursor = max(0, min(len(_RUN_ROWS) - 1, self.cursor + delta))

    def toggle_parallel(self) -> None:
        """Off, or every core — the only two answers this screen offers."""
        self.parallel = (_PARALLEL_AUTO if self.parallel == _PARALLEL_OFF else _PARALLEL_OFF)

    def set_input(self, path: str, progress: Callable[[int], None] | None = None) -> None:
        """Take a chosen input path, and look at what is actually under it.

        Args:
            path: The folder or recording chosen.
            progress: Passed to :meth:`describe_input`, to say what is happening while a large
                folder is being counted.
        """
        self.input = path
        self.describe_input(progress)

    def set_output(self, path: str) -> None:
        """Take a chosen output path. From here on it is the reader's, not the model's."""
        self.out = path
        self.out_touched = bool(path)

    def source(self) -> str:
        """The input path, expanded and absolute — what a run would actually read."""
        return os.path.abspath(os.path.expanduser(self.input)) if self.input else ""

    def destination(self) -> str:
        """The output path, expanded and absolute."""
        return os.path.abspath(os.path.expanduser(self.out)) if self.out else ""

    def describe_input(self, progress: Callable[[int], None] | None = None) -> None:
        """Count what is under the input path, and say so on the form.

        Done once, when a path is chosen, and remembered: it is a walk of the corpus, which is
        free on a folder of clips and is not on a survey drive — and this screen redraws several
        times a second. :meth:`blocking` reads the count rather than walking again for the same
        reason.

        Bounded, by :func:`~siarapp.io.audio.count_recordings`. The path being described is often
        one nobody chose — the working directory this command was typed in — and a folder that
        takes a minute to walk must cost a line that says so, never an interface that has not
        appeared yet. What it cannot do is claim the folder is empty: an unfinished count says
        "more than this", and :meth:`blocking` lets the run go and walk it properly.

        Args:
            progress: Called with the count so far while the walk runs, so the screen can say
                what is happening. ``None`` walks silently.
        """
        source = self.source()
        self.count = 0
        self.counted = True
        if not source:
            self.note = ""
            return
        if not os.path.exists(source):
            self.note = "no such folder or file"
            return
        self.count, self.counted = count_recordings(source, on_progress=progress)
        if not self.counted:
            self.note = (f"{self.count:,}+ recordings — too big to count from here"
                         if self.count else "too big to count from here — the run will walk it")
            return
        if not self.count:
            self.note = "no .wav or .flac here"
            return
        if self.count == 1:
            # A walk that finished inside the budget is one that can be afforded twice, and the
            # length of the single recording is worth the second pass.
            found = find_recordings(source)
            info = probe(found[0]) if found else None
            length = f" · {duration(info.duration_sec)}" if info is not None else ""
            self.note = f"1 recording{length}"
            return
        self.note = f"{self.count:,} recordings"

    def blocking(self, model: Model | None) -> str:
        """Why this run cannot start, or ``""`` when it can.

        One sentence, and the first one that applies, in the order the form is filled in. A form
        that listed every objection at once would be answering a question nobody asked: the
        reader fixes one thing and presses Enter again.
        """
        if model is None:
            return "nothing to run — no models on this machine yet"
        if not model.runnable:
            return model.note or f"{model.slug} cannot run on this machine"
        if not self.input:
            return "choose what to scan (i)"
        source = self.source()
        if not os.path.exists(source):
            return f"{self.input} is not a folder or a recording"
        if self.counted and not self.count:
            return f"no .wav or .flac under {self.input}"
        if not self.out:
            return "choose where to write (o)"
        if self.destination() == source:
            return "the output folder must not be the folder being scanned"
        return ""


class Library:
    """What the screen is showing, and where the cursor is in it.

    Loaded once on open and on demand afterwards: the rows are plain values by then, so redrawing
    costs nothing and neither the algorithm cache nor siar-build's index is read again per frame.

    Attributes:
        models: Everything runnable here — downloaded first, then built.
        runs: This machine's run history, newest first.
        cursor: Which model is selected.
        program_cursor: Which of its bots is selected.
        focus: Which pane the arrow keys move — one of :data:`_PANES`.
        form: The run form.
        picker: The path browser while one is open, else ``None``. It is modal: when it is there,
            it is the screen and it has the keys.
        message: A line under the form: the outcome of the last run, or why one did not start.
        message_kind: ``"ok"``, ``"warn"`` or ``"error"``, which is all the colour it gets.
        counting: What to call while a chosen folder is being counted, set by :func:`run` to a
            function that redraws the frame. Left ``None`` in a test, where there is no screen
            to redraw and nothing waiting to see one.
    """

    __slots__ = ("models", "runs", "cursor", "program_cursor", "focus", "form", "picker",
                 "message", "message_kind", "counting")

    def __init__(self, models: list[Model] | None = None, form: Form | None = None,
                 runs: list[dict] | None = None) -> None:
        self.models = list(models or [])
        self.runs = list(runs or [])
        self.cursor = 0
        self.program_cursor = 0
        self.focus = "models"
        self.form = form or Form()
        self.picker: Picker | None = None
        self.message = ""
        self.message_kind = "ok"
        self.counting: Callable[[int], None] | None = None

    # -- state ---------------------------------------------------------------------------------

    def load(self) -> None:
        """Read the algorithm cache, siar-build's index and the run history.

        All three at once, and only when something has changed — a frame is drawn several times a
        second and none of these would say anything different in between.
        """
        selected = self.model()
        self.models = library()
        self.runs = run_history()
        self.cursor = 0
        if selected is not None:
            for i, model in enumerate(self.models):
                if model.slug == selected.slug and model.source == selected.source:
                    self.cursor = i
                    break
        self.program_cursor = 0
        self.follow_model()

    def model(self) -> Model | None:
        """The selected model, or ``None`` when there are none."""
        return self.models[self.cursor] if self.models else None

    def programs(self) -> list:
        """The selected model's bots."""
        model = self.model()
        return model.programs if model is not None else []

    def program(self):
        """The selected bot, or ``None``."""
        programs = self.programs()
        return programs[self.program_cursor] if programs else None

    def move(self, delta: int) -> None:
        """Move the cursor in whichever pane has the focus."""
        if self.focus == "run":
            self.form.move(delta)
            return
        if self.focus == "bots" and self.programs():
            self.program_cursor = max(0, min(len(self.programs()) - 1,
                                             self.program_cursor + delta))
            return
        if not self.models:
            return
        moved = max(0, min(len(self.models) - 1, self.cursor + delta))
        if moved != self.cursor:
            self.cursor = moved
            self.program_cursor = 0
            self.follow_model()

    def focus_next(self, delta: int = 1) -> None:
        """Move to the next pane, skipping the bots pane when there are no bots to move in."""
        index = _PANES.index(self.focus)
        for _ in range(len(_PANES)):
            index = (index + delta) % len(_PANES)
            if _PANES[index] != "bots" or self.programs():
                break
        self.focus = _PANES[index]

    def follow_model(self) -> None:
        """Point the unchosen output folder at the model that is now selected.

        Only while it is unchosen. The moment somebody picks an output folder it is theirs, and a
        screen that renamed it under them because they scrolled the list would be unusable.
        """
        model = self.model()
        if model is None or self.form.out_touched:
            return
        self.form.out = os.path.join(os.getcwd(), f"{model.slug}-scan")

    def open_picker(self, kind: str) -> None:
        """Open the browser over whichever path that row is currently pointing at."""
        self.focus = "run"
        self.form.cursor = _RUN_ROWS.index(kind if kind == "input" else "output")
        start = self.form.input if kind == "input" else self.form.out
        self.picker = Picker(kind, start)

    def choose(self, path: str) -> None:
        """Take what the browser came back with, and close it.

        The browser is closed *before* the input path is counted, not after: counting is the one
        thing here that can take more than an instant, and the frame drawn under it should be the
        form the answer is going onto rather than the file list the reader has finished with.
        """
        if self.picker is None:
            return
        kind = self.picker.kind
        self.picker = None
        if kind == "input":
            self.say("")
            self.form.set_input(path, self.counting)
            self.say("")
        else:
            self.form.set_output(path)

    def say(self, message: str, kind: str = "ok") -> None:
        """Put one line under the form."""
        self.message = message
        self.message_kind = kind

    def options(self) -> RunOptions:
        """The form as :class:`~siarapp.runner.RunOptions`.

        Everything except ``--parallel`` is left at its default. This screen is for choosing a
        model and pointing it at a folder; a run that needs a different FFT or a channel other
        than the mix is one somebody has thought about, and they have flags.
        """
        return RunOptions(workers=self.form.parallel, max_bytes=DEFAULT_MAX_BYTES)

    def last_run(self) -> dict | None:
        """The most recent run of the selected model on this machine, if there was one."""
        model = self.model()
        if model is None:
            return None
        for row in self.runs:
            if isinstance(row, dict) and row.get("algorithm") == model.slug:
                return row
        return None


# -- drawing -------------------------------------------------------------------------------


#: Header and cells go through one renderer, so the columns are measured rather than counted.
_MODEL_HEADERS = ("NAME", "WHERE FROM", "VERSION", "RUNS ON", "SIZE", "WHEN", "BOTS", "RUNS HERE")
_MODEL_ALIGN = ("<", "<", "<", "<", ">", "<", ">", "<")
_BOT_HEADERS = ("#", "KIND", "FITNESS", "THRESHOLD", "NODES", "DEPTH", "READS")
_BOT_ALIGN = (">", "<", ">", ">", ">", ">", "<")


def _number(value, spec: str = ".3f", missing: str = "—") -> str:
    """A number from the index, formatted, or a dash where the index has nothing."""
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return missing


def _model_rows(models: list[Model]) -> list[list[str]]:
    """One row per model, in the order they will be drawn."""
    return [
        [
            model.slug,
            "downloaded" if not model.built else "built here",
            model.version or "—",
            "this machine" if model.platform == "source" else model.platform,
            human_bytes(model.size_bytes) if model.size_bytes else "—",
            stamp(model.stamped_at),
            str(len(model.programs)) if model.programs else "—",
            "yes" if model.runnable else "no",
        ]
        for model in models
    ]


def _bot_rows(programs: list) -> list[list[str]]:
    """One row per bot of the selected model."""
    rows = []
    for program in programs:
        reads = ", ".join(program.features[:_FEATURE_PREVIEW])
        extra = len(program.features) - _FEATURE_PREVIEW
        if extra > 0:
            reads += f"  +{extra} more"
        rows.append([
            str(program.rank),
            program.kind.replace("_", "-"),
            _number(program.fitness, ".4f"),
            _number(program.threshold, ".4f") if program.calibrated else "—",
            str(program.n_nodes),
            str(program.depth),
            reads or "—",
        ])
    return rows


def _list_lines(headers, rows, align, cursor: int, rows_shown: int, focused: bool) -> list[str]:
    """A scrolling list: its header, then the visible window, the selection marked."""
    header, body = _table_lines(headers, rows, align=align)
    out = [paint(f"   {header}", DIM)]
    start = _window(cursor, len(body), rows_shown)
    for offset, line in enumerate(body[start:start + rows_shown], start=start):
        selected = offset == cursor
        text = f" {'▸' if selected else ' '} {line}"
        out.append(paint(text, CYAN if focused else GREEN) if selected else text)
    return out


def _model_detail(model: Model | None, models: list[Model], width: int) -> list[str]:
    """What the selected model is, in the terms its own side of the seam allows.

    A built model can be described completely — the corpus, the band, the gates, the features —
    because it was built here. A downloaded one is described by its manifest and no further: what
    is inside the bundle is the closed side, and inventing detail about it would be worse than
    the blank.
    """
    if model is None:
        return []
    lines = [_heading(f"MODEL — {model.slug}", width)]
    if not model.runnable:
        lines.append(paint(f"  cannot run here: {model.note}", YELLOW))

    if not model.built:
        family = model.detail.get("family") or "structure_scanners"
        lines += [
            f"  {model.title or model.slug}{paint('   ' + family, DIM)}",
            paint(f"  bundle {model.path}", DIM),
            paint("  downloaded from IDent Dynamics — its internals stay inside the bundle, "
                  "which is the point of it", DIM),
        ]
        return lines

    detail = model.detail
    band = "auto"
    if detail.get("fmax_hz") is not None:
        band = f"{float(detail.get('fmin_hz') or 0):.0f}-{float(detail['fmax_hz']):.0f} Hz"
    audio = ("geometry not recorded" if detail.get("sample_rate") is None else
             f"{detail['sample_rate']} Hz · fft {detail.get('n_fft')} · hop {detail.get('hop')} · "
             f"band {band} · {detail.get('n_bins')} bins")
    gates = (f"held-out {_number(detail.get('held_out_auc'))} window / "
             f"{_number(detail.get('held_out_auc_recording'))} recording · "
             f"null {_number(detail.get('null_auc'))}")
    if detail.get("suspect"):
        gates += paint("   SUSPECT", RED)
    if detail.get("parity_ok") is not None:
        gates += " · parity " + ("ok" if detail["parity_ok"] else paint("FAILED", RED))

    usage = dict(feature_usage(models))
    features = ", ".join(
        f"{name}{paint(f'×{usage[name]}', DIM) if usage.get(name, 0) > 1 else ''}"
        for name in model.features
    )
    lines += [
        f"  target {detail.get('target') or model.slug}"
        f"{paint('   built ' + stamp(model.stamped_at) + ' by siar-build ' + model.version, DIM)}",
        paint(f"  corpus {detail.get('input_dir') or '—'}", DIM),
        f"  audio  {audio}",
        f"  result {gates}",
        f"  reads  {features or '(no features recorded)'}",
        paint(f"  package {model.path or 'not packaged'}", DIM),
    ]
    if detail.get("stopped_at"):
        lines.append(paint(f"  stopped at {detail['stopped_at']} — {detail.get('reason') or ''}",
                           YELLOW))
    return lines


def _bot_detail(program, width: int) -> list[str]:
    """The selected bot's expression and calibration, wrapped to the width."""
    if program is None:
        return []
    lines = [_heading(f"BOT — {program.kind.replace('_', '-')} #{program.rank}", width)]
    room = max(20, width - 4)
    text = program.infix
    for _ in range(2):
        if not text:
            break
        lines.append(f"  {text[:room]}")
        text = text[room:]
    if text:
        lines[-1] = lines[-1][:width - 2] + "…"
    if program.calibrated:
        lines.append(paint(f"  threshold {program.threshold:.6f} · "
                           f"polarity {program.polarity:+d} · {program.n_nodes} nodes · "
                           f"depth {program.depth}", DIM))
    else:
        lines.append(paint("  uncalibrated — recorded by siar-build, but not runnable", YELLOW))
    return lines


#: The run history's columns, in the order a row is read: when, with what, over what it produced.
_RUN_HEADERS = ("WHEN", "MODEL", "FILES", "STRUCTURES", "WORKERS", "TOOK", "OUTPUT FOLDER")
_RUN_ALIGN = ("<", "<", ">", ">", ">", ">", "<")


def _recent_lines(runs: list[dict], limit: int) -> list[str]:
    """What has been run from this machine, newest first, as many as ``limit`` allows.

    This is the section with no natural length, so it is built last and given whatever the frame
    has left — the same bargain :mod:`siarapp.cli.tui` strikes with its completions list. A tall
    terminal is then full of what this machine has done rather than empty below the form, and the
    output folder of the run somebody wants to open again is on screen instead of in another
    command.
    """
    if limit <= 0:
        return []
    rows = []
    for entry in runs:
        if not isinstance(entry, dict):
            continue
        rows.append([
            str(entry.get("at") or "").replace("T", " ")[:16],
            str(entry.get("algorithm") or "—"),
            f"{int(entry.get('files') or 0):,}",
            f"{int(entry.get('structures') or 0):,}",
            str(int(entry.get("workers") or 1)),
            duration(float(entry.get("elapsed_sec") or 0.0)),
            str(entry.get("out") or ""),
        ])
        if len(rows) >= limit:
            break
    if not rows:
        return []
    header, body = _table_lines(_RUN_HEADERS, rows, align=_RUN_ALIGN)
    return [paint(f"   {header}", DIM)] + [f"   {line}" for line in body]


def _parallel_text(workers: int) -> str:
    """How ``--parallel`` reads on the form."""
    if workers == _PARALLEL_OFF:
        return "off — one recording at a time, in this process"
    return "auto — one worker per core, as many as this machine's memory will hold"


def _run_lines(lib: Library, width: int) -> list[str]:
    """The run form: three answers, a start row, and what is in the way of using it."""
    form = lib.form
    model = lib.model()
    blocking = form.blocking(model)
    values = {
        "input": (_short(form.source()) + paint(f"   {form.note}", DIM) if form.input
                  else paint("(press enter to choose)", DIM)),
        "output": (_short(form.destination()) if form.out
                   else paint("(press enter to choose)", DIM)),
        "parallel": _parallel_text(form.parallel),
        "start": (paint(f"run {model.slug} now", BOLD) if model is not None and not blocking
                  else paint("run", DIM)),
    }
    labels = {"input": "scan", "output": "write to", "parallel": "parallel", "start": "▶ start"}

    lines = [_heading("RUN", width)]
    for index, row in enumerate(_RUN_ROWS):
        focused = index == form.cursor and lib.focus == "run"
        text = f" {'▸' if focused else ' '} {labels[row]:<9} {values[row]}"
        lines.append(paint(text, CYAN) if focused else text)

    if blocking:
        lines.append(paint(f"   {blocking}", YELLOW))
    else:
        last = lib.last_run()
        if last:
            lines.append(paint(
                f"   last run of this model: {last.get('files', 0)} files · "
                f"{last.get('structures', 0):,} structures · "
                f"{duration(float(last.get('elapsed_sec') or 0.0))} · {last.get('out', '')}",
                DIM))
        else:
            lines.append(paint("   ready", DIM))
    return lines


def _title(lib: Library, width: int) -> str:
    """The one line that says what this machine has."""
    downloaded = sum(1 for m in lib.models if not m.built)
    built = sum(1 for m in lib.models if m.built)
    bots = sum(len(m.programs) for m in lib.models)
    features = len(feature_usage(lib.models))
    counts = (f"{downloaded} downloaded · {built} built here · "
              f"{bots} bot{'' if bots == 1 else 's'} · "
              f"{features} feature{'' if features == 1 else 's'}")
    return fit(f"{paint('siar-app', BOLD)}  library{paint('   ' + counts, DIM)}", width)


#: What the footer offers, in the order somebody needs it — and the same list for a terminal too
#: narrow to hold it, cut to the keys that cannot be guessed. A footer whose right-hand end is
#: clipped is one that stops mentioning how to leave.
_KEYS = ("↑↓ select   tab pane   enter choose/run   i scan   o write to   p parallel   "
         "R reload   q quit")
_NARROW_KEYS = "↑↓ select  tab pane  enter choose/run  i/o/p rows  q quit"

#: Width below which the short list is used.
_NARROW = 96


def _bottom(lib: Library, width: int) -> list[str]:
    """The part of the frame that is never given up: the run form and the keys.

    Built separately from everything above it, and laid out from the bottom of the screen up,
    because the screen is *for* starting a run: a terminal too short for all of this should lose
    the second half of a model's provenance, never the line saying why the run will not start.
    """
    lines = [_rule(width)] + _run_lines(lib, width)
    if lib.message:
        colours = {"ok": GREEN, "warn": YELLOW, "error": RED}
        lines.append(paint(f" {lib.message}", colours.get(lib.message_kind, GREEN)))
    lines.append(_rule(width))
    lines.append(paint(" " + (_KEYS if width >= _NARROW else _NARROW_KEYS), DIM))
    return lines


def _top(lib: Library, width: int, room: int) -> list[str]:
    """Everything above the run form, laid out into ``room`` lines.

    The two lists take a third each and the detail takes the rest, with the run history filling
    whatever is still spare — it is the one section with no natural length. Anything that still
    does not fit is cut from the end, which is why the history is last and the model list is
    first: the order of the sections is the order they are worth keeping.
    """
    lines = [_title(lib, width), _rule(width)]
    if not lib.models:
        return lines + [
            "",
            "  Nothing on this machine to run yet.",
            paint("  `siar-app algorithms` lists what your account can download, and the first "
                  "run fetches one.", DIM),
            paint("  Models you build yourself with `siar-build` appear here as soon as they "
                  "are packaged.", DIM),
        ]

    programs = lib.programs()
    # Room for the two lists: what is left after this block's own fixed lines — the title, the
    # rules and each list's header — shared a third each, with the rest going to the detail.
    share = max(0, room - len(lines) - 3 - 2)
    model_rows = min(len(lib.models), max(_MIN_MODEL_ROWS, share // 3))
    share -= model_rows
    bot_rows = min(len(programs), max(_MIN_BOT_ROWS, share // 3)) if programs else 0

    lines += _list_lines(_MODEL_HEADERS, _model_rows(lib.models), _MODEL_ALIGN,
                         lib.cursor, model_rows, lib.focus == "models")
    lines.append(_rule(width))

    if programs:
        lines += _list_lines(_BOT_HEADERS, _bot_rows(programs), _BOT_ALIGN,
                             lib.program_cursor, bot_rows, lib.focus == "bots")
    else:
        model = lib.model()
        lines.append(paint(
            "   no bots to show — a downloaded bundle keeps its programs inside it"
            if model is not None and not model.built
            else "   this build recorded no programs", DIM))

    # What the lists did not use is the detail's, and what the detail does not use is the
    # history's. Measured from the lines actually drawn rather than from the share above, so a
    # short list hands its rows down rather than leaving them blank.
    spare = room - len(lines) - 1
    detail = _model_detail(lib.model(), lib.models, width)
    if lib.program() is not None and (lib.focus == "bots" or spare > len(detail) + 4):
        detail += _bot_detail(lib.program(), width)
    detail = detail[: max(0, spare)]
    # The rule belongs to the detail, not to the list above it: two rules with nothing between
    # them is what a screen too short for the provenance would otherwise draw, and it reads as a
    # section that failed rather than as one that was not there.
    if detail:
        lines.append(_rule(width))
        lines += detail

    spare = room - len(lines) - 1
    if spare > 0:
        recent = _recent_lines(lib.runs, spare)
        if recent:
            lines.append(_heading("RUN HISTORY ON THIS MACHINE", width))
            lines += recent
    return lines


def render(lib: Library, width: int, height: int) -> list[str]:
    """The whole screen, as lines. Pure — no terminal, no clock, no I/O.

    Args:
        lib: The state to draw.
        width: Columns available.
        height: Rows available.

    Returns:
        Exactly ``height`` lines of exactly ``width`` visible columns — or fewer lines only when
        the terminal is too short for the run form itself, which nothing above can be traded for.
    """
    if lib.picker is not None:
        return _picker_lines(lib.picker, width, height)

    bottom = _bottom(lib, width)
    room = max(0, height - len(bottom))
    top = _top(lib, width, room)[:room]
    # A rule left at the end by that cut has nothing under it any more. It would read as a
    # section that failed rather than as one there was no room for.
    if top and top[-1] == _rule(width):
        top = top[:-1]
    lines = top + [""] * (room - len(top)) + bottom
    # Every line to exactly the width, here rather than in each section: a listing column or an
    # output folder that outgrows the terminal must lose its tail, not wrap — a wrapped line
    # pushes the frame past the bottom of the screen and the next repaint lands in the wrong
    # place.
    return [fit(line, width) for line in lines[:height]]


# -- the controller -------------------------------------------------------------------------


def _handle_picker(lib: Library, key: str) -> None:
    """One keystroke while the path browser is open. It is modal: nothing else sees these."""
    picker = lib.picker
    if picker is None:
        return

    if picker.naming:
        if key == "enter":
            chosen = picker.named()
            if chosen:
                lib.choose(chosen)
            else:
                picker.cancel_name()
        elif key == "escape":
            picker.cancel_name()
        elif key == "backspace":
            picker.backspace()
        elif key == "tab":
            picker.complete()
        elif key == "space":
            picker.type(" ")
        elif key == "\x15":  # Ctrl-U
            picker.buffer = ""
        elif len(key) == 1:
            picker.type(key)
        return

    if key in ("escape", "eof"):
        lib.picker = None
    elif key in ("up", "k"):
        picker.move(-1)
    elif key in ("down", "j"):
        picker.move(1)
    elif key in ("left", "backspace"):
        picker.up()
    elif key in ("enter", "right", "space"):
        chosen = picker.activate()
        if chosen:
            lib.choose(chosen)
    elif key == "~":
        picker.open_directory(os.path.expanduser("~"))
    elif key == "n" and picker.kind == "output":
        picker.naming = True
        picker.buffer = ""
    elif len(key) == 1 and key.isprintable():
        picker.jump(key)


def _handle_key(lib: Library, key: str) -> str:
    """One keystroke on the library screen.

    Args:
        lib: The state, mutated in place.
        key: What :func:`siarapp.cli.screen.read_key` returned.

    Returns:
        ``"run"`` when the reader asked to start a scan, ``"quit"`` to leave, ``""`` otherwise.
        Returned rather than acted on, so that the whole of the key handling stays testable
        without a terminal or a scan.
    """
    if lib.picker is not None:
        _handle_picker(lib, key)
        return ""

    if key in ("q", "Q", "eof"):
        return "quit"
    if key in ("up", "k"):
        lib.move(-1)
    elif key in ("down", "j"):
        lib.move(1)
    elif key == "tab":
        lib.focus_next()
    elif key in ("left", "right"):
        if lib.focus == "run" and lib.form.row == "parallel":
            lib.form.toggle_parallel()
        else:
            lib.focus_next(1 if key == "right" else -1)
    elif key == "i":
        lib.open_picker("input")
    elif key == "o":
        lib.open_picker("output")
    elif key == "p":
        lib.focus, lib.form.cursor = "run", _RUN_ROWS.index("parallel")
        lib.form.toggle_parallel()
    elif key == "R":
        lib.load()
        lib.say("reloaded")
    elif key == "enter":
        if lib.focus != "run":
            lib.focus = "run"
        elif lib.form.row in ("input", "output"):
            lib.open_picker(lib.form.row)
        elif lib.form.row == "parallel":
            lib.form.toggle_parallel()
        else:
            return "run"
    return ""


def _load_algorithm(model: Model):
    """Load the selected model, whichever side of the seam it is on.

    Raises:
        ScannerError: If the bundle or package will not import, or is no longer on disk.
    """
    if model.built:
        return load_local(model.path, model.slug)
    handle = load_cached(model.slug, model.platform)
    if handle is None:
        raise ScannerError(
            f"{model.slug} is no longer unpacked at {model.path} — "
            f"`siar-app run -a {model.slug}` will fetch it again"
        )
    return handle


def _start_run(lib: Library) -> None:
    """Hand the screen to the run panel, run the scan, and take the screen back.

    The library's frame is on the alternate screen and so is the run panel's, and
    :class:`~siarapp.cli.tui.TuiDisplay` enters and leaves that buffer itself. So the sequence is
    the panel's, not ours: it takes the screen when the corpus is sized, holds the finished run
    until the reader dismisses it, and gives the buffer back on the way out — at which point this
    takes it again and redraws.
    """
    model = lib.model()
    blocking = lib.form.blocking(model)
    if blocking:
        lib.say(blocking, "warn")
        return

    source, destination = lib.form.source(), lib.form.destination()
    try:
        handle = _load_algorithm(model)
    except ScannerError as e:
        lib.say(str(e), "error")
        return

    from siarapp.cli.commands import perform_run

    rate, shares = recent_cost(handle.slug)
    reporter = TuiDisplay(
        handle.slug,
        workers=lib.form.parallel if lib.form.parallel > 0 else 1,
        source=source, out=destination, prior=Throughput(rate, shares),
    )
    try:
        manifest, failure = perform_run(handle, source, destination, lib.options(), reporter)
    finally:
        # Unconditional: the panel may have handed the buffer back, or never taken it, and the
        # library has to be on the alternate screen either way before the next frame is drawn.
        screen.enter()

    if failure:
        lib.say(failure, "error")
        return
    # The run has just written itself into the history; the panel below the form is drawn from
    # this machine's history and would otherwise be one run out of date until the next reload.
    lib.runs = run_history()
    errors = int(manifest["by_status"].get("error") or 0)
    lib.say(
        f"{manifest['files']} file(s) · {manifest['structures']:,} structures · "
        f"{duration(float(manifest.get('elapsed_sec') or 0.0))} · written to {destination}"
        + (f" · {errors} failed" if errors else ""),
        "warn" if errors else "ok",
    )


def _console_counting(label: str) -> Callable[[int], None]:
    """A spinner on the shell's own buffer, for the count that happens before the screen exists.

    The working directory is counted before the alternate screen is taken, because what it turns
    out to be decides what the form opens with. That count is the one thing on the way in that can
    take a noticeable moment — on a home directory, or a survey drive over a slow link — and a
    command that prints nothing while it does is a command that has hung, as far as anyone
    watching can tell.

    Args:
        label: The path being walked, already shortened to fit a line.

    Returns:
        A function to hand to :meth:`Form.describe_input` as its progress callback. Rewrites one
        line in place; :func:`_console_counted` takes the line back.
    """
    ticks = [0]

    def report(count: int) -> None:
        ticks[0] += 1
        found = f" — {count:,} so far" if count else ""
        sys.stdout.write(f"\r{spin(ticks[0])} looking for recordings in {label}{found}"
                         f"{screen.CLEAR_TO_EOL}")
        sys.stdout.flush()

    return report


def _console_counted() -> None:
    """Take back the line :func:`_console_counting` was drawing on.

    Unconditional: a walk that finished inside the first tick never drew anything, and clearing a
    line that was never written costs one escape sequence and leaves the terminal exactly as it
    was.
    """
    sys.stdout.write("\r" + screen.CLEAR_TO_EOL)
    sys.stdout.flush()


def run(args: Any = None) -> int:
    """Show the library. The whole command.

    Args:
        args: The parsed command line. Unused — the screen takes no flags, because every choice
            it offers is one it can also show you the options for.

    Returns:
        ``0`` when the reader left the screen, ``1`` when there is no terminal to draw on, and
        ``130`` on Ctrl-C — the same three codes the rest of the CLI uses.
    """
    if not screen.is_tty():
        print("siar-app: lib needs a terminal. `siar-app installed` lists what is here, and "
              "`siar-app run` scans with it.", file=sys.stderr)
        return 1

    lib = Library(form=Form(input=os.getcwd()))
    lib.load()
    lib.form.describe_input(_console_counting(fit_path(os.getcwd(), 48)))
    _console_counted()
    if not lib.form.count:
        # The working directory is a reasonable guess and a common wrong one. Offer it only when
        # the walk found recordings in it — nothing found, and nothing counted before the walk
        # gave up, both mean this is somebody's home directory rather than a corpus.
        lib.form.input = ""
        lib.form.note = ""
        lib.form.counted = True

    def counting(found: int) -> None:
        """Say what a walk started from the browser is doing, over the form it will land on."""
        ticks[0] += 1
        lib.say(f"{spin(ticks[0])} counting recordings — {found:,} so far")
        columns, rows = screen.size()
        screen.draw(render(lib, columns, rows), columns, rows)

    ticks = [0]
    lib.counting = counting

    try:
        with screen.screen():
            while True:
                width, height = screen.size()
                screen.draw(render(lib, width, height), width, height)
                key = screen.read_key(_TICK_SEC)
                if not key:
                    continue
                action = _handle_key(lib, key)
                if action == "quit":
                    return 0
                if action == "run":
                    _start_run(lib)
    except KeyboardInterrupt:
        # Ctrl-C during a scan started from here. The screen is already handed back; say what a
        # half-written output folder is worth rather than printing a traceback over it.
        print("\nInterrupted. Re-run with --resume, or from the library, to carry on where "
              "this stopped.", file=sys.stderr)
        return 130
