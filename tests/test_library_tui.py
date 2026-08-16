# Vixen Intelligence c.2026
"""The ``siar-app lib`` screen.

Three things are pinned here, and they are the three that break silently.

**Geometry.** Every line is exactly as wide as the terminal and there are never more lines than it
has rows, because the frame is repainted from the top left: a line one character too long wraps,
shifts everything below it, and the next repaint lands in the wrong place. The run form and the
key list are drawn from the bottom up and must survive a terminal too short for everything above
them — losing the provenance of a model is a nuisance, losing the line that says why the run will
not start is a screen nobody can use.

**The keys.** ``_handle_key`` is a pure function of state and one keystroke, so the whole
interaction is testable without a pty: which pane moves, which row opens the browser, and what
actually starts a run.

**The browser.** Choosing a path by pointing at it has to come back with the path the reader
pointed at, and a folder that cannot be read has to be a message rather than an exception.

Rendered directly rather than through a terminal, and every layout test runs twice — colour must
change how a frame looks, never its geometry.
"""
from __future__ import annotations

import os

import pytest

from siarapp.cli import library_tui as tui
from siarapp.cli.format import set_colour, visible_len
from siarapp.library import Model, Program


@pytest.fixture(autouse=True, params=[False, True], ids=["plain", "colour"])
def colour(request):
    """Both ways round, and back to the environment's own answer afterwards."""
    set_colour(request.param)
    yield request.param
    set_colour(None)


def _program(rank=0, kind="champion", threshold=-0.3, features=("band_1000hz",)):
    return Program(rank=rank, kind=kind, fitness=0.99, threshold=threshold, polarity=1,
                   n_nodes=15, depth=8, features=features, infix="band_1000hz - 0.5",
                   saved_path="/m.json")


def _downloaded(slug="all_structures", runnable=True):
    return Model(source="downloaded", slug=slug, title="Everything", version="1.0.0",
                 platform="linux-x86_64-cp313", path="/cache/" + slug, size_bytes=1024,
                 stamped_at=1_700_000_000, runnable=runnable,
                 note="" if runnable else "built for another machine",
                 detail={"family": "structure_scanners"})


def _built(slug="recall", runnable=True, programs=None):
    return Model(source="built", slug=slug, title="recall", version="0.1.0", platform="source",
                 path="/models/siar_recall", size_bytes=140_000, stamped_at=1_700_000_100,
                 runnable=runnable, note="" if runnable else "not packaged",
                 detail={"target": "recall", "input_dir": "/audio", "sample_rate": 96000,
                         "n_fft": 8192, "hop": 2048, "n_bins": 128, "fmin_hz": 5000.0,
                         "fmax_hz": 7800.0, "held_out_auc": 0.78, "null_auc": 0.51,
                         "parity_ok": 1},
                 programs=programs if programs is not None else [
                     _program(), _program(rank=1, kind="runner_up", threshold=None)])


def _library(models=None, runs=None):
    lib = tui.Library(models=models if models is not None else [_downloaded(), _built()],
                      runs=runs or [], form=tui.Form(input="/audio", out="/out"))
    return lib


#: Sizes every layout assertion is made at: a normal terminal, a small one, the floor
#: :func:`siarapp.cli.screen.size` clamps to, and one wide enough to show everything.
SIZES = [(120, 40), (100, 30), (80, 24), (60, 12), (200, 60)]


# -- geometry ---------------------------------------------------------------------------------


@pytest.mark.parametrize("width,height", SIZES)
def test_every_line_is_exactly_the_terminal_width(width, height):
    for line in tui.render(_library(), width, height):
        assert visible_len(line) == width


@pytest.mark.parametrize("width,height", SIZES)
def test_the_frame_never_outgrows_the_terminal(width, height):
    assert len(tui.render(_library(), width, height)) <= height


@pytest.mark.parametrize("width,height", SIZES)
def test_the_run_form_and_the_keys_survive_every_size(width, height):
    frame = "\n".join(tui.render(_library(), width, height))

    assert "start" in frame, "the row that starts a run is what the screen is for"
    assert "quit" in frame, "a screen with no way out drawn on it is one nobody can leave"


@pytest.mark.parametrize("width,height", SIZES)
def test_the_browser_fills_the_frame_too(width, height):
    lib = _library()
    tui._handle_key(lib, "i")

    frame = tui.render(lib, width, height)

    assert len(frame) <= height
    assert all(visible_len(line) == width for line in frame)


def test_an_empty_library_says_what_to_do_about_it():
    frame = "\n".join(tui.render(tui.Library(models=[]), 100, 30))

    assert "Nothing on this machine" in frame
    assert "siar-app algorithms" in frame and "siar-build" in frame


def test_a_downloaded_model_is_not_given_bots_it_does_not_have():
    lib = _library(models=[_downloaded()])

    frame = "\n".join(tui.render(lib, 120, 40))

    assert "no bots to show" in frame


def test_a_built_models_bots_and_features_are_on_the_screen():
    lib = _library(models=[_built()])

    frame = "\n".join(tui.render(lib, 120, 40))

    assert "champion" in frame and "runner-up" in frame
    assert "band_1000hz" in frame
    assert "5000-7800 Hz" in frame, "the band it was evolved in belongs beside its bots"


def test_a_model_that_cannot_run_here_says_so_rather_than_being_hidden():
    lib = _library(models=[_downloaded(runnable=False)])

    frame = "\n".join(tui.render(lib, 120, 40))

    assert "cannot run here" in frame


def test_the_history_fills_a_tall_terminal_rather_than_leaving_it_blank():
    runs = [{"at": "2026-08-15T21:05:00Z", "algorithm": "recall", "files": 8, "structures": 39,
             "workers": 2, "elapsed_sec": 20.5, "out": "/scans/recall"}] * 12

    short = "\n".join(tui.render(_library(runs=runs), 120, 24))
    tall = "\n".join(tui.render(_library(runs=runs), 120, 60))

    assert tall.count("/scans/recall") > short.count("/scans/recall")


# -- the keys ---------------------------------------------------------------------------------


def test_tab_moves_between_the_panes_and_skips_bots_when_there_are_none():
    lib = _library(models=[_downloaded()])

    tui._handle_key(lib, "tab")

    assert lib.focus == "run", "a pane with nothing in it is not a place to put the cursor"


def test_tab_stops_at_the_bots_of_a_model_that_has_some():
    lib = _library(models=[_built()])

    tui._handle_key(lib, "tab")

    assert lib.focus == "bots"


def test_moving_down_the_models_resets_the_bot_selection():
    lib = _library()
    lib.focus = "bots"
    lib.program_cursor = 1
    lib.focus = "models"

    tui._handle_key(lib, "down")

    assert lib.cursor == 1 and lib.program_cursor == 0


def test_the_output_folder_follows_the_model_until_it_is_chosen():
    lib = tui.Library(models=[_downloaded(), _built()], form=tui.Form())
    lib.follow_model()
    followed = lib.form.out

    tui._handle_key(lib, "down")
    assert lib.form.out != followed and lib.form.out.endswith("recall-scan")

    lib.form.set_output("/somewhere/else")
    tui._handle_key(lib, "up")
    assert lib.form.out == "/somewhere/else", "a chosen output folder is the reader's, not ours"


def test_parallel_is_off_or_auto_and_nothing_in_between():
    lib = _library()

    tui._handle_key(lib, "p")
    assert lib.form.parallel == tui._PARALLEL_AUTO
    tui._handle_key(lib, "p")
    assert lib.form.parallel == tui._PARALLEL_OFF


def test_q_quits_and_enter_on_the_start_row_runs():
    lib = _library()
    lib.focus = "run"
    lib.form.cursor = tui._RUN_ROWS.index("start")

    assert tui._handle_key(lib, "enter") == "run"
    assert tui._handle_key(lib, "q") == "quit"


def test_enter_on_a_path_row_opens_the_browser_rather_than_a_text_field():
    lib = _library()
    lib.focus = "run"
    lib.form.cursor = tui._RUN_ROWS.index("output")

    tui._handle_key(lib, "enter")

    assert lib.picker is not None and lib.picker.kind == "output"


def test_the_browser_has_the_keys_while_it_is_open():
    lib = _library()
    tui._handle_key(lib, "i")

    assert tui._handle_key(lib, "q") == "", "q is a jump to the letter q, not a quit, in a list"
    assert lib.picker is not None

    tui._handle_key(lib, "escape")
    assert lib.picker is None


# -- the browser ------------------------------------------------------------------------------


@pytest.fixture
def corpus(tmp_path):
    """A folder with a subfolder, a recording, and a file that is neither."""
    (tmp_path / "stationA").mkdir()
    (tmp_path / "clip.wav").write_bytes(b"RIFF")
    (tmp_path / "notes.txt").write_text("not a recording")
    (tmp_path / ".hidden").write_text("")
    return tmp_path


def test_the_browser_lists_folders_and_recordings_and_counts_the_rest(corpus):
    picker = tui.Picker("input", str(corpus))

    kinds = {label: kind for kind, label, _path, _note in picker.entries}
    assert "stationA" + os.sep in kinds and kinds["stationA" + os.sep] == "dir"
    assert kinds["clip.wav"] == "file"
    assert "notes.txt" not in kinds and ".hidden" not in kinds
    assert picker.hidden == 1


def test_the_browser_opens_on_the_row_that_keeps_the_current_folder(corpus):
    picker = tui.Picker("input", str(corpus))

    kind, _label, path, _note = picker.entries[picker.cursor]
    assert kind == "use" and path == str(corpus)


def test_choosing_a_folder_answers_the_row_that_opened_the_browser(corpus):
    lib = _library()
    tui._handle_key(lib, "i")
    lib.picker.open_directory(str(corpus))

    tui._handle_key(lib, "enter")

    assert lib.picker is None
    assert lib.form.input == str(corpus)


def test_a_single_recording_can_be_chosen_as_the_input(corpus):
    lib = _library()
    tui._handle_key(lib, "i")
    lib.picker.open_directory(str(corpus))
    while lib.picker.entries[lib.picker.cursor][1] != "clip.wav":
        tui._handle_key(lib, "down")

    tui._handle_key(lib, "enter")

    assert lib.form.input == str(corpus / "clip.wav")


def test_the_output_browser_offers_a_folder_that_does_not_exist_yet(corpus):
    lib = _library()
    tui._handle_key(lib, "o")
    lib.picker.open_directory(str(corpus))
    while lib.picker.entries[lib.picker.cursor][0] != "new":
        tui._handle_key(lib, "down")

    tui._handle_key(lib, "enter")
    assert lib.picker.naming, "naming a new folder is typing, and only that row starts it"
    for char in "scan-01":
        tui._handle_key(lib, char)
    tui._handle_key(lib, "enter")

    assert lib.form.out == str(corpus / "scan-01")
    assert lib.form.out_touched


def test_a_typed_name_that_is_a_whole_path_is_taken_as_one(corpus):
    picker = tui.Picker("output", str(corpus))
    picker.naming = True
    picker.buffer = "~/elsewhere/out"

    assert picker.named() == os.path.abspath(os.path.expanduser("~/elsewhere/out"))


def test_the_browser_only_offers_recordings_to_the_input_row(corpus):
    picker = tui.Picker("output", str(corpus))

    assert not any(kind == "file" for kind, *_rest in picker.entries)
    assert picker.hidden == 2, "the recording is counted with the rest, not offered"


def test_going_up_leaves_the_cursor_on_the_folder_just_left(corpus):
    picker = tui.Picker("input", str(corpus / "stationA"))

    picker.up()

    assert picker.directory == str(corpus)
    assert picker.entries[picker.cursor][2] == str(corpus / "stationA")


def test_a_folder_that_cannot_be_read_is_a_message_not_an_exception(tmp_path, monkeypatch):
    def refuse(_path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "listdir", refuse)
    picker = tui.Picker("input", str(tmp_path))

    assert "Permission denied" in picker.error
    assert picker.entries, "the row that leaves the folder has to survive the folder"


def test_typing_a_letter_jumps_to_the_next_row_starting_with_it(corpus):
    picker = tui.Picker("input", str(corpus))

    picker.jump("s")

    assert picker.entries[picker.cursor][1].startswith("stationA")


def test_the_browser_opens_at_the_nearest_folder_that_exists(corpus):
    picker = tui.Picker("output", str(corpus / "not" / "made" / "yet"))

    assert picker.directory == str(corpus)


# -- what the form will and will not run ------------------------------------------------------


def test_a_run_is_blocked_until_every_answer_is_one(tmp_path, corpus):
    form = tui.Form()
    model = _built()

    assert "choose what to scan" in form.blocking(model)
    form.set_input(str(corpus))
    assert "choose where to write" in form.blocking(model)
    form.set_output(str(corpus))
    assert "must not be the folder being scanned" in form.blocking(model)
    form.set_output(str(tmp_path / "out"))
    assert form.blocking(model) == ""


def test_a_model_that_cannot_run_here_blocks_the_run_with_its_own_reason():
    form = tui.Form(input="/audio", out="/out")

    assert "not packaged" in form.blocking(_built(runnable=False))


def test_an_input_with_no_recordings_says_so_before_the_run_is_attempted(tmp_path):
    form = tui.Form()
    form.set_input(str(tmp_path))

    assert form.note == "no .wav or .flac here"
    assert "no .wav or .flac" in form.blocking(_built())


def test_the_form_counts_what_it_is_pointed_at(corpus):
    form = tui.Form()
    form.set_input(str(corpus))

    assert form.note == "1 recording"


def test_the_screen_asks_for_one_worker_or_the_whole_machine():
    lib = _library()

    assert lib.options().workers == tui._PARALLEL_OFF
    lib.form.toggle_parallel()
    assert lib.options().workers == tui._PARALLEL_AUTO


def test_completion_extends_a_path_as_far_as_it_is_unambiguous(tmp_path):
    (tmp_path / "surveys-2026").mkdir()
    (tmp_path / "surveys-2027").mkdir()

    assert tui.complete_path(str(tmp_path / "surv")) == str(tmp_path / "surveys-202")
    assert tui.complete_path(str(tmp_path / "surveys-2026")) == str(tmp_path / "surveys-2026") + os.sep
    assert tui.complete_path(str(tmp_path / "nothing")) == str(tmp_path / "nothing")


def test_a_folder_too_big_to_count_is_not_read_as_an_empty_one(corpus, monkeypatch):
    """`siar-app lib` opens in whatever folder it was typed in, and that is often a home
    directory. Giving up on the walk must cost a note, not a run the form refuses to start."""
    monkeypatch.setattr(tui, "count_recordings", lambda *a, **k: (0, False))
    form = tui.Form(out="/out")
    form.set_input(str(corpus))

    assert "too big to count" in form.note
    assert "no .wav or .flac" not in form.blocking(_built())


def test_a_count_that_stopped_short_says_so_rather_than_rounding_it_off(corpus, monkeypatch):
    monkeypatch.setattr(tui, "count_recordings", lambda *a, **k: (10_000, False))
    form = tui.Form()
    form.set_input(str(corpus))

    assert form.note.startswith("10,000+ recordings")


def test_choosing_a_folder_reports_the_count_as_it_goes(corpus, monkeypatch):
    """The wiring the spinner rides on: the browser's choice reaches the walk's progress."""
    def walk(_source, on_progress=None, **_kw):
        on_progress(7)
        return 12, True

    monkeypatch.setattr(tui, "count_recordings", walk)
    lib = _library()
    seen = []
    lib.counting = seen.append
    lib.open_picker("input")
    lib.choose(str(corpus))

    assert seen == [7]
    assert lib.picker is None, "the browser closes before the counting starts"
    assert lib.form.count == 12


# -- moving a model between machines ------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """A workspace of this test's own, so an import here is not an import into the real one."""
    monkeypatch.setenv("SIAR_APP_HOME", str(tmp_path / "workspace"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _packaged(root):
    """A built model with a real package behind it, ready to be exported."""
    package = root / "siar_thing"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "slug = 'thing'\n"
        "class _S:\n"
        "    def scan(self, grid):\n"
        "        return []\n"
        "def algorithm(manifest=None):\n"
        "    return _S()\n"
    )
    return tui.Model(source="built", slug="thing", version="0.1.0", platform="source",
                     path=str(package), runnable=True, stamped_at=1_760_000_000,
                     detail={"target": "thing"}, programs=[_program()])


def test_a_downloaded_model_cannot_be_exported_and_the_screen_says_why(workspace):
    lib = _library(models=[_downloaded()])

    tui._handle_key(lib, "e")

    assert "licensed per machine" in lib.message
    assert lib.message_kind == "error"
    assert not list(workspace.glob("*.siarmodel")), "and nothing was written"


def test_exporting_a_built_model_writes_one_file_and_says_how_to_use_it(workspace):
    lib = _library(models=[_packaged(workspace / "built")])

    tui._handle_key(lib, "e")

    written = list(workspace.glob("*.siarmodel"))
    assert len(written) == 1
    assert "siar-app import" in lib.message, "the other half of the instruction travels with it"


def test_the_import_browser_offers_bundles_and_nothing_else(workspace):
    (workspace / "thing.siarmodel").write_bytes(b"x")
    (workspace / "clip.wav").write_bytes(b"RIFF")
    (workspace / "notes.txt").write_text("")

    picker = tui.Picker("bundle", str(workspace))
    labels = [label for _kind, label, _path, _note in picker.entries]

    assert "thing.siarmodel" in labels
    assert "clip.wav" not in labels and "notes.txt" not in labels
    assert not any(label.startswith("use this folder") for label in labels), \
        "there is no such thing as importing a directory"


def test_importing_from_the_screen_lands_the_model_and_selects_it(workspace):
    lib = _library(models=[_packaged(workspace / "built")])
    tui._handle_key(lib, "e")
    bundle = str(next(iter(workspace.glob("*.siarmodel"))))

    tui._handle_key(lib, "m")
    assert lib.picker is not None and lib.picker.kind == "bundle"
    lib.choose(bundle)

    assert lib.picker is None
    assert "imported thing" in lib.message
    assert lib.model() is not None and lib.model().imported, "the cursor lands on what arrived"


def test_a_bundle_that_is_not_one_is_a_message_and_not_a_traceback(workspace):
    (workspace / "junk.siarmodel").write_bytes(b"not a tarball at all")
    lib = _library(models=[_packaged(workspace / "built")])

    lib.take(str(workspace / "junk.siarmodel"))

    assert "not a readable model bundle" in lib.message
    assert lib.message_kind == "error"


# -- what a model is called ---------------------------------------------------------------------


def test_a_model_says_what_it_looks_for_as_well_as_what_it_is_called(workspace):
    """Once a model can be named anything, the listing has to say what it detects."""
    built = _packaged(workspace / "built")
    built.detail = {"target": "recall", "positive_tags": '["recall", "recall_nb"]'}
    bundle = tui.Model(source="downloaded", slug="all_structures", runnable=True,
                       detail={"shapes": ["sweep", "click"]})

    rows = {row[0]: row[1] for row in tui._model_rows([built, bundle])}

    assert rows["thing"] == "recall +1", "the target, and the tags that counted as it"
    assert rows["all_structures"] == "sweep +1", "a bundle declares its shapes in its manifest"


def test_renaming_an_imported_model_changes_what_it_is_called_and_nothing_else(workspace):
    lib = _library(models=[_packaged(workspace / "built")])
    tui._handle_key(lib, "e")
    lib.take(str(next(iter(workspace.glob("*.siarmodel")))))
    before = lib.model()

    tui._handle_key(lib, "n")
    assert lib.naming and lib.buffer == before.slug, "the current name is there to edit"
    for _ in before.slug:
        lib.backspace()
    for char in "porpoise":
        lib.type(char)
    tui._handle_key(lib, "enter")

    after = lib.model()
    assert not lib.naming
    assert after.slug == "porpoise" and after.imported
    assert after.looks_for == before.looks_for, "what it detects did not change"
    assert after.path == before.path, "and it is the same package on disk"


def test_the_renamed_model_is_the_one_a_run_finds_by_that_name(workspace):
    from siarapp.library import local_model

    lib = _library(models=[_packaged(workspace / "built")])
    tui._handle_key(lib, "e")
    lib.take(str(next(iter(workspace.glob("*.siarmodel")))))
    lib.naming, lib.buffer = True, "porpoise"
    lib.commit_name()

    assert local_model("porpoise") is not None
    assert local_model("thing") is None, "and not under the name it arrived with"


def test_a_model_built_here_is_renamed_where_it_lives(workspace):
    """siar-app reads siar-build's index and never writes it, so it says who can."""
    lib = _library(models=[_built()])

    tui._handle_key(lib, "n")

    assert not lib.naming
    assert "siar-build name" in lib.message
    assert lib.message_kind == "warn"


def test_a_downloaded_bundles_name_is_not_this_machines_to_change(workspace):
    lib = _library(models=[_downloaded()])

    tui._handle_key(lib, "n")

    assert not lib.naming and "the server's" in lib.message


def test_typing_a_name_does_not_trigger_the_screens_own_keys(workspace):
    """"q" in a name must be a letter, not the end of the session."""
    lib = _library(models=[_packaged(workspace / "built")])
    tui._handle_key(lib, "e")
    lib.take(str(next(iter(workspace.glob("*.siarmodel")))))
    tui._handle_key(lib, "n")
    lib.buffer = ""

    for char in "quiet":
        assert tui._handle_key(lib, char) == "", "no key typed into a name may act"
    assert lib.buffer == "quiet"
