# Vixen Intelligence c.2026
"""Getting the documentation in front of somebody who is not going to read a man page.

Two things ship with the package and both open in a browser with no network at all: the
quickstart deck under ``local_web/``, and this package's README.

The README is not duplicated into the package to make that work. ``pyproject.toml`` names it
as the project's long description, so a wheel already carries the whole of it in its metadata;
:func:`readme_markdown` reads it back from there. One copy, at the repository root, and no
build step keeping a second one in step with it.

Markdown is rendered here rather than by a dependency, for the same reason the PNG encoder and
the HTTP client are stdlib: this package is numpy, soundfile and nothing else. :func:`to_html`
is therefore not a Markdown implementation — it handles the constructs *this* README uses, and
passes anything it does not recognise through unharmed rather than mangling it.
"""
from __future__ import annotations

import html as _html
import os
import re
import sys
import tempfile
import webbrowser

__all__ = [
    "DocsError",
    "open_path",
    "quickstart_path",
    "readme_html",
    "readme_markdown",
    "to_html",
    "write_readme_page",
]


class DocsError(RuntimeError):
    """A document could not be found or written."""


# -- finding what shipped --------------------------------------------------------------------


def quickstart_path() -> str:
    """Absolute path to the packaged quickstart page.

    Returns:
        The path to ``local_web/quickstart.html`` inside the installed package.

    Raises:
        DocsError: If the file is missing, which means the wheel was built without its
            package data — worth saying plainly rather than opening a blank tab.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "local_web", "quickstart.html")
    if not os.path.isfile(path):
        raise DocsError(
            f"the quickstart page is not where it should be ({path}).\n"
            "  This install is missing its package data — reinstall with:\n"
            "      uv tool install --force --python 3.13 "
            "git+https://github.com/energy-master/siar.git"
        )
    return path


def readme_markdown() -> str:
    """The README text, from the source tree if there is one, else the wheel's metadata.

    The file wins when it exists, and the order matters. An editable install keeps serving the
    long description it was built with, so asking metadata first means a developer editing
    README.md reads back the version from whenever they last installed — stale documentation
    presented as current, which is worse than none.

    A real customer install has no repository beside it, and falls through to the metadata:
    ``pyproject.toml`` names the README as the project's long description, so every wheel
    already carries the whole of it and nothing extra had to be packaged.

    Returns:
        The raw Markdown.

    Raises:
        DocsError: If neither source has it.
    """
    # Development checkout: src/siarapp/docs.py -> three levels up is the repository root.
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(root, "README.md")
    if os.path.isfile(candidate):
        try:
            with open(candidate, encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            pass  # unreadable for some reason — the metadata copy is still worth trying

    try:
        from importlib.metadata import metadata

        md = metadata("siar-app")
        text = md.get_payload() if hasattr(md, "get_payload") else None
        if not text:
            text = md.get("Description") or ""
        if text and text.strip():
            return text
    except Exception:  # noqa: BLE001 - not installed, or metadata without a description
        pass

    raise DocsError(
        "could not find the README — this install carries no long description and there is "
        "no README.md beside the package. Try `siar-app quick-start` instead."
    )


# -- just enough Markdown --------------------------------------------------------------------

#: Inline rules, applied in order, to text that has already been HTML-escaped. Code spans are
#: taken first and parked, so that emphasis inside `--flag` or `**kwargs` is left alone.
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\)")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")


def _inline(text: str) -> str:
    """Render the inline constructs of one already-escaped line."""
    parked: list[str] = []

    def park(m: re.Match[str]) -> str:
        parked.append(m.group(1))
        return f"\x00{len(parked) - 1}\x00"

    text = re.sub(r"`([^`]+)`", park, text)
    text = _LINK.sub(r'<a href="\2">\1</a>', text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{parked[int(m.group(1))]}</code>", text)


def _slug(text: str) -> str:
    """A GitHub-style anchor, so the README's own internal links keep working."""
    plain = re.sub(r"<[^>]+>", "", text).lower()
    return re.sub(r"[^a-z0-9\s-]", "", plain).strip().replace(" ", "-")


def _row(line: str) -> list[str]:
    """Split one table row on unescaped pipes."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def to_html(md: str) -> str:
    """Render Markdown to an HTML fragment.

    Handles the constructs this README uses: ATX headings, fenced code, pipe tables,
    bullet and numbered lists, blockquotes, horizontal rules, paragraphs, and the inline
    set (links, bold, italic, code). Raw HTML blocks — the ``<details>`` sections — are
    passed straight through, which is what makes them keep working.

    Args:
        md: The Markdown source.

    Returns:
        An HTML fragment, to be dropped into :func:`readme_html`'s page shell.
    """
    out: list[str] = []
    lines = md.replace("\r\n", "\n").split("\n")
    i = 0
    para: list[str] = []
    list_stack: list[str] = []

    def flush_para() -> None:
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    def close_lists() -> None:
        while list_stack:
            out.append(f"</{list_stack.pop()}>")

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        # Fenced code. Taken first: everything inside is literal.
        fence = re.match(r"^\s*```+\s*([\w+-]*)\s*$", line)
        if fence:
            flush_para()
            close_lists()
            i += 1
            body: list[str] = []
            while i < len(lines) and not re.match(r"^\s*```+\s*$", lines[i]):
                body.append(lines[i])
                i += 1
            i += 1
            lang = fence.group(1)
            cls = f' class="lang-{_html.escape(lang)}"' if lang else ""
            out.append(f"<pre{cls}><code>" + _html.escape("\n".join(body)) + "</code></pre>")
            continue

        if not stripped:
            flush_para()
            close_lists()
            i += 1
            continue

        # Raw HTML block: <details>, <summary>, and their closers. Passed through as-is.
        if re.match(r"^\s*</?(details|summary|br|img|div|p|a)\b", line, re.I):
            flush_para()
            out.append(line)
            i += 1
            continue

        if re.match(r"^\s*(---+|\*\*\*+|___+)\s*$", line):
            flush_para()
            close_lists()
            out.append("<hr>")
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_para()
            close_lists()
            level = len(heading.group(1))
            text = _inline(_html.escape(heading.group(2)))
            out.append(f'<h{level} id="{_slug(text)}">{text}</h{level}>')
            i += 1
            continue

        # Pipe table: a header row, a separator of dashes, then body rows.
        if stripped.startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            flush_para()
            close_lists()
            head = _row(stripped)
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_row(lines[i]))
                i += 1
            cells = "".join(f"<th>{_inline(_html.escape(c))}</th>" for c in head)
            body_html = "".join(
                "<tr>" + "".join(f"<td>{_inline(_html.escape(c))}</td>" for c in r) + "</tr>"
                for r in rows
            )
            out.append(
                '<div class="scroll"><table><thead><tr>' + cells +
                "</tr></thead><tbody>" + body_html + "</tbody></table></div>"
            )
            continue

        if stripped.startswith("> "):
            flush_para()
            close_lists()
            quote: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + _inline(_html.escape(" ".join(quote))) + "</blockquote>")
            continue

        bullet = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)
        if bullet:
            flush_para()
            tag = "ul" if bullet.group(2) in "-*+" else "ol"
            if not list_stack:
                list_stack.append(tag)
                out.append(f"<{tag}>")
            out.append("<li>" + _inline(_html.escape(bullet.group(3))) + "</li>")
            i += 1
            continue

        # Lazy continuation: a wrapped bullet. Its second and third lines are not bullets
        # themselves, and closing the list on them would turn one three-line item into a
        # one-line item followed by an orphaned paragraph.
        if list_stack and out and out[-1].endswith("</li>"):
            out[-1] = out[-1][: -len("</li>")] + " " + _inline(_html.escape(stripped)) + "</li>"
            i += 1
            continue

        close_lists()
        para.append(_html.escape(stripped))
        i += 1

    flush_para()
    close_lists()
    return "\n".join(out)


#: The page shell. Same night palette as the quickstart deck, cut down to what prose needs, and
#: inline so the rendered README is one self-contained file in a temporary directory.
_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>siar-app — README</title>
<style>
:root {{
  --ground: #05090D; --surface: #0C1319; --ink: #E4EDEE; --ink-soft: #A6B8BE;
  --ink-faint: #63777E; --rule: #1B252C; --accent: #22A884; --accent-ink: #7AD151;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", "DejaVu Sans Mono", Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  --serif: Charter, "Bitstream Charter", "Iowan Old Style", Georgia, serif;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--ground); color: var(--ink);
  font-family: var(--sans); font-size: 1rem; line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}}
main {{ max-width: 52rem; margin: 0 auto; padding: 3rem 1.5rem 6rem; }}
.banner {{
  font-family: var(--mono); font-size: .74rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 2rem; padding-bottom: .75rem;
  border-bottom: 1px solid var(--rule);
}}
h1, h2, h3, h4 {{ font-family: var(--serif); font-weight: 600; line-height: 1.2; text-wrap: balance; }}
h1 {{ font-size: 2.25rem; margin: 0 0 1rem; letter-spacing: -.015em; }}
h2 {{ font-size: 1.55rem; margin: 3rem 0 .85rem; padding-top: 1.25rem; border-top: 1px solid var(--rule); }}
h3 {{ font-size: 1.2rem; margin: 2rem 0 .6rem; color: var(--accent-ink); font-family: var(--mono); }}
h4 {{ font-size: 1.02rem; margin: 1.5rem 0 .5rem; }}
p, li {{ color: var(--ink-soft); }}
a {{ color: var(--accent-ink); text-decoration-color: rgba(122, 209, 81, .4); }}
a:hover {{ text-decoration-color: currentColor; }}
code {{ font-family: var(--mono); font-size: .88em; color: var(--accent-ink); }}
pre {{
  background: #03070A; border: 1px solid var(--rule); border-radius: 6px;
  padding: .9rem 1.1rem; overflow-x: auto; line-height: 1.55;
}}
pre code {{ color: #CFDCDF; font-size: .84rem; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }}
th, td {{ text-align: left; padding: .5rem .7rem; border-bottom: 1px solid var(--rule); vertical-align: top; }}
th {{ font-family: var(--mono); font-size: .74rem; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-faint); }}
td {{ color: var(--ink-soft); }}
blockquote {{ margin: 1rem 0; padding-left: 1rem; border-left: 2px solid var(--accent); color: var(--ink-soft); }}
hr {{ border: 0; border-top: 1px solid var(--rule); margin: 2.5rem 0; }}
ul, ol {{ padding-left: 1.35rem; }}
li {{ margin: .3rem 0; }}
details {{ margin: 1rem 0; padding: .75rem 1rem; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }}
summary {{ cursor: pointer; color: var(--accent-ink); font-family: var(--mono); font-size: .86rem; }}
</style>
</head>
<body>
<main>
<p class="banner">SIaR &middot; siar-app manual &middot; rendered offline</p>
{body}
</main>
</body>
</html>
"""


def readme_html() -> str:
    """The README as a complete, self-contained HTML page."""
    return _PAGE.format(body=to_html(readme_markdown()))


def write_readme_page() -> str:
    """Render the README to a temporary HTML file and return its path.

    A temporary file rather than the workspace: this is a view of a document, regenerated
    every time, and leaving stale renders in ``~/.siar-app`` would mean an upgraded CLI
    still showing yesterday's manual.

    Raises:
        DocsError: If the file cannot be written.
    """
    try:
        directory = tempfile.mkdtemp(prefix="siar-app-readme-")
        path = os.path.join(directory, "siar-app-readme.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(readme_html())
        return path
    except OSError as e:
        raise DocsError(f"could not write the rendered README: {e}") from None


# -- opening it ------------------------------------------------------------------------------


def open_path(path: str) -> bool:
    """Open a local file in the user's browser.

    Args:
        path: An absolute path to an HTML file.

    Returns:
        True if a browser was launched. False on a machine with no browser to launch — a
        headless server, a container, an ssh session without X — where the caller should
        print the path instead of pretending it worked.
    """
    url = "file://" + os.path.abspath(path)
    # webbrowser falls back to a text-mode browser under $BROWSER; on a headless box it
    # simply finds nothing and returns False. Its stderr chatter is not ours to show.
    try:
        opened = webbrowser.open(url)
    except Exception:  # noqa: BLE001 - a broken BROWSER setting is not a crash
        return False
    if not opened and sys.platform == "darwin":
        # webbrowser occasionally reports False on macOS even having handed off to `open`.
        return os.system(f'open "{url}" >/dev/null 2>&1') == 0
    return bool(opened)
