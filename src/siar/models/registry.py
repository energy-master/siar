# Vixen Intelligence c.2026
"""The detector registry.

A plain dict and a decorator. No entry-point scanning, no ``importlib`` globbing of a plugin
directory, no ``eval``-based class lookup — a detector is registered because a module explicitly
imported it (see :mod:`siar.models`), and you can find every one of them by reading that file.

Adding a new anomaly solution is:

1. write ``siar/models/<name>.py`` implementing :class:`~siar.models.base.Detector`;
2. decorate it with ``@register_detector("<name>")``;
3. add one import line to ``siar/models/__init__.py``.

If step 3 ever requires touching anything else, the seam is wrong.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

__all__ = ["DETECTORS", "get_detector", "list_detectors", "register_detector"]

#: Registered detectors, keyed by their short name (``"conv_ae"``), not their ``format``.
DETECTORS: dict[str, type] = {}

#: Model ``format`` string -> detector class, so a model JSON can be loaded without being told
#: which class made it.
_BY_FORMAT: dict[str, type] = {}

_T = TypeVar("_T", bound=type)


def register_detector(name: str) -> Callable[[_T], _T]:
    """Register a detector class under a short name.

    Args:
        name: The key users type, e.g. ``"conv_ae"``.

    Returns:
        A class decorator.

    Raises:
        ValueError: If ``name`` or the class's ``format`` is already registered.
    """

    def decorate(cls: _T) -> _T:
        if name in DETECTORS:
            raise ValueError(f"detector {name!r} is already registered")
        fmt = getattr(cls, "format", None)
        if not fmt:
            raise ValueError(f"detector {name!r} has no `format` class attribute")
        if fmt in _BY_FORMAT:
            raise ValueError(f"model format {fmt!r} is already registered")
        DETECTORS[name] = cls
        _BY_FORMAT[fmt] = cls
        return cls

    return decorate


def get_detector(name: str) -> type:
    """Look up a detector class by its short name.

    Args:
        name: e.g. ``"conv_ae"``.

    Returns:
        The detector class.

    Raises:
        KeyError: If no such detector is registered.
    """
    try:
        return DETECTORS[name]
    except KeyError:
        known = ", ".join(sorted(DETECTORS)) or "(none)"
        raise KeyError(f"unknown detector {name!r}; registered: {known}") from None


def detector_for_format(fmt: str) -> type:
    """Look up the detector class that produced a model of this ``format``.

    Args:
        fmt: The ``format`` field of a model descriptor, e.g. ``"siar-conv-ae-v1"``.

    Returns:
        The detector class.

    Raises:
        KeyError: If the format is not recognised.
    """
    try:
        return _BY_FORMAT[fmt]
    except KeyError:
        known = ", ".join(sorted(_BY_FORMAT)) or "(none)"
        raise KeyError(f"unknown model format {fmt!r}; registered: {known}") from None


def list_detectors() -> list[str]:
    """Every registered detector's short name, sorted."""
    return sorted(DETECTORS)
