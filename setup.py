# Vixen Intelligence c.2026
"""Cython build for distribution.

Development:  ``pip install -e .``   — editable install, uses .py source directly.
Distribution: ``pip wheel --no-build-isolation .`` — compiles every module to a .so extension
              and strips .py source from the wheel.  The installed package contains only compiled
              C extensions, ``__init__.py`` stubs (package markers), and static assets.

The ``NoPySources`` command keeps ``__init__.py`` files (Python needs them to recognise
subpackages) and lets ``build_ext`` handle everything else.  Package data (web/static/) is
unaffected — ``build_py`` copies it regardless.
"""
from __future__ import annotations

import os

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_py import build_py as _build_py

SRC = os.path.join("src", "siar")


def _extensions() -> list[Extension]:
    """Discover every .py module under ``src/siar/`` except ``__init__.py`` stubs.

    Returns:
        One :class:`Extension` per module, ready for :func:`cythonize`.
    """
    exts: list[Extension] = []
    for root, _dirs, files in os.walk(SRC):
        for fn in sorted(files):
            if not fn.endswith(".py") or fn == "__init__.py":
                continue
            path = os.path.join(root, fn)
            # src/siar/auth.py -> siar.auth
            dotted = path.removeprefix(f"src{os.sep}").replace(os.sep, ".").removesuffix(".py")
            exts.append(Extension(dotted, [path]))
    return exts


class NoPySources(_build_py):
    """Custom ``build_py`` that excludes compiled modules and generated C from the wheel.

    Only ``__init__.py`` stubs survive — they are the package markers Python's import machinery
    needs.  Every other ``.py`` file has a corresponding ``.so`` built by ``build_ext``, so
    shipping the source would be redundant *and* would defeat the point of compiling.

    Cython-generated ``.c`` files are also excluded: they are human-readable transpiled source
    and would undermine the obfuscation if shipped.

    Package data (``web/static/**/*``) is still copied by the inherited ``build_package_data``.
    """

    def find_package_modules(
        self,
        package: str,
        package_dir: str,
    ) -> list[tuple[str, str, str]]:
        """Keep only ``__init__`` modules; drop everything that Cython compiled."""
        modules = super().find_package_modules(package, package_dir)
        return [(pkg, mod, path) for pkg, mod, path in modules if mod == "__init__"]

    def find_data_files(
        self,
        package: str,
        src_dir: str,
    ) -> list[str]:
        """Exclude Cython-generated ``.c`` files from package data."""
        files = super().find_data_files(package, src_dir)
        return [f for f in files if not f.endswith(".c")]


setup(
    ext_modules=cythonize(
        _extensions(),
        language_level="3str",
        compiler_directives={
            # Do not embed function signatures in the docstring — they reveal the API surface
            # and make reverse-engineering easier.
            "embedsignature": False,
        },
    ),
    cmdclass={"build_py": NoPySources},
)
