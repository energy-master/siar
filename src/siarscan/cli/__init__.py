# Vixen Intelligence c.2026
"""The command line: the argparse tree, the subcommand handlers, and the table renderer."""
from __future__ import annotations

__all__ = ["main"]


def main(argv=None) -> int:
    """Run the CLI. Imported lazily so ``import siarscan.cli`` stays cheap."""
    from siarscan.cli.main import main as _main

    return _main(argv)
