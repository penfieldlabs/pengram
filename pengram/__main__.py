# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Entry point for ``python -m pengram``.

All CLI logic lives in :mod:`pengram.cli`; this module just wires up
``main()`` and the ``if __name__`` guard.
"""

from __future__ import annotations

import logging
import sys

from ._ui import error as _ui_error
from .cli import _rewrite_default_run, build_parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns an exit code (0 success, 1 user error, 2 internal)."""
    raw = list(sys.argv[1:]) if argv is None else list(argv)
    raw = _rewrite_default_run(raw)

    parser = build_parser()
    args = parser.parse_args(raw)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:  # pragma: no cover
        return 2
    except Exception as exc:
        logging.getLogger("pengram").exception("Unexpected error")
        _ui_error(f"internal error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
