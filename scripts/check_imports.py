#!/usr/bin/env python3
"""Import every module in the package and report failures.

The pipeline cannot be *run* without the (private) dataset, but it can all be
*imported*. This is the check that the package layout and the import rewrite are
sound; CI runs it, and it is the fastest way to catch a bad move locally.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback

import vertpois


def main() -> int:
    """Import every submodule of :mod:`vertpois`. Returns a process exit code."""
    failures: list[tuple[str, str]] = []
    names = [name for _, name, _ in pkgutil.walk_packages(vertpois.__path__, prefix="vertpois.")]

    for name in sorted(names):
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - we want to report every failure, not the first
            failures.append((name, traceback.format_exc()))
        else:
            print(f"  ok   {name}")

    if failures:
        print(f"\n{len(failures)} module(s) failed to import:\n")
        for name, tb in failures:
            print(f"--- {name} ---\n{tb}")
        return 1
    print(f"\nAll {len(names)} modules imported cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
