#!/usr/bin/env python3
"""Pre-commit guard against committing machine-specific paths or patient identifiers.

This repository was extracted from a research codebase that contained absolute NAS
paths, a private clinical cohort's subject IDs, and real scan dates. This hook exists
so that class of leak cannot silently return.

Exits non-zero and prints ``file:line`` for every offending line in the staged files.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Patterns that must never appear in committed source.
PATTERNS: dict[str, re.Pattern[str]] = {
    "absolute machine path": re.compile(r"(?<![\w.])/(?:DATA|home|media|mnt)/"),
    "private cohort subject id": re.compile(r"\bWS-\d{2}\b"),
    "scan session date": re.compile(r"\bses-(?:19|20)\d{6}\b"),
}

#: Files allowed to mention the patterns, because explaining them is their job.
ALLOWLIST = {
    "scripts/check_no_private_data.py",
    "config/paths.example.yaml",
    "CHANGES.md",
    "CONTRIBUTING.md",
}


def check_file(path: Path) -> list[str]:
    """Return a list of human-readable findings for one file."""
    if path.as_posix() in ALLOWLIST:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    findings = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if "noqa: private-data" in line:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(line):
                findings.append(f"{path}:{lineno}: {label}: {line.strip()[:120]}")
    return findings


def main(argv: list[str]) -> int:
    """Check every path given on the command line. Returns a process exit code."""
    findings: list[str] = []
    for name in argv:
        findings.extend(check_file(Path(name)))

    if findings:
        print("Refusing to commit: machine-specific or private data found.\n")
        for finding in findings:
            print(f"  {finding}")
        print(
            "\nUse the path config (verpex.paths.get_path) instead of an absolute path.\n"
            "If a match is genuinely fine, append '# noqa: private-data' to that line."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
