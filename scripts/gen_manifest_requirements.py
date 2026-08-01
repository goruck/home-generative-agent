#!/usr/bin/env python3
"""Generate runtime requirements from manifest.json."""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path("custom_components/home_generative_agent/manifest.json")
OUTFILE = Path("requirements_runtime_manifest.txt")


def main() -> None:
    """Generate runtime requirements from manifest.json."""
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reqs: list[str] = data.get("requirements", [])

    # Write one requirement per line, stable output
    OUTFILE.write_text("\n".join(reqs) + ("\n" if reqs else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
