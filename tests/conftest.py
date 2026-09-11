"""Shared test fixture: make the repo root importable for ``from beacon import ...``.

Every test module previously inserted ROOT into sys.path itself, which forced
a ``# noqa: E402`` on the follow-up beacon imports. Centralising the insert
here means the beacon imports read as normal top-of-file imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
