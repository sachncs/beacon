"""Generate ``docs/swa-mask.svg`` from ``beacon.patch.mask``.

The diagram shows the SWA-with-sinks attend-set for a representative
``(window=4, sink=2)`` configuration on a 16x16 grid. Black cells attend,
light cells mask out. The two leftmost columns are the sink (highlighted
in amber); the trailing diagonal stripe is the sliding window.

Run from the repo root::

    python docs/make_mask.py

The SVG is checked in alongside this script so the README renders without
re-running it. The script stays in sync with ``beacon.patch.mask``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beacon.patch import mask  # noqa: E402

SIZE = 16
WINDOW = 4
SINK = 2
OUT = Path(__file__).resolve().parent / "swa-mask.svg"

ATTEND = "#1f2937"
MASK = "#f3f4f6"
SINK_BAND = "#fde68a"
WINDOW_HIGHLIGHT = "#bfdbfe"
STROKE = "#9ca3af"


def main() -> None:
    m = mask(SIZE, SIZE, window=WINDOW, sink=SINK).view(SIZE, SIZE)
    finite = m.isfinite().tolist()

    cell = 28
    pad = 16
    side = pad * 2 + cell * SIZE

    rows: list[str] = []
    rows.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side} {side}" '
        f'width="{side}" height="{side}" role="img" aria-label="SWA-with-sinks mask '
        f'(window={WINDOW}, sink={SINK})">'
    )
    rows.append(
        f'<rect width="{side}" height="{side}" fill="#ffffff"/>'
    )

    rows.append(
        f'<rect x="{pad}" y="{pad}" width="{SINK * cell}" height="{SIZE * cell}" '
        f'fill="{SINK_BAND}" opacity="0.55"/>'
    )

    for i in range(SIZE):
        for j in range(SIZE):
            x = pad + j * cell
            y = pad + i * cell
            fill = ATTEND if finite[i][j] else MASK
            rows.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{fill}" stroke="{STROKE}" stroke-width="0.5"/>'
            )

    rows.append(
        f'<rect x="{pad + SINK * cell}" y="{pad}" width="{(SIZE - SINK) * cell}" '
        f'height="{SIZE * cell}" fill="{WINDOW_HIGHLIGHT}" opacity="0.25" '
        f'pointer-events="none"/>'
    )

    rows.append(
        f'<text x="{pad}" y="{pad - 4}" font-family="ui-sans-serif, system-ui" '
        f'font-size="11" fill="#374151">query ↓ / key →   '
        f'sink={SINK}  window={WINDOW}</text>'
    )

    rows.append("</svg>")

    OUT.write_text("\n".join(rows))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()