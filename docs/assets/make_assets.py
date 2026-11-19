"""Draw integrix's icon and logo: ``icon.svg`` and ``logo-{light,dark}.svg``.

The icon is the running integral: trapezoid strips under an integrand f, and
the curve F(x) = ∫ f that climbs through one dot per node, which is what a
scan returns. A sum is the curve's last dot and a march is the same climb
with an integrand that depends on the solution, so one picture holds all
three operations. Standard library plus loguru; the tile and wordmark follow
the other repos of the stack (gaussx, functax, argminax).
Run from the repository root::

    uv run --no-project --with loguru python docs/assets/make_assets.py
"""

import itertools
import math
from pathlib import Path

from loguru import logger


HERE = Path(__file__).resolve().parent

NAME = "integrix"
WORDMARK = ("integri", "x")
LOGO_WIDTH = 540
# Amber: no other repo of the stack uses a warm tile.
GRADIENT = ("#F59E0B", "#B45309")
THEMES = {
    "light": {"ink": "#1F2937", "accent": "#B45309"},
    "dark": {"ink": "#E5E7EB", "accent": "#FBBF24"},
}
SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
WHITE = "#FFFFFF"

# the plot box inside the tile: left, right, baseline and top
LEFT, RIGHT, BASE, TOP = 22.0, 106.0, 98.0, 30.0
NODES = 7  # the nodes of the trapezoid strips, ends included
SAMPLES = 400  # the points the running integral is drawn from


def integrand(u: float) -> float:
    """A skewed bump on [0, 1], the f the strips sample."""
    main = 0.75 * math.exp(-(((u - 0.38) / 0.22) ** 2))
    shoulder = 0.25 * math.exp(-(((u - 0.8) / 0.12) ** 2))
    return 0.18 + main + shoulder


def point(u: float, v: float) -> tuple[float, float]:
    """The tile coordinates of (u, v) in the unit square of the plot box."""
    return LEFT + u * (RIGHT - LEFT), BASE - v * (BASE - TOP)


def svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{NAME}">\n'
        f"<title>{NAME}</title>\n{body}\n</svg>\n"
    )


def tile(gradient_id: str) -> str:
    start, stop = GRADIENT
    return (
        f'<defs><linearGradient id="{gradient_id}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{start}"/>'
        f'<stop offset="1" stop-color="{stop}"/></linearGradient></defs>'
        '<rect x="4" y="4" width="120" height="120" rx="26" '
        f'fill="url(#{gradient_id})"/>'
    )


def strips(nodes: list[float]) -> str:
    """One trapezoid per pair of neighbouring nodes, from the baseline to f."""
    body = ""
    for left, right in itertools.pairwise(nodes):
        (x_left, y_left) = point(left, integrand(left))
        (x_right, y_right) = point(right, integrand(right))
        body += (
            f'<path d="M{x_left:.1f} {BASE} L{x_left:.1f} {y_left:.1f} '
            f'L{x_right:.1f} {y_right:.1f} L{x_right:.1f} {BASE} Z" '
            f'fill="{WHITE}" fill-opacity="0.2" stroke="{WHITE}" '
            'stroke-opacity="0.6" stroke-width="1.6" stroke-linejoin="round"/>'
        )
    return body


def running_integral() -> list[tuple[float, float]]:
    """F(u) = ∫₀ᵘ f, by a left Riemann sum, scaled to 0.95 of the box."""
    total = sum(integrand(i / SAMPLES) for i in range(SAMPLES)) / SAMPLES
    partial, points = 0.0, []
    for i in range(SAMPLES + 1):
        u = i / SAMPLES
        points.append(point(u, 0.95 * partial / total))
        partial += integrand(u) / SAMPLES
    return points


def glyph() -> str:
    """The strips under f, the running integral over them and a dot per node."""
    nodes = [i / (NODES - 1) for i in range(NODES)]
    climb = running_integral()
    commands = " L".join(f"{x:.1f} {y:.1f}" for x, y in climb[::4])
    body = strips(nodes)
    body += (
        f'<path d="M{commands}" fill="none" stroke="{WHITE}" stroke-width="5" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )
    for u in nodes:
        x, _ = point(u, 0.0)
        _, y = climb[round(u * SAMPLES)]
        body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{WHITE}"/>'
    return body + (
        f'<line x1="{LEFT}" y1="{BASE}" x2="{RIGHT}" y2="{BASE}" stroke="{WHITE}" '
        'stroke-width="3" stroke-linecap="round" stroke-opacity="0.8"/>'
    )


def icon() -> str:
    return svg(128, 128, tile(f"{NAME}-icon") + glyph())


def logo(theme: str) -> str:
    colours = THEMES[theme]
    head, tail = WORDMARK
    mark = f'<g transform="translate(6 6)">{tile(f"{NAME}-{theme}")}{glyph()}</g>'
    wordmark = (
        f'<text x="154" y="92" font-family="{SANS}" font-size="72" font-weight="700" '
        f'letter-spacing="-1.5" fill="{colours["ink"]}">{head}'
        f'<tspan fill="{colours["accent"]}">{tail}</tspan></text>'
    )
    return svg(LOGO_WIDTH, 140, mark + wordmark)


def main() -> None:
    outputs = {"icon.svg": icon()}
    for theme in THEMES:
        outputs[f"logo-{theme}.svg"] = logo(theme)
    for filename, content in outputs.items():
        (HERE / filename).write_text(content)
        logger.info("wrote {}", HERE / filename)


if __name__ == "__main__":
    main()
