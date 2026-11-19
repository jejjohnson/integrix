"""The code on the theory and guide pages runs, and prints what the pages say.

Every number quoted in `docs/theory/` and `docs/guide/` comes from a ``python``
block on the page.
A block followed by a ``text`` block must print exactly that text, except that a
number the page claims at or below 1e-12 in magnitude is round-off, and matches
any printed number at or below the same bound: its digits depend on the CPU, the
BLAS and the XLA version, not on the code (integrix#83). Each page runs in its own
interpreter, top to bottom in one namespace, so a page that turns on 64-bit mode
does so for itself only.
It runs in the slow tier, on main, nightly and on PRs labelled ``ci:full``.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


DOCS = Path(__file__).resolve().parent.parent / "docs"
PAGES = sorted(DOCS.glob("theory/[0-9][0-9]_*.md")) + sorted(DOCS.glob("guide/*.md"))
# A ``python`` block, then (after blank lines only) an optional ``text`` block.
BLOCK = re.compile(
    r"^```python\n(?P<code>.*?)^```\n(?:\s*^```text\n(?P<output>.*?)^```\n)?",
    re.DOTALL | re.MULTILINE,
)


# A signed decimal or scientific number, as the pages print them.
NUMBER = re.compile(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
# Below this, a printed number is round-off and its digits are not the code's.
ROUND_OFF = 1e-12


def matches_page(actual: str, claimed: str) -> bool:
    """Whether printed text matches a page's claim, up to round-off digits."""
    claimed_numbers = NUMBER.findall(claimed)
    actual_numbers = NUMBER.findall(actual)
    # the text between the numbers must agree exactly
    if NUMBER.split(claimed) != NUMBER.split(actual):
        return False
    if len(claimed_numbers) != len(actual_numbers):
        return False
    for want, got in zip(claimed_numbers, actual_numbers, strict=True):
        if want == got:
            continue
        if abs(float(want)) > ROUND_OFF or abs(float(got)) > ROUND_OFF:
            return False
    return True


def page_blocks(page: Path) -> list[tuple[str, str | None]]:
    """The page's ``python`` blocks, each with the output it claims, if any."""
    text = page.read_text(encoding="utf-8")
    return [(m["code"], m["output"]) for m in BLOCK.finditer(text)]


def run_blocks(codes: list[str]) -> list[str]:
    """Run the blocks in one namespace and return what each one printed."""
    namespace: dict[str, object] = {"__name__": "__theory__"}
    printed = []
    for code in codes:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            # dont_inherit: this module's `from __future__ import annotations` must
            # not turn a page's annotations into strings (eqx.Module fields need them)
            exec(compile(code, "<theory>", "exec", dont_inherit=True), namespace)
        printed.append(buffer.getvalue())
    return printed


# Each page runs JAX in a fresh interpreter (4–26 s), so this is the slow tier.
@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES, ids=[page.stem for page in PAGES])
def test_page_prints_what_it_says(page: Path) -> None:
    blocks = page_blocks(page)
    assert blocks, f"{page.name} has no python blocks"
    codes = [code for code, _ in blocks]
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_theory_pages import run_blocks\n"
        "print(json.dumps(run_blocks(json.loads(sys.stdin.read()))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(codes),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    printed = json.loads(result.stdout.splitlines()[-1])
    for index, ((_, claimed), actual) in enumerate(zip(blocks, printed, strict=True)):
        if claimed is not None:
            where = f"{page.name}, block {index + 1}"
            assert matches_page(actual, claimed), (
                f"{where}:\nclaimed:\n{claimed}\nprinted:\n{actual}"
            )


@pytest.mark.parametrize(
    ("actual", "claimed", "expected"),
    [
        # round-off digits may differ, and so may the sign of round-off
        ("error 2.6e-14, ok\n", "error 2.7e-14, ok\n", True),
        ("areas -2.4e-15\n", "areas +2.2e-16\n", True),
        # a number above the bound must match exactly
        ("error 4.4e-11\n", "error 4.5e-11\n", False),
        ("N = 12\n", "N = 11\n", False),
        # round-off on the page, a real error printed
        ("error 3.0e-05\n", "error 2.7e-14\n", False),
        # the words around the numbers must match exactly
        ("err 2.6e-14\n", "error 2.7e-14\n", False),
        ("1.0 2.0\n", "1.0 2.0\n", True),
    ],
)
def test_round_off_digits_are_not_compared(
    actual: str, claimed: str, *, expected: bool
) -> None:
    assert matches_page(actual, claimed) is expected
