"""The code on the theory pages runs, and prints what the pages say it prints.

Every number quoted in `docs/theory/` comes from a ``python`` block on the page.
A block followed by a ``text`` block must print exactly that text. Each page runs
in its own interpreter, top to bottom in one namespace, so a page that turns on
64-bit mode does so for itself only.
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


THEORY = Path(__file__).resolve().parent.parent / "docs" / "theory"
PAGES = sorted(THEORY.glob("[0-9][0-9]_*.md"))
# A ``python`` block, then (after blank lines only) an optional ``text`` block.
BLOCK = re.compile(
    r"^```python\n(?P<code>.*?)^```\n(?:\s*^```text\n(?P<output>.*?)^```\n)?",
    re.DOTALL | re.MULTILINE,
)


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
            exec(compile(code, "<theory>", "exec"), namespace)
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
            assert actual == claimed, f"{page.name}, block {index + 1}"
