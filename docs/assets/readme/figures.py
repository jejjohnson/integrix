"""Draw the README's gallery: specimens of what integrix computes, on toy data.

Each figure calls integrix directly, on an integrand with a closed form, so
every curve is a measured error, not an expected one. The numbers the README
quotes under a figure are logged by the run that drew it. Each figure has a
title that says its idea and a subtitle that names the calls behind it. Run
from the repo root, all figures or the ones named:

    uv run --group dev --with matplotlib python docs/assets/readme/figures.py
    uv run --group dev --with matplotlib python docs/assets/readme/figures.py \\
        --only convergence
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from cyclopts import App
from functax import quad
from loguru import logger
from scipy.special import erf

from integrix import rule


# The errors below fall to the round-off of float64, so every figure needs it.
jax.config.update("jax_enable_x64", True)

OUT = Path(__file__).parent
app = App(help="Draw the README's gallery of integrix specimens.")
# Errors are drawn at no less than this, so an exact result stays on the plot.
FLOOR = 1e-16
# One colour per method, the same in every figure.
COLOURS = {"global": "#B45309", "local": "#2563EB", "reference": "0.35"}


def save(fig: plt.Figure, name: str) -> None:
    """Write a figure next to this script at 150 dpi."""
    fig.savefig(OUT / name, dpi=150, facecolor="white")
    plt.close(fig)
    logger.info("wrote {}", name)


def headed(
    title: str, subtitle: str, size: tuple[float, float]
) -> tuple[plt.Figure, float]:
    """A figure with a bold title and a muted subtitle in a 0.75-inch band on top.

    Returns the figure and the top of the space below the band, as a fraction of
    the figure height, for the caller's ``fig.subplots_adjust``.
    """
    fig = plt.figure(figsize=size, facecolor="white")
    height = size[1]
    fig.text(0.012, 1 - 0.1 / height, title, fontsize=12, fontweight="bold", va="top")
    fig.text(
        0.012,
        1 - 0.36 / height,
        subtitle,
        fontsize=9,
        color="0.35",
        va="top",
        linespacing=1.35,
    )
    return fig, 1 - 0.75 / height


def slope(counts: np.ndarray, errors: np.ndarray) -> float:
    """The least-squares slope of log error against log N."""
    fit = np.polyfit(np.log(counts), np.log(errors), 1)
    return float(fit[0])


# ---------------------------------------------------------------- convergence


def band(mu: jnp.ndarray) -> jnp.ndarray:
    """f_band(μ) = exp(−μ² / (2 · 0.3²)), a band of tropical moisture."""
    return jnp.exp(-(mu**2) / (2.0 * 0.3**2))


def ice(mu: jnp.ndarray) -> jnp.ndarray:
    """f_ice(μ) = max(0, |μ| − 0.8), a sea-ice fraction with a kink at each edge."""
    return jnp.maximum(0.0, jnp.abs(mu) - 0.8)


def gauss_legendre(n_nodes: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """functax's Gauss–Legendre nodes and weights on [−1, 1]."""
    nodes, weights = quad.gauss_legendre(n_nodes)
    return jnp.asarray(nodes), jnp.asarray(weights)


def trapezoid(n_nodes: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """functax's trapezoid rule on equally spaced nodes in [−1, 1]."""
    nodes = jnp.linspace(-1.0, 1.0, n_nodes)
    return nodes, quad.trapezoid(nodes).weights


def convergence_figure() -> None:
    """The error of a global and a local rule against N, smooth and kinked."""
    # ∫ f_band dμ = 0.3 √(2π) erf(1 / (0.3 √2)), and ∫ f_ice dμ = 2 · 0.2² / 2
    exact = {
        "band": 0.3 * np.sqrt(2.0 * np.pi) * erf(1.0 / (0.3 * np.sqrt(2.0))),
        "ice": 0.04,
    }
    fields = {"band": band, "ice": ice}
    rules = {
        "Gauss–Legendre (global)": (gauss_legendre, COLOURS["global"]),
        "trapezoid (local)": (trapezoid, COLOURS["local"]),
    }
    counts = np.unique(np.round(np.geomspace(3, 256, 40)).astype(int))
    errors = {}
    for field_name, field in fields.items():
        for rule_name, (make_rule, _) in rules.items():
            sweep = []
            for n_nodes in counts:
                nodes, weights = make_rule(int(n_nodes))
                # Î_N = ∑ₙ wₙ f(μₙ)
                estimate = rule.integrate(weights, field(nodes)).value
                sweep.append(abs(float(estimate) - exact[field_name]))
            errors[field_name, rule_name] = np.maximum(np.array(sweep), FLOOR)

    # --- the numbers the README quotes ---
    smooth_global = errors["band", "Gauss–Legendre (global)"]
    reached = int(counts[np.argmax(smooth_global < 1e-14)])
    logger.info(
        "convergence: Gauss–Legendre on the band below 1e-14 at N = {}", reached
    )
    for field_name in fields:
        for rule_name in rules:
            sweep = errors[field_name, rule_name]
            # fit only above round-off, and leave out the node counts where a
            # node falls on the kink and the rule is exact
            kept = (counts >= 16) & (sweep > 1e-13)
            if kept.sum() < 5:
                continue
            fitted = slope(counts[kept], sweep[kept])
            logger.info(
                "convergence: {} on {}, slope {:.2f} over {} node counts",
                rule_name,
                field_name,
                fitted,
                int(kept.sum()),
            )

    fig, top = headed(
        "integrix: a global rule is spectral, until the integrand has a kink",
        "rule.integrate(weights, f(μ)) over μ = sin φ ∈ [−1, 1], with functax's "
        "Gauss–Legendre and trapezoid weights, against the exact integral.\n"
        "Left, a band of moisture exp(−μ² / (2 · 0.3²)); right, a sea-ice edge "
        "max(0, |μ| − 0.8). Errors below 10⁻¹⁶ are drawn at 10⁻¹⁶.",
        (10, 4.0),
    )
    axes = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=top - 0.07, wspace=0.08)
    titles = {"band": "smooth: a band of moisture", "ice": "kinked: a sea-ice edge"}
    for ax, field_name in zip(axes, fields, strict=True):
        for rule_name, (_, colour) in rules.items():
            ax.loglog(
                counts,
                errors[field_name, rule_name],
                "o-",
                color=colour,
                markersize=3,
                linewidth=1.4,
                label=rule_name,
            )
        reference = 2.0 * counts.astype(float) ** -2.0
        ax.loglog(
            counts,
            reference,
            "--",
            color=COLOURS["reference"],
            linewidth=1,
            label="N⁻²",
        )
        ax.set_title(titles[field_name], fontsize=10)
        ax.set_xlabel("number of nodes N", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="major", linewidth=0.3, alpha=0.5)
    axes[0].set_ylabel(r"error $|I - \hat{I}_N|$", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    save(fig, "convergence.png")


FIGURES = {"convergence": convergence_figure}


@app.default
def main(*, only: list[str] | None = None) -> None:
    """Draw every figure of the gallery, or only the ones named.

    Args:
        only: Names from the gallery: convergence.
    """
    for name in only or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    app()
