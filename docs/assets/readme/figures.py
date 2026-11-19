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

from integrix import expect, march, rule


# The errors below fall to the round-off of float64, so every figure needs it.
jax.config.update("jax_enable_x64", True)

OUT = Path(__file__).parent
app = App(help="Draw the README's gallery of integrix specimens.")
# Errors are drawn at no less than this, so an exact result stays on the plot.
FLOOR = 1e-16
# One colour per method, the same in every figure.
COLOURS = {
    "global": "#B45309",
    "local": "#2563EB",
    "reference": "0.35",
    "euler": "#B45309",
    "heun": "#2563EB",
    "rk3": "#059669",
    "rk4": "#7C3AED",
    "monte_carlo": "#B45309",
    "qmc": "#2563EB",
}


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


# ---------------------------------------------------------------- order


def decay(state: jnp.ndarray) -> jnp.ndarray:
    """f(y) = −y, whose solution from y(0) = 1 is y(t) = e^(−t)."""
    return -state


def march_to_one(step, n_steps: int) -> float:
    """y_N after N steps of h = 1 / N from y₀ = 1, with one step function."""
    size = 1.0 / n_steps

    def body(_, state):
        # yₙ₊₁ = Φ(yₙ, h)
        return step(state, decay, size)

    final = jax.lax.fori_loop(0, n_steps, body, jnp.asarray(1.0))
    return float(final)


def order_figure() -> None:
    """The global error at t = 1 against h, one line per step function."""
    steppers = {
        "euler_step": (march.euler_step, 1, COLOURS["euler"]),
        "heun_step": (march.heun_step, 2, COLOURS["heun"]),
        "rk3_ssp_step": (march.rk3_ssp_step, 3, COLOURS["rk3"]),
        "rk4_step": (march.rk4_step, 4, COLOURS["rk4"]),
    }
    counts = 2 ** np.arange(1, 11)
    sizes = 1.0 / counts
    exact = float(np.exp(-1.0))
    errors = {}
    for name, (step, order, _) in steppers.items():
        sweep = [abs(march_to_one(step, int(n)) - exact) for n in counts]
        errors[name] = np.maximum(np.array(sweep), FLOOR)
        # fit only above round-off
        kept = errors[name] > 1e-13
        fitted = slope(sizes[kept], errors[name][kept])
        logger.info(
            "order: {} (order {}), slope {:.2f} over {} step sizes, error {:.1e} "
            "at h = 1/1024",
            name,
            order,
            fitted,
            int(kept.sum()),
            errors[name][-1],
        )

    fig, top = headed(
        "integrix: each step function converges at its order",
        "The global error |y_N − e⁻¹| at t = 1 for dy/dt = −y, y(0) = 1, after "
        "N = 1/h steps of each march step function,\nwith h halved from 1/2 to "
        "1/1024. Dashed lines are hᵖ for p = 1 to 4, through each method's "
        "coarsest point.",
        (9, 4.6),
    )
    ax = fig.subplots()
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=top - 0.06)
    for name, (_, order, colour) in steppers.items():
        ax.loglog(
            sizes,
            errors[name],
            "o-",
            color=colour,
            markersize=3.5,
            linewidth=1.4,
            label=f"{name} (order {order})",
        )
        # C hᵖ through the coarsest point
        reference = errors[name][0] * (sizes / sizes[0]) ** order
        ax.loglog(sizes, reference, "--", color=colour, linewidth=0.8, alpha=0.6)
    ax.set_xlabel("step h", fontsize=9)
    ax.set_ylabel(r"error $|y_N - e^{-1}|$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="major", linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    save(fig, "order.png")


# ---------------------------------------------------------------- expectations


def product_of_cosines(point: jnp.ndarray) -> jnp.ndarray:
    """f(s) = cos s₁ cos s₂, with 𝔼[f] = e⁻¹ for s ~ N(0, I)."""
    return jnp.prod(jnp.cos(point))


def standard_normal(key: jax.Array, n: int) -> jnp.ndarray:
    """S draws of s ~ N(0, I) in two dimensions."""
    return jax.random.normal(key, (n, 2))


def expectation_figure() -> None:
    """The error of Monte Carlo and randomised QMC against the evaluations S."""
    exact = float(np.exp(-1.0))
    shifts = 16
    replicates = 20
    counts = 2 ** np.arange(8, 17)
    keys = jax.random.split(jax.random.key(0), replicates)
    estimators = {
        "monte_carlo": lambda key, n: expect.monte_carlo(
            product_of_cosines, standard_normal, n, key=key
        ),
        # n points per shift, so S = n R evaluations, the same as Monte Carlo
        "qmc": lambda key, n: expect.qmc(
            product_of_cosines,
            2,
            n // shifts,
            n_shifts=shifts,
            transform=jax.scipy.special.ndtri,
            key=key,
        ),
    }
    rms_error, mean_stderr = {}, {}
    for name, estimate in estimators.items():
        rms, reported = [], []
        for n in counts:
            results = [estimate(key, int(n)) for key in keys]
            errors = np.array([float(r.value) - exact for r in results])
            # √(mean of squared errors) over the replicates
            rms.append(float(np.sqrt(np.mean(errors**2))))
            reported.append(float(np.mean([float(r.standard_error) for r in results])))
        rms_error[name] = np.array(rms)
        mean_stderr[name] = np.array(reported)
        fitted = slope(counts, rms_error[name])
        logger.info(
            "expect: {}, slope {:.2f}; at S = {}, RMS error {:.1e} and mean "
            "standard error {:.1e}",
            name,
            fitted,
            int(counts[-1]),
            rms_error[name][-1],
            mean_stderr[name][-1],
        )
    ratio = rms_error["monte_carlo"][-1] / rms_error["qmc"][-1]
    logger.info("expect: Monte Carlo over QMC at S = {}: {:.0f}", counts[-1], ratio)

    fig, top = headed(
        "integrix: quasi-Monte Carlo beats Monte Carlo, and both know their error",
        "𝔼[cos s₁ cos s₂] = e⁻¹ for s ~ N(0, I), by expect.monte_carlo and by "
        f"expect.qmc (Sobol points through ndtri, {shifts} random shifts),\nwith "
        f"the same S evaluations. Solid: the RMS error over {replicates} seeds; "
        "dashed: the mean standard error each call reported.",
        (9, 4.6),
    )
    ax = fig.subplots()
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=top - 0.06)
    labels = {"monte_carlo": "monte_carlo", "qmc": "qmc (Sobol)"}
    for name, label in labels.items():
        colour = COLOURS[name]
        ax.loglog(
            counts,
            rms_error[name],
            "o-",
            color=colour,
            markersize=3.5,
            linewidth=1.4,
            label=f"{label}: RMS error",
        )
        ax.loglog(
            counts,
            mean_stderr[name],
            "--",
            color=colour,
            linewidth=1,
            label=f"{label}: standard error",
        )
    # S^(−1/2) and S^(−1) through each method's first point
    for name, power, text in (("monte_carlo", -0.5, "S⁻¹ᐟ²"), ("qmc", -1.0, "S⁻¹")):
        reference = rms_error[name][0] * (counts / counts[0]) ** power
        ax.loglog(counts, reference, ":", color=COLOURS["reference"], linewidth=1)
        ax.annotate(
            text,
            (counts[-1], reference[-1]),
            textcoords="offset points",
            xytext=(4, 0),
            fontsize=8,
            color=COLOURS["reference"],
            va="center",
        )
    ax.set_xlabel("evaluations S", fontsize=9)
    ax.set_ylabel(r"error of $\hat{I}$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="major", linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    save(fig, "expectations.png")


FIGURES = {
    "convergence": convergence_figure,
    "order": order_figure,
    "expectations": expectation_figure,
}


@app.default
def main(*, only: list[str] | None = None) -> None:
    """Draw every figure of the gallery, or only the ones named.

    Args:
        only: Names from the gallery: convergence, order and expectations.
    """
    for name in only or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    app()
