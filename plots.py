"""Publication-style figures for the qPCR Fold-Change Analyzer (matplotlib)."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Colour-blind-checked categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#d9d8d4"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fold_change_plot(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame | None,
    group_order: list,
    targets: list,
    scale: str = "linear",
    error: str = "ci",
    show_points: bool = True,
    title: str = "",
    width: float | None = None,
    height: float = 4.2,
):
    """
    Grouped bar chart: one cluster per target, one bar per group.
    Bars = geometric mean fold change; error bars = 95 % CI or ±SEM (log2 scale,
    back-transformed); dots = individual samples; stars = adjusted p vs control.
    """
    n_t, n_g = len(targets), len(group_order)
    width = width or max(4.5, 1.1 + n_t * (0.55 + 0.42 * n_g))
    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    _style(ax)

    bar_w = 0.8 / n_g
    rng = np.random.default_rng(0)
    log = scale == "log2"
    top_by_target = {}

    for gi, group in enumerate(group_order):
        color = PALETTE[gi % len(PALETTE)]
        for ti, target in enumerate(targets):
            row = summary[(summary["Target"] == target) & (summary["Group"] == group)]
            if row.empty or row["n"].iloc[0] == 0:
                continue
            r = row.iloc[0]
            x = ti - 0.4 + bar_w * (gi + 0.5)
            m = r["Mean log2 FC"]
            if error == "ci":
                lo_l2 = np.log2(r["95% CI low"]) if pd.notna(r["95% CI low"]) else m
                hi_l2 = np.log2(r["95% CI high"]) if pd.notna(r["95% CI high"]) else m
            else:
                sem = r["SEM log2 FC"] if pd.notna(r["SEM log2 FC"]) else 0
                lo_l2, hi_l2 = m - sem, m + sem

            if log:
                h, lo, hi = m, lo_l2, hi_l2
            else:
                h, lo, hi = 2 ** m, 2 ** lo_l2, 2 ** hi_l2

            ax.bar(x, h, width=bar_w * 0.86, color=color, alpha=0.85, edgecolor="white", linewidth=1,
                   label=group if ti == 0 else None, zorder=2)
            ax.errorbar(x, h, yerr=[[h - lo], [hi - h]], fmt="none", ecolor=INK, elinewidth=1.1,
                        capsize=3, capthick=1.1, zorder=4)

            pts = results[(results["Target"] == target) & (results["Group"] == group)]["log2 FC"].dropna()
            vals = pts.to_numpy() if log else 2 ** pts.to_numpy()
            if show_points and len(vals):
                jitter = rng.uniform(-bar_w * 0.22, bar_w * 0.22, len(vals))
                ax.scatter(x + jitter, vals, s=16, color="white", edgecolor=INK, linewidth=0.7, zorder=5)

            extent = [hi, lo, h] + (list(vals) if show_points and len(vals) else [])
            if log and h < 0:  # down-regulated on log scale: label goes below the bar
                top_by_target[(target, group)] = (x, min(extent), "below")
            else:
                top_by_target[(target, group)] = (x, max(extent), "above")

    # Significance stars vs control
    y_min, y_max = ax.get_ylim()
    pad = (y_max - y_min) * 0.04
    if comparisons is not None and not comparisons.empty:
        for _, c in comparisons.iterrows():
            key = (c["Target"], c["Group"])
            if key in top_by_target and c["Target"] in targets and c["Significance"]:
                x, edge, side = top_by_target[key]
                star = c["Significance"]
                y = edge + pad if side == "above" else edge - pad
                ax.text(x, y, star, ha="center", va="bottom" if side == "above" else "top",
                        fontsize=11 if star != "ns" else 8.5,
                        color=INK if star != "ns" else MUTED, zorder=6)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - (y_max - y_min) * (0.06 if log else 0), y_max + (y_max - y_min) * 0.06)

    ax.axhline(0 if log else 1, color=MUTED, linewidth=0.9, linestyle="--", zorder=3)
    if not log:
        ax.set_ylim(bottom=0)
    ax.set_xticks(range(n_t))
    ax.set_xticklabels(targets, fontsize=10.5, color=INK)
    ax.set_xlim(-0.6, n_t - 0.4)
    ax.set_ylabel("log2 fold change (−ΔΔCt)" if log else "Relative expression (fold change)",
                  fontsize=10.5, color=INK)
    if title:
        ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    err_txt = "95% CI" if error == "ci" else "±SEM"
    fig.text(0.01, 0.005, f"Bars: geometric mean; error bars: {err_txt}; dots: individual samples; "
             "stars: adjusted p vs control (* <0.05, ** <0.01, *** <0.001).",
             fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def reference_ct_plot(qc: pd.DataFrame, candidates: list, group_order: list):
    """Raw mean Ct of each candidate reference gene by group (dots + group mean)."""
    fig, ax = plt.subplots(figsize=(max(4.5, 1.6 * len(candidates) + 1.5), 3.6), dpi=110)
    _style(ax)
    n_g = len(group_order)
    w = 0.8 / n_g
    rng = np.random.default_rng(1)
    for gi, group in enumerate(group_order):
        color = PALETTE[gi % len(PALETTE)]
        for ti, gene in enumerate(candidates):
            v = qc[(qc["Target"] == gene) & (qc["Group"] == group)]["Mean Ct"].dropna().to_numpy()
            if not len(v):
                continue
            x = ti - 0.4 + w * (gi + 0.5)
            ax.scatter(x + rng.uniform(-w * 0.2, w * 0.2, len(v)), v, s=22, color=color,
                       edgecolor="white", linewidth=0.6, zorder=3, label=group if ti == 0 else None)
            ax.hlines(v.mean(), x - w * 0.35, x + w * 0.35, color=INK, linewidth=1.6, zorder=4)
    ax.set_xticks(range(len(candidates)))
    ax.set_xticklabels(candidates, fontsize=10.5)
    ax.set_ylabel("Mean Ct", fontsize=10.5)
    ax.invert_yaxis()  # lower Ct = more abundant, shown higher
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig


def standard_curve_plot(log_q, ct, curve):
    fig, ax = plt.subplots(figsize=(5, 3.6), dpi=110)
    _style(ax)
    ax.scatter(log_q, ct, s=28, color=PALETTE[0], edgecolor="white", zorder=3)
    xs = np.linspace(np.nanmin(log_q), np.nanmax(log_q), 50)
    ax.plot(xs, curve.slope * xs + curve.intercept, color=INK, linewidth=1.4)
    ax.set_xlabel("log10 quantity / dilution")
    ax.set_ylabel("Ct")
    ax.set_title(f"Efficiency {curve.efficiency_percent:.1f} %   R² {curve.r_squared:.4f}",
                 fontsize=10.5, loc="left")
    fig.tight_layout()
    return fig


def to_bytes(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def sample_qc_plot(table: pd.DataFrame, value_col: str, group_order: list, lines: dict,
                   ylabel: str = "", highlight=None):
    """
    One dot per sample (ordered and coloured by group) with labelled threshold lines.
    Samples in `highlight` are labelled with their names.
    """
    highlight = set(highlight or [])
    df = table.copy()
    order = {g: i for i, g in enumerate(group_order)}
    df["_g"] = df["Group"].map(order).fillna(len(order))
    df = df.sort_values(["_g", "Sample"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(5.5, 0.32 * len(df) + 2.5), 3.4), dpi=110)
    _style(ax)
    for gi, group in enumerate(group_order):
        sub = df[df["Group"] == group]
        if sub.empty:
            continue
        ax.scatter(sub.index, sub[value_col], s=34, color=PALETTE[gi % len(PALETTE)],
                   edgecolor="white", linewidth=0.7, zorder=3, label=group)
    for _, r in df[df["Sample"].isin(highlight)].iterrows():
        if pd.notna(r[value_col]):
            ax.annotate(str(r["Sample"]), (r.name, r[value_col]), xytext=(5, 4), textcoords="offset points",
                        fontsize=8.5, color=INK)
    x_end = len(df) - 0.5
    for label, y in lines.items():
        ax.axhline(y, color=MUTED, linewidth=0.9, linestyle="--", zorder=2)
        ax.text(x_end, y, f" {label}", va="center", ha="left", fontsize=8, color=MUTED)
    ax.set_xticks([])
    ax.set_xlim(-0.8, x_end)
    ax.set_xlabel("Samples (grouped)", fontsize=9.5, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.12, 1.0))
    fig.tight_layout()
    return fig
