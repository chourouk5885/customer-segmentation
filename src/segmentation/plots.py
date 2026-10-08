"""Static (Matplotlib) and interactive (Plotly, optional) figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

try:  # inside Jupyter keep the inline backend; in scripts/CI use a headless one
    get_ipython  # type: ignore[name-defined]  # noqa: B018
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import silhouette_samples

from .interpret import base_name
from .rfm import RFM_COLS

# ---- Style ------------------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
NEUTRAL = "#f0efec"

# Colour follows the segment *persona*, so a segment keeps its colour across runs.
SEGMENT_COLORS = {
    "Champions": BLUE,
    "Loyal Customers": AQUA,
    "Promising / New": VIOLET,
    "Need Attention": YELLOW,
    "At Risk": ORANGE,
    "Can't Lose Them": RED,
    "Hibernating / Lost": "#8a8984",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

DIVERGING = LinearSegmentedColormap.from_list("rfm_div", [RED, NEUTRAL, BLUE])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def segment_color(segment: str) -> str:
    return SEGMENT_COLORS.get(base_name(segment), INK_2)


def _save(fig, outdir: Path | None, name: str):
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{name}.png")
    return fig


# ---- EDA --------------------------------------------------------------------
def plot_eda_overview(tx: pd.DataFrame, outdir: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})

    monthly = tx.set_index("InvoiceDate")["Revenue"].resample("MS").sum() / 1e3
    ax = axes[0]
    ax.plot(monthly.index, monthly.values, color=BLUE, lw=2, marker="o", ms=5)
    last = tx["InvoiceDate"].max()
    partial = last.day < last.days_in_month
    ax.set_title("Monthly revenue" + (f"  ({last:%b %Y} is a partial month)" if partial else ""))
    ax.set_ylabel("Revenue (thousands)")
    ax.grid(axis="x", visible=False)

    by_country = tx.groupby("Country")["Revenue"].sum()
    top = (100 * by_country / by_country.sum()).sort_values().tail(10)
    ax = axes[1]
    ax.barh(top.index, top.values, color=BLUE, height=0.7)
    for y, v in enumerate(top.values):
        ax.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=8, color=INK_2)
    ax.set_title("Top 10 countries, share of revenue")
    ax.set_xlabel("% of revenue")
    ax.set_xlim(0, top.max() * 1.15)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, outdir, "01_eda_overview")


def plot_rfm_distributions(rfm: pd.DataFrame, outdir: Path | None = None):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for i, col in enumerate(RFM_COLS):
        axes[0, i].hist(rfm[col], bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
        axes[0, i].set_title(f"{col} (raw)  skew={rfm[col].skew():.1f}")
        logged = np.log1p(rfm[col])
        axes[1, i].hist(logged, bins=50, color=AQUA, edgecolor=SURFACE, linewidth=0.5)
        axes[1, i].set_title(f"log1p({col})  skew={logged.skew():.1f}")
        for ax in axes[:, i]:
            ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("Customers")
    axes[1, 0].set_ylabel("Customers")
    fig.suptitle("RFM features are heavily right-skewed; log1p makes them usable for K-Means",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, outdir, "02_rfm_distributions")


# ---- Choosing K ---------------------------------------------------------------
def plot_k_selection(scores: pd.DataFrame, chosen_k: int, elbow_k: int, outdir: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ks = scores.index

    ax = axes[0]
    ax.plot(ks, scores["inertia"], color=BLUE, lw=2, marker="o", ms=6)
    ax.scatter([elbow_k], [scores.loc[elbow_k, "inertia"]], s=160, facecolor="none",
               edgecolor=ORANGE, linewidth=2, zorder=3)
    ax.annotate(f"elbow k={elbow_k}", (elbow_k, scores.loc[elbow_k, "inertia"]),
                xytext=(12, 12), textcoords="offset points", color=INK)
    ax.set_title("Elbow method: inertia (within-cluster SS)")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Inertia")

    ax = axes[1]
    colors = [ORANGE if k == chosen_k else BLUE for k in ks]
    ax.bar(ks, scores["silhouette"], color=colors, width=0.65)
    for k, s in scores["silhouette"].items():
        ax.text(k, s + 0.005, f"{s:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_title(f"Silhouette score (chosen k={chosen_k} in orange)")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Mean silhouette")
    ax.set_xticks(list(ks))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, outdir, "03_k_selection")


def plot_silhouette_detail(X: np.ndarray, labels: np.ndarray, names: dict[int, str], outdir: Path | None = None):
    sil = silhouette_samples(X, labels)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = 0
    for c in sorted(np.unique(labels), key=lambda c: names[c]):
        vals = np.sort(sil[labels == c])
        ax.fill_betweenx(np.arange(y, y + len(vals)), 0, vals, color=segment_color(names[c]), alpha=0.9)
        ax.text(-0.02, y + len(vals) / 2, names[c], ha="right", va="center", fontsize=9, color=INK)
        y += len(vals) + 30
    ax.axvline(sil.mean(), color=INK, ls="--", lw=1)
    ax.text(sil.mean() + 0.01, 1.01, f"mean = {sil.mean():.2f}", va="bottom", color=INK,
            transform=ax.get_xaxis_transform())
    ax.set_yticks([])
    ax.set_xlabel("Silhouette coefficient")
    ax.set_title("Silhouette per customer, by segment")
    ax.grid(axis="y", visible=False)
    return _save(fig, outdir, "04_silhouette_by_segment")


# ---- Segments -----------------------------------------------------------------
def plot_pca_segments(X2: np.ndarray, segments: pd.Series, pca, centroids2: np.ndarray,
                      names: dict[int, str], outdir: Path | None = None):
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, seg in enumerate(sorted(segments.unique())):
        mask = (segments == seg).to_numpy()
        ax.scatter(X2[mask, 0], X2[mask, 1], s=12, alpha=0.55, color=segment_color(seg),
                   marker=MARKERS[i % len(MARKERS)], label=f"{seg} (n={mask.sum():,})", linewidths=0)
    for c, (x, y) in enumerate(centroids2):
        ax.scatter(x, y, s=220, marker="o", color=segment_color(names[c]), edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(names[c], (x, y), xytext=(10, 8), textcoords="offset points", fontweight="bold",
                    color=INK, fontsize=9, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRID, alpha=0.9))
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0f}% of variance)")
    ax.set_title(f"Customer segments in PCA space ({ev[:2].sum():.0f}% of variance shown)")
    ax.legend(loc="best", markerscale=2, fontsize=9)
    return _save(fig, outdir, "05_pca_segments")


def plot_pca_loadings(pca, outdir: Path | None = None):
    load = pd.DataFrame(pca.components_[:2].T, index=RFM_COLS, columns=["PC1", "PC2"])
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(load.values, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(2), load.columns)
    ax.set_yticks(range(3), load.index)
    for (i, j), v in np.ndenumerate(load.values):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=INK)
    ax.grid(False)
    ax.set_title("What the PCA axes mean (loadings)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, outdir, "06_pca_loadings")


def plot_segment_heatmap(centroids: np.ndarray, names: dict[int, str], outdir: Path | None = None):
    """Centroids as z-scores vs the average customer. Recency is flipped so that
    blue always means 'better than average'."""
    df = pd.DataFrame(centroids, columns=RFM_COLS)
    df["Recency"] = -df["Recency"]
    df.index = [names[i] for i in df.index]
    df = df.sort_values("Monetary", ascending=False)
    df.columns = ["Recency\n(recent = +)", "Frequency", "Monetary"]
    lim = max(1.0, np.abs(df.values).max())
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(df) + 1.4))
    im = ax.imshow(df.values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(3), df.columns)
    ax.set_yticks(range(len(df)), df.index)
    for (i, j), v in np.ndenumerate(df.values):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=INK, fontsize=10)
    ax.grid(False)
    ax.set_title("Segment profiles (std. deviations vs average customer)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="blue = better than average")
    return _save(fig, outdir, "07_segment_heatmap")


def plot_segment_share(profile: pd.DataFrame, outdir: Path | None = None):
    prof = profile.sort_values("Revenue_share_%")
    y = np.arange(len(prof))
    h = 0.38
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(prof) + 1.5))
    ax.barh(y + h / 2, prof["Customer_share_%"], height=h - 0.02, color=INK_2, label="% of customers")
    ax.barh(y - h / 2, prof["Revenue_share_%"], height=h - 0.02, color=BLUE, label="% of revenue")
    for yi, (c, r) in enumerate(zip(prof["Customer_share_%"], prof["Revenue_share_%"])):
        ax.text(c + 0.5, yi + h / 2, f"{c:.0f}%", va="center", fontsize=8, color=INK_2)
        ax.text(r + 0.5, yi - h / 2, f"{r:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, prof.index)
    ax.set_xlabel("Share (%)")
    ax.set_title("Who drives the revenue? Customer share vs revenue share")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return _save(fig, outdir, "08_segment_share")


def plot_segment_boxplots(rfm_labeled: pd.DataFrame, outdir: Path | None = None):
    order = rfm_labeled.groupby("Segment")["Monetary"].median().sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.55 * len(order) + 2), sharey=True)
    for ax, col in zip(axes, RFM_COLS):
        data = [rfm_labeled.loc[rfm_labeled["Segment"] == s, col] for s in order]
        bp = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False, widths=0.6,
                        medianprops=dict(color=INK, lw=1.5))
        for patch, s in zip(bp["boxes"], order):
            patch.set_facecolor(segment_color(s))
            patch.set_alpha(0.8)
            patch.set_edgecolor(SURFACE)
        if col != "Recency":
            ax.set_xscale("log")
        ax.set_title(col + (" (days)" if col == "Recency" else " (log scale)"))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(1, len(order) + 1), order)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return _save(fig, outdir, "09_segment_boxplots")


# ---- Interactive (optional) -----------------------------------------------------
def plotly_3d(rfm_labeled: pd.DataFrame, path: Path) -> bool:
    """Interactive 3D RFM scatter saved as a standalone HTML file. Returns False if
    plotly isn't installed."""
    try:
        import plotly.express as px
    except ImportError:
        return False
    df = rfm_labeled.reset_index()
    fig = px.scatter_3d(
        df, x="Recency", y="Frequency", z="Monetary", color="Segment",
        color_discrete_map={s: segment_color(s) for s in df["Segment"].unique()},
        log_y=True, log_z=True, opacity=0.7, hover_data=["CustomerID"],
        title="Customer segments in RFM space (drag to rotate)",
    )
    fig.update_traces(marker=dict(size=3))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    return True


def plotly_pca(X2: np.ndarray, rfm_labeled: pd.DataFrame, path: Path) -> bool:
    try:
        import plotly.express as px
    except ImportError:
        return False
    df = rfm_labeled.reset_index().assign(PC1=X2[:, 0], PC2=X2[:, 1])
    fig = px.scatter(
        df, x="PC1", y="PC2", color="Segment", symbol="Segment",
        color_discrete_map={s: segment_color(s) for s in df["Segment"].unique()},
        hover_data=["CustomerID", "Recency", "Frequency", "Monetary"], opacity=0.7,
        title="Customer segments in PCA space (hover for details)",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    return True
