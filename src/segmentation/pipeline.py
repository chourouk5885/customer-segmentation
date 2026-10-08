"""End-to-end pipeline: load -> clean -> RFM -> choose K -> cluster -> PCA -> interpret -> report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import clustering, data, interpret, plots, rfm as rfm_mod
from .synthetic import make_transactions


@dataclass
class Config:
    data_path: Path = Path("data/raw/Online Retail.xlsx")
    reports_dir: Path = Path("reports")
    processed_dir: Path = Path("data/processed")
    country: str | None = None
    k: int | None = None          # None = choose automatically
    k_min: int = 2
    k_max: int = 10
    demo: bool = False
    log: list = field(default_factory=list)

    def say(self, msg: str = ""):
        print(msg)
        self.log.append(msg)


def run(cfg: Config) -> dict:
    figs = cfg.reports_dir / "figures"

    # 1. Load & clean -------------------------------------------------------------
    if cfg.demo:
        cfg.say("DEMO MODE: using synthetic transactions (same schema as Online Retail).")
        raw = make_transactions()
    else:
        raw = data.load_raw(cfg.data_path)
    tx, clean_report = data.clean(raw, country=cfg.country)
    cfg.say("\n== Cleaning ==")
    for k, v in clean_report.items():
        cfg.say(f"  {k:<32} {v:>10,}")
    cfg.say(f"  period: {tx['InvoiceDate'].min():%Y-%m-%d} -> {tx['InvoiceDate'].max():%Y-%m-%d}")

    # 2. RFM ------------------------------------------------------------------------
    rfm = rfm_mod.compute_rfm(tx)
    X, _ = rfm_mod.scale_rfm(rfm)
    cfg.say("\n== RFM summary ==")
    cfg.say(rfm.describe().round(1).to_string())

    plots.plot_eda_overview(tx, figs)
    plots.plot_rfm_distributions(rfm, figs)

    # 3. Choose K -----------------------------------------------------------------------
    scores = clustering.evaluate_k(X, range(cfg.k_min, cfg.k_max + 1))
    elbow_k = clustering.find_elbow(scores.index, scores["inertia"])
    if cfg.k:
        k, reason = cfg.k, f"k={cfg.k} set manually (elbow suggests {elbow_k})."
    else:
        k, reason = clustering.choose_k(scores, min_k=max(3, cfg.k_min), max_k=min(8, cfg.k_max))
    cfg.say("\n== Choosing K ==")
    cfg.say(scores.round(3).to_string())
    cfg.say(f"  -> {reason}")
    plots.plot_k_selection(scores, k, elbow_k, figs)

    # 4. Cluster & name -----------------------------------------------------------------
    km = clustering.fit_kmeans(X, k)
    names = interpret.name_clusters(km.cluster_centers_)
    labeled = rfm.copy()
    labeled["Cluster"] = km.labels_
    labeled["Segment"] = labeled["Cluster"].map(names)
    labeled = labeled.join(rfm_mod.rfm_scores(rfm))
    profile = interpret.segment_profile(labeled)

    # 5. Visualise ------------------------------------------------------------------------
    X2, pca = clustering.pca_project(X)
    c2 = pca.transform(km.cluster_centers_)
    plots.plot_silhouette_detail(X, km.labels_, names, figs)
    plots.plot_pca_segments(X2, labeled["Segment"], pca, c2, names, figs)
    plots.plot_pca_loadings(pca, figs)
    plots.plot_segment_heatmap(km.cluster_centers_, names, figs)
    plots.plot_segment_share(profile, figs)
    plots.plot_segment_boxplots(labeled, figs)
    plt.close("all")
    interactive = plots.plotly_3d(labeled, cfg.reports_dir / "interactive" / "rfm_3d.html")
    interactive &= plots.plotly_pca(X2, labeled, cfg.reports_dir / "interactive" / "pca_segments.html")
    if not interactive:
        cfg.say("\n(plotly not installed - skipped interactive HTML charts: pip install plotly)")

    # 6. Save outputs ------------------------------------------------------------------------
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(cfg.processed_dir / "customer_segments.csv")
    profile.to_csv(cfg.reports_dir / "segment_profiles.csv")
    scores.to_csv(cfg.reports_dir / "k_selection_scores.csv")
    summary = {
        "k": k,
        "k_reason": reason,
        "elbow_k": elbow_k,
        "silhouette": round(float(scores.loc[k, "silhouette"]) if k in scores.index else float("nan"), 4),
        "pca_explained_variance_2d": round(float(pca.explained_variance_ratio_.sum()), 4),
        "customers": int(len(rfm)),
        "cleaning": clean_report,
        "demo": cfg.demo,
    }
    (cfg.reports_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown_report(cfg.reports_dir / "SEGMENTS.md", profile, summary)

    cfg.say("\n== Segments ==")
    cols = ["Customers", "Customer_share_%", "Revenue_share_%", "Recency_median_days",
            "Frequency_median", "Monetary_median"]
    cfg.say(profile[cols].to_string())
    cfg.say(f"\nOutputs written to {cfg.reports_dir}/ and {cfg.processed_dir}/")
    return {"rfm": labeled, "profile": profile, "scores": scores, "summary": summary}


def write_markdown_report(path: Path, profile: pd.DataFrame, summary: dict):
    lines = [
        "# Customer Segments - Results",
        "",
        "_Auto-generated by `python main.py`. Do not edit by hand._",
        "",
    ]
    if summary["demo"]:
        lines += ["> **Demo run on synthetic data** - run without `--demo` for real results.", ""]
    lines += [
        f"- Customers clustered: **{summary['customers']:,}**",
        f"- Number of segments (K): **{summary['k']}** (silhouette = {summary['silhouette']:.3f})",
        f"- Why this K: {summary['k_reason']}",
        f"- 2-D PCA view keeps {summary['pca_explained_variance_2d']:.0%} of the variance",
        "",
        "## Segment overview",
        "",
        "| Segment | Customers | % customers | % revenue | Median recency (days) | Median orders | Median spend |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seg, r in profile.iterrows():
        lines.append(
            f"| {seg} | {int(r.Customers):,} | {r['Customer_share_%']:.1f}% | {r['Revenue_share_%']:.1f}% | "
            f"{r.Recency_median_days:.0f} | {r.Frequency_median:.0f} | {r.Monetary_median:,.0f} |"
        )
    lines += ["", "## What each segment means and what to do", ""]
    for seg, r in profile.iterrows():
        lines += [
            f"### {seg}",
            f"{int(r.Customers):,} customers ({r['Customer_share_%']:.0f}%) generating "
            f"{r['Revenue_share_%']:.0f}% of revenue. Typical customer: last order "
            f"{r.Recency_median_days:.0f} days ago, {r.Frequency_median:.0f} orders, "
            f"{r.Monetary_median:,.0f} total spend.",
            "",
            f"- **Who they are:** {r.Description}",
            f"- **Action:** {r.Recommended_action}",
            "",
        ]
    lines += [
        "## Figures",
        "",
        "![K selection](figures/03_k_selection.png)",
        "![PCA segments](figures/05_pca_segments.png)",
        "![Segment heatmap](figures/07_segment_heatmap.png)",
        "![Revenue share](figures/08_segment_share.png)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
