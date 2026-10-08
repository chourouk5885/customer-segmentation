"""Turn clusters into named business segments.

Each cluster centroid lives in standardized log-RFM space, so its coordinates
are "how many standard deviations above/below the average customer" this
segment sits on Recency, Frequency and Monetary. We combine them into two axes:

    engagement = -z(Recency)                 (bought recently -> high)
    value      = mean(z(Frequency), z(Monetary))

and match each cluster to the closest familiar RFM persona on those two axes.
Everything is relative to the customer base, so it works for any dataset and K.
Always sanity-check the names against the profile table - they are a
starting point for the business conversation, not ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rfm import RFM_COLS

SEGMENT_PLAYBOOK = {
    "Champions": (
        "Bought recently, buy often and spend the most.",
        "Reward them: VIP/loyalty perks, early access to new products, ask for reviews and referrals.",
    ),
    "Loyal Customers": (
        "Regular buyers with above-average spend who are still active.",
        "Upsell higher-value products, cross-sell bundles, invite into the loyalty program.",
    ),
    "Promising / New": (
        "Bought recently but only a few times and with modest spend so far.",
        "Onboarding journey, second-purchase incentive, product recommendations to build the habit.",
    ),
    "At Risk": (
        "Used to buy often and spend well, but haven't purchased in a while.",
        "Win-back campaign now: personalised emails, limited-time offer, ask what changed.",
    ),
    "Can't Lose Them": (
        "Top historical spenders who have gone quiet.",
        "Highest-priority retention: direct outreach from account manager, strong reactivation offer.",
    ),
    "Need Attention": (
        "Average customers whose activity is starting to slip.",
        "Re-engage with relevant, time-limited promotions before they lapse.",
    ),
    "Hibernating / Lost": (
        "Last purchase long ago, few orders, low spend.",
        "Low-cost automated reactivation only; exclude from expensive campaigns if they don't respond.",
    ),
}


# Where each persona "lives" on the (engagement, value) plane, after both axes
# are rescaled to [-1, 1] across the clusters found.
PERSONA_PROTOTYPES = {
    "Champions": (0.8, 1.0),
    "Loyal Customers": (0.4, 0.4),
    "Promising / New": (0.7, -0.6),
    "Need Attention": (0.0, -0.1),
    "At Risk": (-0.6, 0.4),
    "Can't Lose Them": (-0.7, 0.9),
    "Hibernating / Lost": (-1.0, -1.0),
}
LAPSED = ["At Risk", "Can't Lose Them", "Hibernating / Lost"]  # need below-average recency
ACTIVE = ["Champions", "Loyal Customers", "Promising / New"]   # need roughly average-or-better recency


def name_clusters(centroids: np.ndarray) -> dict[int, str]:
    """Map cluster id -> unique segment name.

    Each cluster is placed on the (engagement, value) plane, both axes scaled
    to [-1, 1], and clusters are matched one-to-one to the nearest persona
    prototypes with the Hungarian algorithm (minimum total distance). With
    more clusters than personas, personas are reused and suffixed by tier.
    """
    from scipy.optimize import linear_sum_assignment

    df = pd.DataFrame(centroids, columns=RFM_COLS)
    eng = -df["Recency"].to_numpy()
    val = ((df["Frequency"] + df["Monetary"]) / 2).to_numpy()
    pts = np.column_stack([eng / (np.abs(eng).max() or 1), val / (np.abs(val).max() or 1)])

    names = list(PERSONA_PROTOTYPES)
    reps = int(np.ceil(len(pts) / len(names))) + 1
    slots = names * reps
    protos = np.array([PERSONA_PROTOTYPES[n] for n in slots])
    cost = ((pts[:, None, :] - protos[None, :, :]) ** 2).sum(-1)
    # Prefer using every persona once before reusing one ...
    cost += np.repeat(np.arange(reps), len(names))[None, :] * 10.0
    # ... and never call a recently-active cluster "lapsed" (or vice versa).
    z_eng = eng[:, None]
    lapsed = np.isin(slots, LAPSED)[None, :]
    active = np.isin(slots, ACTIVE)[None, :]
    cost += 100.0 * ((lapsed & (z_eng > 0)) | (active & (z_eng < -0.25)))
    rows, cols = linear_sum_assignment(cost)
    df["name"] = pd.Series([slots[c] for c in cols], index=rows).sort_index().to_numpy()
    df["value"] = val

    for name, group in df.groupby("name"):
        if len(group) > 1:
            order = group.sort_values("value", ascending=False).index
            for tier, idx in zip(["high value", "mid value", "low value", "tier 4"], order):
                df.loc[idx, "name"] = f"{name} ({tier})"
    return df["name"].to_dict()


def base_name(segment: str) -> str:
    return segment.split(" (")[0]


def segment_profile(rfm_labeled: pd.DataFrame) -> pd.DataFrame:
    """One row per segment: size, revenue share and typical (median) RFM values."""
    total_rev = rfm_labeled["Monetary"].sum()
    prof = rfm_labeled.groupby("Segment").agg(
        Customers=("Recency", "size"),
        Recency_median_days=("Recency", "median"),
        Frequency_median=("Frequency", "median"),
        Monetary_median=("Monetary", "median"),
        Recency_mean_days=("Recency", "mean"),
        Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"),
        Revenue=("Monetary", "sum"),
    )
    prof["Customer_share_%"] = 100 * prof["Customers"] / prof["Customers"].sum()
    prof["Revenue_share_%"] = 100 * prof["Revenue"] / total_rev
    prof["Description"] = [SEGMENT_PLAYBOOK[base_name(s)][0] for s in prof.index]
    prof["Recommended_action"] = [SEGMENT_PLAYBOOK[base_name(s)][1] for s in prof.index]
    return prof.sort_values("Revenue", ascending=False).round(2)
