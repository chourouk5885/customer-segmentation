"""RFM feature engineering.

Recency   - days since the customer's last purchase (lower is better)
Frequency - number of distinct invoices (orders)
Monetary  - total revenue from the customer
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RFM_COLS = ["Recency", "Frequency", "Monetary"]


def compute_rfm(transactions: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Aggregate cleaned transactions into one RFM row per customer.

    `snapshot_date` defaults to the day after the last transaction, so the most
    recent customer has Recency = 1.
    """
    if snapshot_date is None:
        snapshot_date = transactions["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    rfm = transactions.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    rfm["Recency"] = (snapshot_date - rfm["LastPurchase"].dt.normalize()).dt.days
    return rfm[RFM_COLS].astype({"Recency": int, "Frequency": int, "Monetary": float})


def scale_rfm(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """log1p-transform then standardize.

    All three RFM features are heavily right-skewed (a few wholesale customers
    spend 100x the median). K-Means uses Euclidean distance, so without the log
    those outliers would dominate the clusters; without scaling, Monetary
    (thousands) would swamp Frequency (single digits).
    """
    logged = np.log1p(rfm[RFM_COLS])
    scaler = StandardScaler()
    return scaler.fit_transform(logged), scaler


def rfm_scores(rfm: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """Classic 1-5 quintile RFM scores - a rule-based baseline to compare with K-Means."""
    scores = pd.DataFrame(index=rfm.index)
    # rank(method="first") breaks ties so qcut always gets q unique edges
    scores["R_score"] = pd.qcut(rfm["Recency"].rank(method="first"), q, labels=range(q, 0, -1)).astype(int)
    scores["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q, labels=range(1, q + 1)).astype(int)
    scores["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q, labels=range(1, q + 1)).astype(int)
    scores["RFM_score"] = scores[["R_score", "F_score", "M_score"]].sum(axis=1)
    return scores
