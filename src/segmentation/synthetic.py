"""Synthetic transactions in the exact Online Retail schema.

Used by `python main.py --demo` (try the pipeline without downloading anything)
and by the unit tests. Customers are drawn from a few hidden behaviour types
so the clustering has real structure to find. Includes the same kinds of dirt
as the real file: missing CustomerIDs, cancellations, negative quantities and
duplicates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (share of customers, mean days since last order, orders lambda, mean basket £)
_PERSONAS = {
    "vip": (0.08, 10, 20, 600),
    "loyal": (0.22, 30, 7, 350),
    "new": (0.20, 25, 1.2, 250),
    "slipping": (0.20, 120, 3, 300),
    "lost": (0.30, 260, 1.0, 180),
}
_COUNTRIES = ["United Kingdom"] * 9 + ["Germany", "France", "EIRE", "Spain", "Netherlands"]


def make_transactions(n_customers: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2011-12-09 12:00")
    start = pd.Timestamp("2010-12-01 08:00")
    span_days = (end - start).days

    names = list(_PERSONAS)
    probs = np.array([p[0] for p in _PERSONAS.values()])
    rows, invoice = [], 536365
    for i in range(n_customers):
        cid = 12346 + i
        persona = _PERSONAS[names[rng.choice(len(names), p=probs / probs.sum())]]
        _, rec_mean, lam, basket = persona
        last = min(span_days - 1, int(rng.exponential(rec_mean)))
        n_orders = 1 + rng.poisson(lam)
        earliest = min(span_days - 1, last + int(rng.integers(30, 365)))
        offsets = np.sort(rng.integers(last, earliest + 1, size=n_orders))
        offsets[0] = last
        country = rng.choice(_COUNTRIES)
        for off in offsets:
            invoice += 1
            ts = end - pd.Timedelta(days=int(off), minutes=int(rng.integers(0, 600)))
            n_lines = 1 + rng.poisson(4)
            for _ in range(n_lines):
                qty = int(1 + rng.poisson(6))
                price = round(float(rng.lognormal(np.log(basket / n_lines / qty + 0.5), 0.5)), 2)
                rows.append((str(invoice), f"2{rng.integers(1000, 3999)}", "ITEM", qty, ts, price, float(cid), country))

    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    # --- add realistic dirt ---
    n = len(df)
    idx = rng.choice(n, size=n // 20, replace=False)
    df.loc[idx, "CustomerID"] = np.nan                       # guests
    canc = df.sample(n // 50, random_state=seed).copy()
    canc["InvoiceNo"] = "C" + canc["InvoiceNo"]
    canc["Quantity"] = -canc["Quantity"]                     # cancellations
    zero_price = df.sample(n // 200, random_state=seed + 1).copy()
    zero_price["UnitPrice"] = 0.0                            # free items / adjustments
    dupes = df.sample(n // 100, random_state=seed + 2)       # duplicate lines
    return pd.concat([df, canc, zero_price, dupes], ignore_index=True).sort_values("InvoiceDate").reset_index(drop=True)
