"""Loading and cleaning the UCI Online Retail dataset.

Dataset: https://archive.ics.uci.edu/dataset/352/online+retail
~541k transactions from a UK-based online gift retailer, Dec 2010 - Dec 2011.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw transactions from .xlsx (the UCI original) or .csv.

    Reading the 23 MB Excel file takes ~1 minute, so after the first read a
    CSV copy is cached next to it and used on later runs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first, "
            "or use `python main.py --demo` to try the pipeline on synthetic data."
        )

    dtypes = {"InvoiceNo": str, "StockCode": str, "Description": str, "Country": str}

    if path.suffix.lower() in {".xlsx", ".xls"}:
        cache = path.with_suffix(".csv")
        if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            df = pd.read_csv(cache, dtype=dtypes, parse_dates=["InvoiceDate"])
        else:
            print(f"Reading {path.name} (first time only, ~1 min)...")
            df = pd.read_excel(path, dtype=dtypes)
            df.to_csv(cache, index=False)
    else:
        df = pd.read_csv(path, dtype=dtypes, parse_dates=["InvoiceDate"], encoding_errors="replace")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def clean(df: pd.DataFrame, country: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean transactions for RFM analysis.

    Steps (each one is logged in the returned report):
      1. drop rows without CustomerID (can't attribute them to a customer)
      2. drop cancellations (InvoiceNo starting with "C")
      3. keep Quantity > 0 and UnitPrice > 0 (returns, adjustments, free items)
      4. drop exact duplicate rows
      5. optionally keep a single country
    Adds a `Revenue` column = Quantity * UnitPrice.
    """
    report = {"raw_rows": len(df)}
    out = df.copy()

    out = out.dropna(subset=["CustomerID"])
    report["after_drop_missing_customer"] = len(out)

    out["InvoiceNo"] = out["InvoiceNo"].astype(str)
    out = out[~out["InvoiceNo"].str.upper().str.startswith("C")]
    report["after_drop_cancellations"] = len(out)

    out = out[(out["Quantity"] > 0) & (out["UnitPrice"] > 0)]
    report["after_positive_qty_price"] = len(out)

    out = out.drop_duplicates()
    report["after_drop_duplicates"] = len(out)

    if country:
        out = out[out["Country"] == country]
        report[f"after_country_{country}"] = len(out)

    out["CustomerID"] = out["CustomerID"].astype(float).astype(int).astype(str)
    out["Revenue"] = out["Quantity"] * out["UnitPrice"]
    report["customers"] = out["CustomerID"].nunique()
    return out.reset_index(drop=True), report
