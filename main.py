"""Run the full customer segmentation pipeline.

Examples
--------
    python main.py                       # real data, K chosen automatically
    python main.py --k 4                 # force 4 segments
    python main.py --country "United Kingdom"
    python main.py --demo                # synthetic data, no download needed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from segmentation.pipeline import Config, run  # noqa: E402


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="RFM + K-Means customer segmentation")
    p.add_argument("--data", type=Path, default=Path("data/raw/Online Retail.xlsx"),
                   help="Path to the Online Retail .xlsx or .csv")
    p.add_argument("--k", type=int, default=None, help="Number of clusters (default: auto)")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--country", default=None, help='Keep only one country, e.g. "United Kingdom"')
    p.add_argument("--demo", action="store_true", help="Use synthetic data (outputs go to reports/demo)")
    a = p.parse_args()
    reports = Path("reports/demo") if a.demo else Path("reports")
    processed = Path("data/processed/demo") if a.demo else Path("data/processed")
    return Config(data_path=a.data, reports_dir=reports, processed_dir=processed,
                  country=a.country, k=a.k, k_min=a.k_min, k_max=a.k_max, demo=a.demo)


if __name__ == "__main__":
    try:
        run(parse_args())
    except FileNotFoundError as e:
        sys.exit(f"\nERROR: {e}")
