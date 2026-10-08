import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation import clustering, data, interpret, rfm  # noqa: E402
from segmentation.pipeline import Config, run  # noqa: E402
from segmentation.synthetic import make_transactions  # noqa: E402


@pytest.fixture(scope="module")
def tx():
    cleaned, _ = data.clean(make_transactions(n_customers=400, seed=1))
    return cleaned


def test_clean_removes_dirt(tx):
    assert tx["CustomerID"].notna().all()
    assert not tx["InvoiceNo"].str.startswith("C").any()
    assert (tx["Quantity"] > 0).all() and (tx["UnitPrice"] > 0).all()
    assert not tx.duplicated().any()
    assert np.allclose(tx["Revenue"], tx["Quantity"] * tx["UnitPrice"])


def test_rfm_hand_computed():
    t = pd.DataFrame({
        "CustomerID": ["A", "A", "A", "B"],
        "InvoiceNo": ["1", "1", "2", "3"],
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-01", "2011-01-10", "2011-01-05"]),
        "Revenue": [10.0, 5.0, 20.0, 7.0],
    })
    out = rfm.compute_rfm(t)
    assert out.loc["A"].tolist() == [1, 2, 35.0]   # snapshot = 2011-01-11
    assert out.loc["B"].tolist() == [6, 1, 7.0]


def test_scaling_is_standardized(tx):
    X, _ = rfm.scale_rfm(rfm.compute_rfm(tx))
    assert np.allclose(X.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(X.std(axis=0), 1, atol=1e-9)


def test_rfm_scores_range(tx):
    s = rfm.rfm_scores(rfm.compute_rfm(tx))
    assert s[["R_score", "F_score", "M_score"]].isin(range(1, 6)).all().all()


def test_find_elbow():
    assert clustering.find_elbow([1, 2, 3, 4, 5, 6], [100, 20, 15, 12, 10, 9]) == 2
    assert clustering.find_elbow([1, 2, 3, 4, 5, 6], [100, 80, 60, 15, 12, 10]) == 4


def test_choose_k_in_bounds(tx):
    X, _ = rfm.scale_rfm(rfm.compute_rfm(tx))
    scores = clustering.evaluate_k(X, range(2, 9))
    k, reason = clustering.choose_k(scores, 3, 8)
    assert 3 <= k <= 8 and "elbow" in reason


@pytest.mark.parametrize("k", [3, 4, 5, 6, 8, 10])
def test_segment_names_unique_and_known(tx, k):
    X, _ = rfm.scale_rfm(rfm.compute_rfm(tx))
    km = clustering.fit_kmeans(X, k)
    names = interpret.name_clusters(km.cluster_centers_)
    assert len(set(names.values())) == k
    assert all(interpret.base_name(n) in interpret.SEGMENT_PLAYBOOK for n in names.values())


def test_names_follow_recency():
    # cluster 0: very recent, high value; cluster 1: long gone, low value
    c = np.array([[-1.5, 1.5, 1.5], [1.5, -1.0, -1.0], [-0.8, -0.8, -0.8]])
    names = interpret.name_clusters(c)
    assert names[0] == "Champions"
    assert names[1] == "Hibernating / Lost"
    assert names[2] not in interpret.LAPSED


def test_full_pipeline_demo(tmp_path):
    cfg = Config(reports_dir=tmp_path / "reports", processed_dir=tmp_path / "processed", demo=True)
    out = run(cfg)
    assert (tmp_path / "reports" / "SEGMENTS.md").exists()
    assert len(list((tmp_path / "reports" / "figures").glob("*.png"))) == 9
    assert out["profile"]["Customers"].sum() == len(out["rfm"])
    assert abs(out["profile"]["Revenue_share_%"].sum() - 100) < 0.1
