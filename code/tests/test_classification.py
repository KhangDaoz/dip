"""① Phân loại 3 lớp (Hiếu)."""
import numpy as np
import pytest

from classification import classify, features
from common.testing import todo


@pytest.mark.parametrize("fn, needs_binary", [
    ("hog_features", False), ("lbp_hist", False), ("layout_density", True),
    ("line_ratio", True), ("text_block_stats", True),
])
@todo
def test_feature_is_finite_vector(page_gray, text_mask, fn, needs_binary):
    vec = getattr(features, fn)(text_mask if needs_binary else page_gray)
    assert vec.ndim == 1 and len(vec) > 0 and np.isfinite(vec).all()


@todo
def test_predict_with_reject_threshold():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.1, (20, 5)) for c in (0, 3, 6)])
    y = np.repeat(["prescription", "lab_report", "other"], 20)
    model = classify.train_knn(X, y, k=3)
    labels, conf = classify.predict_with_reject(model, X, threshold=0.0)
    assert (np.asarray(labels) == y).mean() > 0.95
    labels, _ = classify.predict_with_reject(model, X, threshold=1.01)
    assert all(lbl == classify.REJECT for lbl in labels)
