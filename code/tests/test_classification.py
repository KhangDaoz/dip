"""① Phân loại 3 lớp."""
import numpy as np

from classification import classify, features
from common.testing import todo


@todo
def test_extract_all_is_vector(page_gray):
    assert features.extract_all(page_gray).ndim == 1


@todo
def test_knn_rejects_far_sample():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (20, 4)), rng.normal(5, 1, (20, 4))])
    y = np.array(["prescription"] * 20 + ["lab_report"] * 20)
    model = classify.train_knn(X, y, k=3)
    labels, _ = classify.predict_with_reject(model, np.array([[100.0] * 4]))
    assert labels[0] == "other"
