"""H11 phân loại kNN + từ chối tài liệu lạ (KeHoach.md mục 5.1; C8 · S37–42)."""
import numpy as np


def choose_k_cv(X: np.ndarray, y: np.ndarray, ks=(1, 3, 5, 7), metrics=("euclidean", "cosine", "chi2"),
                folds: int = 5) -> dict:
    """Chọn k lẻ và độ đo khoảng cách bằng 5-fold trên train (C8 · S39–41)."""
    raise NotImplementedError("choose_k_cv")


def train_knn(X: np.ndarray, y: np.ndarray, k: int = 5, metric: str = "euclidean", reject_percentile: float = 95):
    """Chuẩn hóa z-score (C8 · S40), lưu tập mẫu và ngưỡng từ chối theo phân vị khoảng cách."""
    raise NotImplementedError("train_knn")


def predict_with_reject(model, X: np.ndarray) -> tuple:
    """Trả về (nhãn, độ tin cậy); khoảng cách láng giềng gần nhất > ngưỡng thì nhãn "other"."""
    raise NotImplementedError("predict_with_reject")
