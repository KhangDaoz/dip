"""Phân loại 3 lớp có ngưỡng từ chối."""
import numpy as np

REJECT = "reject"   # không chắc → lưu làm tệp đính kèm, không đưa vào pipeline


def train_knn(X: np.ndarray, y: np.ndarray, k: int = 5, metric: str = "euclidean"):
    """Huấn luyện k-NN (kèm chuẩn hóa / PCA nếu cần). Trả về mô hình sklearn."""
    raise NotImplementedError("train_knn")


def predict_with_reject(model, X: np.ndarray, threshold: float = 0.6):
    """Trả về (labels, confidences); nhãn = REJECT khi xác suất cao nhất < threshold."""
    raise NotImplementedError("predict_with_reject")
