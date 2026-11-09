"""H11 phân loại kNN + từ chối tài liệu lạ (KeHoach.md mục 5.1; C8 · S37–42)."""
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

# Mô hình dùng cho pipeline: do experiments/d10_classify.py huấn luyện và lưu (có commit, vài MB)
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "doc_knn.pkl"


def save_model(model, path: Path = DEFAULT_MODEL) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: Path = DEFAULT_MODEL):
    """Nạp mô hình đã huấn luyện; chưa có thì báo chưa làm (run_pipeline sẽ yêu cầu --doc-type)."""
    path = Path(path)
    if not path.exists():
        raise NotImplementedError(f"chưa có mô hình phân loại {path.name}: chạy experiments/d10_classify.py")
    with open(path, "rb") as f:
        return pickle.load(f)


def _distances(A: np.ndarray, B: np.ndarray, metric: str) -> np.ndarray:
    """Ma trận khoảng cách |A| × |B| theo C8 · S40: Euclid, Manhattan, Cosine, χ²."""
    if metric == "euclidean":
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.sqrt(np.maximum(d2, 0))
    if metric == "manhattan":
        return np.abs(A[:, None, :] - B[None, :, :]).sum(-1)
    if metric == "cosine":
        na = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
        nb = np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
        return 1 - (A / na) @ (B / nb).T
    if metric == "chi2":   # ½ Σ (x−y)² / (x+y), cần đặc trưng không âm (min-max)
        out = np.empty((len(A), len(B)))
        for i, a in enumerate(A):
            s = a[None, :] + B
            out[i] = 0.5 * np.where(s > 0, (a[None, :] - B) ** 2 / np.where(s > 0, s, 1), 0).sum(1)
        return out
    raise ValueError(metric)


class KNNModel:
    """Lưu mẫu huấn luyện đã chuẩn hóa (kNN không học tham số, C8 · S42: huấn luyện O(1))."""

    def __init__(self, X, y, k, metric, reject_percentile):
        self.k, self.metric = int(k), metric
        self.y = np.asarray(y)
        X = np.asarray(X, dtype=np.float64)
        if metric == "chi2":   # min-max về [0, 1] (C8 · S40)
            self.lo, self.hi = X.min(0), X.max(0)
            self.scale = np.where(self.hi > self.lo, self.hi - self.lo, 1)
        else:                  # z-score (C8 · S40)
            self.lo, self.scale = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1)
        self.X = self.transform(X).astype(np.float32)   # float32 cho file mô hình nhỏ
        # ngưỡng từ chối: phân vị khoảng cách tới láng giềng gần nhất (bỏ chính nó) trong tập huấn luyện
        D = self._dist(self.X)
        np.fill_diagonal(D, np.inf)
        self.reject_threshold = float(np.percentile(D.min(1), reject_percentile))

    def transform(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.lo) / self.scale
        return np.clip(Z, 0, None) if self.metric == "chi2" else Z

    def _dist(self, Z, chunk: int = 256):
        return np.vstack([_distances(Z[i:i + chunk], self.X, self.metric) for i in range(0, len(Z), chunk)])

    def neighbors(self, X):
        D = self._dist(self.transform(X))
        idx = np.argsort(D, axis=1)[:, :self.k]
        return D, idx


def train_knn(X: np.ndarray, y: np.ndarray, k: int = 5, metric: str = "euclidean", reject_percentile: float = 95):
    """Chuẩn hóa z-score (C8 · S40), lưu tập mẫu và ngưỡng từ chối theo phân vị khoảng cách."""
    return KNNModel(X, y, k, metric, reject_percentile)


def predict_with_reject(model: KNNModel, X: np.ndarray, reject: bool = True) -> tuple:
    """Trả về (nhãn, độ tin cậy); khoảng cách láng giềng gần nhất > ngưỡng thì nhãn "other".

    Bỏ phiếu đa số trong k láng giềng (C8 · S37); độ tin cậy = tỉ lệ phiếu của lớp thắng.
    """
    D, idx = model.neighbors(np.atleast_2d(X))
    labels, confs = [], []
    for row, nn in zip(D, idx):
        votes = Counter(model.y[nn])
        lab, n = votes.most_common(1)[0]
        if reject and row[nn[0]] > model.reject_threshold:
            lab, n = "other", model.k
        labels.append(lab)
        confs.append(n / model.k)
    return np.array(labels), np.array(confs)


def macro_f1(y_true, y_pred) -> float:
    """Trung bình F1 các lớp (C8 · S49) — dữ liệu lệch lớp nên không dùng accuracy."""
    f1s = []
    for c in np.unique(y_true):
        tp = np.sum((y_pred == c) & (y_true == c))
        p = tp / max(np.sum(y_pred == c), 1)
        r = tp / max(np.sum(y_true == c), 1)
        f1s.append(0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s))


def choose_k_cv(X: np.ndarray, y: np.ndarray, ks=(1, 3, 5, 7), metrics=("euclidean", "cosine", "chi2"),
                folds: int = 5, seed: int = 42) -> dict:
    """Chọn k lẻ và độ đo khoảng cách bằng 5-fold trên train (C8 · S39–41). Trả về {"k", "metric", "scores"}."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), int)
    for c in np.unique(y):           # chia phân tầng: mỗi fold giữ tỉ lệ lớp
        idx = rng.permutation(np.nonzero(y == c)[0])
        fold_of[idx] = np.arange(len(idx)) % folds
    scores = {}
    for metric in metrics:
        for k in ks:
            s = []
            for f in range(folds):
                tr, va = fold_of != f, fold_of == f
                m = train_knn(X[tr], y[tr], k=k, metric=metric)
                pred, _ = predict_with_reject(m, X[va], reject=False)
                s.append(macro_f1(y[va], pred))
            scores[(metric, k)] = float(np.mean(s))
    (metric, k), _ = max(scores.items(), key=lambda kv: kv[1])
    return {"k": k, "metric": metric, "scores": {f"{m}_k{kk}": v for (m, kk), v in scores.items()}}
