"""Compare OCR engines on the human-confirmed Dataset 3 phrase crops.

Run from ``code/`` with the OCR extras installed:
    python experiments/k0_ocr_benchmark.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.io import read_image
from common.paths import RESULTS_DIR
from ocr.engines import ocr_line
from ocr.metrics import edit_distance, normalize_text, wer, corpus_cer


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tesseract_version() -> str | None:
    try:
        import pytesseract
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return None


def load_samples(dataset_dir: Path) -> list[dict]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    if len(samples) != 50:
        raise ValueError(f"Cần đúng 50 mẫu trong manifest, nhận {len(samples)}")
    invalid = [s.get("sample_id") for s in samples
               if s.get("review_status") != "human_confirmed" or not s.get("text_gt")]
    if invalid:
        raise ValueError(f"Mẫu chưa có transcript đã xác nhận: {invalid}")
    for sample in samples:
        crop_path = dataset_dir / sample["crop_path"]
        if not crop_path.is_file():
            raise FileNotFoundError(f"Thiếu crop {sample['sample_id']}: {crop_path}")
    return samples


def run_benchmark(dataset_dir: Path, engines: list[str], out_dir: Path,
                  lang: str = "eng") -> dict:
    samples = load_samples(dataset_dir)
    rows: list[dict] = []
    warmup_ms: dict[str, float] = {}
    for engine in engines:
        warmup_image = read_image(dataset_dir / samples[0]["crop_path"], gray=True)
        warmup_start = time.perf_counter()
        ocr_line(warmup_image, engine=engine, lang=lang)
        warmup_ms[engine] = (time.perf_counter() - warmup_start) * 1000
        for sample in samples:
            image = read_image(dataset_dir / sample["crop_path"], gray=True)
            start = time.perf_counter()
            prediction, confidence = ocr_line(image, engine=engine, lang=lang)
            elapsed_ms = (time.perf_counter() - start) * 1000
            gt_norm = normalize_text(sample["text_gt"])
            pred_norm = normalize_text(prediction)
            char_errors = edit_distance(gt_norm, pred_norm)
            gt_words, pred_words = gt_norm.split(), pred_norm.split()
            word_errors = edit_distance(gt_words, pred_words)
            rows.append({
                "sample_id": sample["sample_id"],
                "document_id": sample["document_id"],
                "region_type": sample["region_type"],
                "engine": engine,
                "text_gt": sample["text_gt"],
                "text_pred": prediction,
                "confidence": confidence,
                "char_errors": char_errors,
                "gt_characters": len(gt_norm),
                "sample_cer": char_errors / len(gt_norm) if gt_norm else (0.0 if not pred_norm else 1.0),
                "word_errors": word_errors,
                "gt_words": len(gt_words),
                "sample_wer": wer(gt_norm, pred_norm),
                "latency_ms": round(elapsed_ms, 3),
            })

    summaries = []
    for engine in engines:
        selected = [r for r in rows if r["engine"] == engine]
        refs = [r["text_gt"] for r in selected]
        hyps = [r["text_pred"] for r in selected]
        total_words = sum(r["gt_words"] for r in selected)
        summaries.append({
            "engine": engine,
            "sample_count": len(selected),
            "document_count": len({r["document_id"] for r in selected}),
            "corpus_cer": corpus_cer(refs, hyps),
            "corpus_wer": sum(r["word_errors"] for r in selected) / total_words if total_words else 0.0,
            "mean_sample_cer": statistics.mean(r["sample_cer"] for r in selected),
            "mean_confidence": statistics.mean(r["confidence"] for r in selected),
            "warmup_ms_excluded": round(warmup_ms[engine], 3),
            "mean_latency_ms": statistics.mean(r["latency_ms"] for r in selected),
            "median_latency_ms": statistics.median(r["latency_ms"] for r in selected),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "dataset": str(dataset_dir.relative_to(REPO_DIR)) if dataset_dir.is_relative_to(REPO_DIR) else str(dataset_dir),
        "dataset_description": "Dataset 3 phrase crops; 50 human-confirmed transcripts, 15 source documents",
        "normalization": "Unicode NFC, collapse whitespace, preserve case and punctuation",
        "crop_preprocessing": "original crops, grayscale decode only; no resize/enhancement",
        "ocr_mode": "ocr_line; Tesseract PSM 7, PaddleOCR line recognition",
        "language": lang,
        "model_instance_cache": "PaddleOCR/VietOCR model instances are cached; OCR results are not cached",
        "latency_method": "One warm-up inference per engine is excluded; per-sample latency is measured on 50 crops",
        "software": {name: package_version(name) for name in ("pytesseract", "paddleocr", "paddlepaddle", "Pillow", "opencv-python")},
        "tesseract_binary_version": tesseract_version(),
        "summaries": summaries,
        "by_region_type": [],
    }
    for engine in engines:
        for region in sorted({r["region_type"] for r in rows}):
            subset = [r for r in rows if r["engine"] == engine and r["region_type"] == region]
            payload["by_region_type"].append({
                "engine": engine,
                "region_type": region,
                "sample_count": len(subset),
                "corpus_cer": corpus_cer([r["text_gt"] for r in subset], [r["text_pred"] for r in subset]),
                "mean_latency_ms": statistics.mean(r["latency_ms"] for r in subset),
            })
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Tesseract/Paddle trên 50 crop đã duyệt")
    parser.add_argument("--dataset", type=Path, default=REPO_DIR / "data/dataset2_50")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "ocr_benchmark_dataset3_50")
    parser.add_argument("--engines", nargs="+", choices=("tesseract", "paddle"),
                        default=("tesseract", "paddle"))
    parser.add_argument("--lang", default="eng", help="Ngôn ngữ Tesseract; Paddle chuẩn hóa eng thành en")
    args = parser.parse_args()
    payload = run_benchmark(args.dataset.resolve(), list(args.engines), args.out.resolve(),
                            lang=args.lang)
    for item in payload["summaries"]:
        print(f"{item['engine']}: CER={item['corpus_cer']:.4f}, WER={item['corpus_wer']:.4f}, "
              f"mean={item['mean_latency_ms']:.1f} ms, n={item['sample_count']}")
    print(f"Kết quả: {args.out.resolve()}")


if __name__ == "__main__":
    main()
