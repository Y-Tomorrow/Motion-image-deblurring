"""
深度学习组 NRC-HybridNet：MIMO-UNet + 无参考置信度融合，GOPRO 评估 PSNR/SSIM。

推荐使用 yolov8 环境（含 PyTorch + CUDA）：
  D:\\anaconda3\\envs\\yolov8\\python.exe evaluate_dl_deblur.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deblur_dl import deblur_image_dl


def collect_pairs(data_root: Path) -> list[tuple[Path, Path, str]]:
    train_dir = data_root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"train dir not found: {train_dir}")
    pairs: list[tuple[Path, Path, str]] = []
    for seq_dir in sorted(train_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        blur_dir, sharp_dir = seq_dir / "blur_gamma", seq_dir / "sharp"
        if not blur_dir.is_dir() or not sharp_dir.is_dir():
            continue
        for blur_path in sorted(blur_dir.glob("*.png")):
            sharp_path = sharp_dir / blur_path.name
            if sharp_path.is_file():
                pairs.append((blur_path, sharp_path, seq_dir.name))
    if not pairs:
        raise RuntimeError(f"no pairs under {train_dir}")
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"failed to read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_psnr(restored: np.ndarray, sharp: np.ndarray) -> float:
    if restored.shape != sharp.shape:
        sharp = cv2.resize(sharp, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_AREA)
    mse = np.mean((sharp.astype(np.float64) - restored.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf")


def compute_ssim(restored: np.ndarray, sharp: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        if restored.shape != sharp.shape:
            sharp = cv2.resize(sharp, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_AREA)
        return float(structural_similarity(sharp, restored, channel_axis=2, data_range=255))
    except ImportError:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="NRC-HybridNet DL deblur evaluation")
    parser.add_argument("--data-root", type=Path, default=ROOT / "GOPRO_Large")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "dl_nrc_hybridnet.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-tiles", action="store_true", help="禁用分块推理")
    parser.add_argument("--raw-dl", action="store_true", help="仅 DL 输出，不做 NR 融合")
    args = parser.parse_args()

    pairs = collect_pairs(args.data_root)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.perf_counter()

    for i, (blur_path, sharp_path, seq_name) in enumerate(pairs, 1):
        blur = load_rgb(blur_path)
        sharp = load_rgb(sharp_path)
        t_img = time.perf_counter()
        result = deblur_image_dl(blur, use_tiles=not args.no_tiles, skip_fusion=args.raw_dl)
        psnr = compute_psnr(result.image, sharp)
        ssim = compute_ssim(result.image, sharp)
        time_ms = (time.perf_counter() - t_img) * 1000
        rows.append(
            {
                "sequence": seq_name,
                "filename": blur_path.name,
                "blend": result.blend,
                "confidence": result.confidence,
                "psnr": psnr,
                "ssim": ssim,
                "time_ms": time_ms,
            }
        )
        if i % 10 == 0 or i == len(pairs):
            print(
                f"[{i}/{len(pairs)}] {seq_name}/{blur_path.name}  "
                f"conf={result.confidence:.3f} blend={result.blend:.2f}  "
                f"PSNR={psnr:.4f}  SSIM={ssim:.4f}  time={time_ms:.0f} ms"
            )

    elapsed = time.perf_counter() - t0
    fields = ["sequence", "filename", "blend", "confidence", "psnr", "ssim", "time_ms"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n========== NRC-HybridNet vs sharp (GT) ==========")
    print(f"count: {len(rows)}")
    print(f"mean PSNR: {np.nanmean([r['psnr'] for r in rows]):.4f} dB")
    print(f"mean SSIM: {np.nanmean([r['ssim'] for r in rows]):.4f}")
    print(f"mean time: {np.mean([r['time_ms'] for r in rows]):.2f} ms/img")
    print(f"total time: {elapsed:.2f} s")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
