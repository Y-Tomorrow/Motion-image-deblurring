"""
基础组：对 train 集 blur_gamma 做中值滤波，相对 sharp（GT）计算 PSNR / SSIM 与耗时。
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from evaluate_baseline import collect_pairs, load_rgb


def median_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(img, k)


def compute_metrics(restored: np.ndarray, sharp: np.ndarray) -> tuple[float, float]:
    if restored.shape != sharp.shape:
        sharp = cv2.resize(sharp, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_AREA)
    psnr = peak_signal_noise_ratio(sharp, restored, data_range=255)
    ssim = structural_similarity(sharp, restored, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def main() -> None:
    parser = argparse.ArgumentParser(description="GOPRO train blur_gamma 中值滤波 PSNR/SSIM 评估")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "GOPRO_Large",
        help="GOPRO_Large 根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "median_filter.csv",
        help="逐张结果 CSV 路径",
    )
    parser.add_argument("--ksize", type=int, default=5, help="中值滤波核大小（奇数，默认 5）")
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 张（0 表示全部）")
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
        restored = median_filter(blur, args.ksize)
        psnr, ssim = compute_metrics(restored, sharp)
        time_ms = (time.perf_counter() - t_img) * 1000

        rows.append(
            {
                "sequence": seq_name,
                "filename": blur_path.name,
                "psnr": psnr,
                "ssim": ssim,
                "time_ms": time_ms,
            }
        )
        if i % 100 == 0 or i == len(pairs):
            print(
                f"[{i}/{len(pairs)}] {seq_name}/{blur_path.name}  "
                f"PSNR={psnr:.4f}  SSIM={ssim:.4f}  time={time_ms:.2f} ms"
            )

    elapsed = time.perf_counter() - t0
    psnr_vals = [r["psnr"] for r in rows]
    ssim_vals = [r["ssim"] for r in rows]
    time_vals = [r["time_ms"] for r in rows]

    fieldnames = ["sequence", "filename", "psnr", "ssim", "time_ms"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n========== median filter (ksize={args.ksize}) vs sharp (GT) ==========")
    print(f"count: {len(rows)}")
    print(f"mean PSNR: {np.mean(psnr_vals):.4f} dB")
    print(f"mean SSIM: {np.mean(ssim_vals):.4f}")
    print(f"mean time: {np.mean(time_vals):.2f} ms/img")
    print(f"total time: {elapsed:.2f} s")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
