"""
改进组 HAP-Deblur：序列级共享模糊核 + 自适应融合，在 GOPRO train 集评估 PSNR/SSIM。
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from deblur_wiener import deblur_image, estimate_motion_kernel
from evaluate_baseline import collect_pairs, load_rgb


def compute_metrics(restored: np.ndarray, sharp: np.ndarray) -> tuple[float, float]:
    if restored.shape != sharp.shape:
        sharp = cv2.resize(sharp, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_AREA)
    psnr = peak_signal_noise_ratio(sharp, restored, data_range=255)
    ssim = structural_similarity(sharp, restored, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def group_pairs(pairs: list) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for item in pairs:
        grouped[item[2]].append(item)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="GOPRO HAP-Deblur PSNR/SSIM 评估")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "GOPRO_Large",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "hap_deblur.csv",
    )
    parser.add_argument("--rl-iter", type=int, default=12)
    parser.add_argument("--no-sequence-kernel", action="store_true", help="每张图单独估计模糊核")
    parser.add_argument("--super-resolve", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pairs = collect_pairs(args.data_root)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.perf_counter()
    idx = 0

    if args.no_sequence_kernel:
        work_items = [(None, p) for p in pairs]
    else:
        work_items = []
        for seq_name, seq_pairs in group_pairs(pairs).items():
            first_blur = load_rgb(seq_pairs[0][0])
            kernel, k_len, k_angle = estimate_motion_kernel(first_blur, search_iter=6)
            print(f"sequence {seq_name}: shared kernel len={k_len} angle={k_angle:.0f}")
            for pair in seq_pairs:
                work_items.append(((kernel, k_len, k_angle), pair))

    total = len(work_items)
    for shared, (blur_path, sharp_path, seq_name) in work_items:
        idx += 1
        blur = load_rgb(blur_path)
        sharp = load_rgb(sharp_path)

        t_img = time.perf_counter()
        if shared is None:
            result = deblur_image(
                blur,
                estimate_kernel=True,
                rl_iterations=args.rl_iter,
                super_resolve=args.super_resolve,
            )
        else:
            kernel, k_len, k_angle = shared
            result = deblur_image(
                blur,
                kernel=kernel,
                kernel_length=k_len,
                kernel_angle=k_angle,
                estimate_kernel=False,
                rl_iterations=args.rl_iter,
                super_resolve=args.super_resolve,
            )
        psnr, ssim = compute_metrics(result.image, sharp)
        time_ms = (time.perf_counter() - t_img) * 1000

        rows.append(
            {
                "sequence": seq_name,
                "filename": blur_path.name,
                "kernel_length": result.kernel_length,
                "kernel_angle": result.kernel_angle,
                "blend": result.blend,
                "psnr": psnr,
                "ssim": ssim,
                "time_ms": time_ms,
            }
        )
        if idx % 50 == 0 or idx == total:
            print(
                f"[{idx}/{total}] {seq_name}/{blur_path.name}  "
                f"blend={result.blend:.2f}  PSNR={psnr:.4f}  SSIM={ssim:.4f}  time={time_ms:.0f} ms"
            )

    elapsed = time.perf_counter() - t0
    fieldnames = ["sequence", "filename", "kernel_length", "kernel_angle", "blend", "psnr", "ssim", "time_ms"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n========== HAP-Deblur vs sharp (GT) ==========")
    print(f"count: {len(rows)}")
    print(f"mean PSNR: {np.mean([r['psnr'] for r in rows]):.4f} dB")
    print(f"mean SSIM: {np.mean([r['ssim'] for r in rows]):.4f}")
    print(f"mean time: {np.mean([r['time_ms'] for r in rows]):.2f} ms/img")
    print(f"total time: {elapsed:.2f} s")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
