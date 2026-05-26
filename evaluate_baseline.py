"""
对照组：以 train 集 blur_gamma（原始模糊输入）相对 sharp（GT）计算 PSNR / SSIM。
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def collect_pairs(data_root: Path) -> list[tuple[Path, Path, str]]:
    """返回 (blur_gamma_path, sharp_path, sequence_name) 列表。"""
    train_dir = data_root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"train dir not found: {train_dir}")

    pairs: list[tuple[Path, Path, str]] = []
    for seq_dir in sorted(train_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        blur_dir = seq_dir / "blur_gamma"
        sharp_dir = seq_dir / "sharp"
        if not blur_dir.is_dir() or not sharp_dir.is_dir():
            continue
        for blur_path in sorted(blur_dir.glob("*.png")):
            sharp_path = sharp_dir / blur_path.name
            if sharp_path.is_file():
                pairs.append((blur_path, sharp_path, seq_dir.name))
    if not pairs:
        raise RuntimeError(f"no blur_gamma/sharp pairs under {train_dir}")
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_metrics(blur: np.ndarray, sharp: np.ndarray) -> tuple[float, float]:
    if blur.shape != sharp.shape:
        sharp = cv2.resize(sharp, (blur.shape[1], blur.shape[0]), interpolation=cv2.INTER_AREA)
    psnr = peak_signal_noise_ratio(sharp, blur, data_range=255)
    ssim = structural_similarity(sharp, blur, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def main() -> None:
    parser = argparse.ArgumentParser(description="GOPRO train blur_gamma 对照组 PSNR/SSIM 评估")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "GOPRO_Large",
        help="GOPRO_Large 根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "baseline_blur_gamma.csv",
        help="逐张结果 CSV 路径",
    )
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
        psnr, ssim = compute_metrics(blur, sharp)
        rows.append(
            {
                "sequence": seq_name,
                "filename": blur_path.name,
                "psnr": psnr,
                "ssim": ssim,
            }
        )
        if i % 100 == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] {seq_name}/{blur_path.name}  PSNR={psnr:.4f}  SSIM={ssim:.4f}")

    elapsed = time.perf_counter() - t0
    psnr_vals = [r["psnr"] for r in rows]
    ssim_vals = [r["ssim"] for r in rows]

    fieldnames = ["sequence", "filename", "psnr", "ssim"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== baseline: blur_gamma (input) vs sharp (GT) ==========")
    print(f"count: {len(rows)}")
    print(f"mean PSNR: {np.mean(psnr_vals):.4f} dB")
    print(f"mean SSIM: {np.mean(ssim_vals):.4f}")
    print(f"time: {elapsed:.2f} s  ({elapsed / len(rows) * 1000:.2f} ms/img)")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
