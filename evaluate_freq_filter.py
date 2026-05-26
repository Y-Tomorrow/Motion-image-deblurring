"""
频域滤波组：对 train 集 blur_gamma 做 FFT 理想/巴特沃斯低通、高通滤波，
相对 sharp（GT）计算 PSNR / SSIM 与耗时。
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
from freq_filters import FILTER_CHOICES, filter_image


def compute_metrics(restored: np.ndarray, sharp: np.ndarray) -> tuple[float, float]:
    if restored.shape != sharp.shape:
        sharp = cv2.resize(sharp, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_AREA)
    psnr = peak_signal_noise_ratio(sharp, restored, data_range=255)
    ssim = structural_similarity(sharp, restored, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def save_demo(
    blur: np.ndarray,
    restored: np.ndarray,
    sharp: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    h, w = blur.shape[:2]
    gap = 8
    canvas = np.ones((h, w * 3 + gap * 2, 3), dtype=np.uint8) * 255
    canvas[:, :w] = blur
    canvas[:, w + gap : 2 * w + gap] = restored
    canvas[:, 2 * w + 2 * gap : 3 * w + 2 * gap] = sharp
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"demo saved: {out_path} ({title})")


def evaluate_one_filter(
    pairs: list,
    kind: str,
    cutoff: float,
    order: int,
    save_demo_path: Path | None,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    h_cache: dict = {}
    demo_saved = False

    for i, (blur_path, sharp_path, seq_name) in enumerate(pairs, 1):
        blur = load_rgb(blur_path)
        sharp = load_rgb(sharp_path)

        t_img = time.perf_counter()
        restored = filter_image(blur, kind, cutoff, order, h_cache)
        psnr, ssim = compute_metrics(restored, sharp)
        time_ms = (time.perf_counter() - t_img) * 1000

        rows.append(
            {
                "sequence": seq_name,
                "filename": blur_path.name,
                "filter": kind,
                "cutoff": cutoff,
                "order": order,
                "psnr": psnr,
                "ssim": ssim,
                "time_ms": time_ms,
            }
        )

        if save_demo_path and not demo_saved:
            save_demo(blur, restored, sharp, save_demo_path, kind)
            demo_saved = True

        if i % 100 == 0 or i == len(pairs):
            print(
                f"[{kind}] [{i}/{len(pairs)}] {seq_name}/{blur_path.name}  "
                f"PSNR={psnr:.4f}  SSIM={ssim:.4f}  time={time_ms:.2f} ms"
            )

    psnr_vals = [r["psnr"] for r in rows]
    ssim_vals = [r["ssim"] for r in rows]
    time_vals = [r["time_ms"] for r in rows]
    summary = {
        "filter": kind,
        "cutoff": cutoff,
        "order": order,
        "count": len(rows),
        "mean_psnr": float(np.mean(psnr_vals)),
        "mean_ssim": float(np.mean(ssim_vals)),
        "mean_time_ms": float(np.mean(time_vals)),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="GOPRO blur_gamma FFT 频域滤波 PSNR/SSIM 评估")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "GOPRO_Large",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--filter",
        choices=[*FILTER_CHOICES, "all"],
        default="all",
        help="滤波器类型，all 表示运行全部四种",
    )
    parser.add_argument("--cutoff", type=float, default=30.0, help="截止频率 D0（像素）")
    parser.add_argument("--order", type=int, default=2, help="巴特沃斯滤波器阶数 n")
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 张（0 表示全部）")
    parser.add_argument(
        "--save-demo",
        action="store_true",
        help="保存首张图 blur | filtered | sharp 对比到 image/ 目录",
    )
    args = parser.parse_args()

    pairs = collect_pairs(args.data_root)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kinds = list(FILTER_CHOICES) if args.filter == "all" else [args.filter]

    all_rows: list[dict] = []
    summaries: list[dict] = []
    t0 = time.perf_counter()

    for kind in kinds:
        demo_path = None
        if args.save_demo:
            demo_path = Path(__file__).resolve().parent / "image" / f"freq_{kind}.png"

        rows, summary = evaluate_one_filter(pairs, kind, args.cutoff, args.order, demo_path)
        all_rows.extend(rows)
        summaries.append(summary)

        out_csv = args.output_dir / f"freq_{kind}.csv"
        fieldnames = ["sequence", "filename", "filter", "cutoff", "order", "psnr", "ssim", "time_ms"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved: {out_csv}")

    elapsed = time.perf_counter() - t0

    summary_csv = args.output_dir / "freq_filter_summary.csv"
    summary_fields = ["filter", "cutoff", "order", "count", "mean_psnr", "mean_ssim", "mean_time_ms"]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\n========== FFT freq filter summary (cutoff={args.cutoff}, order={args.order}) ==========")
    for s in summaries:
        print(
            f"{s['filter']:20s}  PSNR={s['mean_psnr']:.4f} dB  "
            f"SSIM={s['mean_ssim']:.4f}  time={s['mean_time_ms']:.2f} ms/img"
        )
    print(f"total time: {elapsed:.2f} s")
    print(f"summary saved: {summary_csv}")


if __name__ == "__main__":
    main()
