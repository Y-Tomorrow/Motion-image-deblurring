"""
批量图像 / 视频去模糊处理（HAP-Deblur / NRC-HybridNet DL）。

示例：
  python process_media.py --method hap --input blur_dir --output out --workers 4
  python process_media.py --method dl --input blur_dir --output out
  python process_media.py --method dl --input video.mp4 --output out.mp4 --max-frames 100
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from deblur_wiener import deblur_image, temporal_fuse as wiener_temporal_fuse

try:
    from deblur_dl import deblur_image_dl, temporal_ema as dl_temporal_ema
except ImportError:
    deblur_image_dl = None
    dl_temporal_ema = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"failed to read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _resolve_io(input_path: Path, output_path: Path) -> tuple[list[Path], callable]:
    if input_path.is_file():
        return [input_path], (lambda f: output_path if output_path.suffix else output_path / f.name)
    files = collect_images(input_path)
    return files, (lambda f: output_path / f.relative_to(input_path))


def _process_one_hap(args_tuple: tuple) -> tuple[str, int, float]:
    in_path, out_path, rl_iter, estimate, k_len, k_ang, super_resolve = args_tuple
    img = load_rgb(Path(in_path))
    result = deblur_image(
        img,
        estimate_kernel=estimate,
        kernel_length=k_len,
        kernel_angle=k_ang,
        rl_iterations=rl_iter,
        super_resolve=super_resolve,
    )
    save_rgb(Path(out_path), result.image)
    return str(out_path), result.kernel_length, result.kernel_angle


def process_image_batch_hap(
    input_path: Path,
    output_path: Path,
    *,
    workers: int = 1,
    rl_iter: int = 12,
    estimate: bool = True,
    kernel_length: int = 15,
    kernel_angle: float = 0.0,
    super_resolve: bool = False,
) -> None:
    files, dst_fn = _resolve_io(input_path, output_path)
    if not files:
        raise RuntimeError(f"no images under {input_path}")

    jobs = [
        (str(f), str(dst_fn(f) if input_path.is_file() else output_path / f.relative_to(input_path)),
         rl_iter, estimate, kernel_length, kernel_angle, super_resolve)
        for f in files
    ]
    if input_path.is_file():
        jobs = [(str(files[0]), str(output_path), rl_iter, estimate, kernel_length, kernel_angle, super_resolve)]

    t0 = time.perf_counter()
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            out, k_len, k_ang = _process_one_hap(job)
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] saved {out}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_one_hap, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                out, _, _ = fut.result()
                if i % 20 == 0 or i == len(jobs):
                    print(f"[{i}/{len(jobs)}] saved {out}")
    print(f"hap batch done: {len(jobs)} images in {time.perf_counter() - t0:.2f} s")


def process_image_batch_dl(input_path: Path, output_path: Path, *, use_tiles: bool = True) -> None:
    if deblur_image_dl is None:
        raise RuntimeError("DL 模式需要 yolov8 环境: D:\\anaconda3\\envs\\yolov8\\python.exe")
    files = [input_path] if input_path.is_file() else collect_images(input_path)
    if not files:
        raise RuntimeError(f"no images under {input_path}")

    t0 = time.perf_counter()
    for i, f in enumerate(files, 1):
        dst = output_path if input_path.is_file() else output_path / f.relative_to(input_path)
        result = deblur_image_dl(load_rgb(f), use_tiles=use_tiles)
        save_rgb(dst, result.image)
        if i % 10 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] saved {dst}  conf={result.confidence:.3f}")
    print(f"dl batch done: {len(files)} images in {time.perf_counter() - t0:.2f} s")


def process_video_hap(
    input_path: Path,
    output_path: Path,
    *,
    rl_iter: int = 12,
    estimate: bool = True,
    kernel_length: int = 15,
    kernel_angle: float = 0.0,
    super_resolve: bool = False,
    reuse_kernel: bool = True,
    max_frames: int = 0,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    shared_kernel = None
    shared_len, shared_ang = kernel_length, kernel_angle
    recent: list[np.ndarray] = []
    frame_idx = 0
    t0 = time.perf_counter()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if reuse_kernel and shared_kernel is not None:
            result = deblur_image(
                frame, kernel=shared_kernel, kernel_length=shared_len, kernel_angle=shared_ang,
                estimate_kernel=False, rl_iterations=rl_iter, super_resolve=super_resolve,
            )
        else:
            result = deblur_image(
                frame, estimate_kernel=estimate, kernel_length=kernel_length,
                kernel_angle=kernel_angle, rl_iterations=rl_iter, super_resolve=super_resolve,
            )
            if reuse_kernel and shared_kernel is None:
                shared_kernel = result.kernel
                shared_len, shared_ang = result.kernel_length, result.kernel_angle
                print(f"hap kernel locked: len={shared_len} ang={shared_ang:.0f}")

        recent.append(result.image)
        if len(recent) > 3:
            recent.pop(0)
        writer.write(cv2.cvtColor(wiener_temporal_fuse(recent), cv2.COLOR_RGB2BGR))
        if frame_idx % 30 == 0:
            print(f"hap video frame {frame_idx}")

    cap.release()
    writer.release()
    print(f"hap video done: {frame_idx} frames in {time.perf_counter() - t0:.2f} s")


def process_video_dl(
    input_path: Path,
    output_path: Path,
    *,
    use_tiles: bool = True,
    max_frames: int = 0,
    ema_beta: float = 0.8,
) -> None:
    if deblur_image_dl is None:
        raise RuntimeError("DL 模式需要 yolov8 环境")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    prev: np.ndarray | None = None
    frame_idx = 0
    t0 = time.perf_counter()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = deblur_image_dl(frame, use_tiles=use_tiles)
        fused = dl_temporal_ema(prev, result.image, beta=ema_beta)
        prev = fused
        writer.write(cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))

        if frame_idx == 1 or frame_idx % 10 == 0:
            elapsed = time.perf_counter() - t0
            eta = (elapsed / frame_idx) * (total - frame_idx) if total > frame_idx else 0
            print(f"dl frame {frame_idx}/{total or '?'}  conf={result.confidence:.3f}  eta={eta:.0f}s")

    cap.release()
    writer.release()
    print(f"dl video done: {frame_idx} frames in {time.perf_counter() - t0:.2f} s -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量图像/视频去模糊")
    parser.add_argument("--method", choices=("hap", "dl"), default="hap", help="hap=传统改进组, dl=深度学习")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rl-iter", type=int, default=12)
    parser.add_argument("--no-estimate", action="store_true")
    parser.add_argument("--kernel-length", type=int, default=15)
    parser.add_argument("--kernel-angle", type=float, default=0.0)
    parser.add_argument("--super-resolve", action="store_true")
    parser.add_argument("--reuse-kernel", action="store_true", default=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-tiles", action="store_true", help="DL：禁用分块推理")
    args = parser.parse_args()

    is_video = args.input.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    if args.method == "dl":
        if is_video:
            process_video_dl(args.input, args.output, use_tiles=not args.no_tiles, max_frames=args.max_frames)
        else:
            process_image_batch_dl(args.input, args.output, use_tiles=not args.no_tiles)
    elif is_video:
        process_video_hap(
            args.input, args.output, rl_iter=args.rl_iter, estimate=not args.no_estimate,
            kernel_length=args.kernel_length, kernel_angle=args.kernel_angle,
            super_resolve=args.super_resolve, reuse_kernel=args.reuse_kernel, max_frames=args.max_frames,
        )
    else:
        process_image_batch_hap(
            args.input, args.output, workers=args.workers, rl_iter=args.rl_iter,
            estimate=not args.no_estimate, kernel_length=args.kernel_length,
            kernel_angle=args.kernel_angle, super_resolve=args.super_resolve,
        )


if __name__ == "__main__":
    main()
