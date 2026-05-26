"""
空频结合自适应去模糊（HAP-Deblur）：
1. 频谱方向估计 + 轻量模糊核搜索（序列/视频内共享核）
2. Richardson-Lucy 频域反卷积 + 自适应原图融合（抑制错误核带来的伪影）
3. 空域双边滤波 + Unsharp 细节增强
4. 锐度回退：去模糊无效时退化为边缘增强
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage import img_as_float
from skimage.restoration import richardson_lucy


@dataclass
class DeblurResult:
    image: np.ndarray
    kernel_length: int
    kernel_angle: float
    kernel: np.ndarray
    blend: float


def motion_psf(length: int, angle_deg: float) -> np.ndarray:
    size = max(int(length), 3)
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    for offset in range(-center, center + 1):
        x = int(round(center + offset * cos_a))
        y = int(round(center + offset * sin_a))
        if 0 <= y < size and 0 <= x < size:
            kernel[y, x] = 1.0
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def sharpness_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_angle_from_spectrum(gray: np.ndarray) -> float:
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    ps = np.log1p(np.abs(f))
    h, w = ps.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx) - 1
    best_angle, best_score = 0.0, -1.0
    for angle in range(0, 180, 3):
        rad = np.deg2rad(angle)
        score = 0.0
        for r in range(2, max_r, 2):
            y = int(round(cy + r * np.sin(rad)))
            x = int(round(cx + r * np.cos(rad)))
            if 0 <= y < h and 0 <= x < w:
                score += ps[y, x]
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def rl_deconv(img: np.ndarray, psf: np.ndarray, iterations: int = 12) -> np.ndarray:
    out = np.zeros_like(img, dtype=np.float64)
    for c in range(img.shape[2]):
        channel = img_as_float(img[:, :, c])
        restored = richardson_lucy(channel, psf, num_iter=iterations, clip=False)
        out[:, :, c] = restored
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _search_kernel(
    img: np.ndarray,
    lengths: tuple[int, ...],
    angles: tuple[float, ...],
    iterations: int,
) -> tuple[np.ndarray, int, float]:
    best_score = -1.0
    best_kernel = motion_psf(15, 0.0)
    best_len, best_angle = 15, 0.0

    for length in lengths:
        for angle in angles:
            psf = motion_psf(length, angle)
            restored = rl_deconv(img, psf, iterations=iterations)
            score = sharpness_score(cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY))
            if score > best_score:
                best_score = score
                best_kernel = psf
                best_len, best_angle = length, angle

    return best_kernel, best_len, best_angle


def estimate_motion_kernel(img: np.ndarray, search_iter: int = 6) -> tuple[np.ndarray, int, float]:
    coarse_angle = estimate_angle_from_spectrum(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    angles = tuple({coarse_angle, (coarse_angle + 15) % 180, (coarse_angle - 15) % 180})
    return _search_kernel(img, lengths=(11, 15, 19), angles=angles, iterations=search_iter)


def auto_blend_ratio(img: np.ndarray, restored: np.ndarray) -> float:
    """无参考：在若干融合比例中选锐度最高且不过度偏离原图的一项。"""
    input_sharp = sharpness_score(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    best_alpha, best_score = 0.0, input_sharp
    for alpha in (0.0, 0.08, 0.12, 0.16, 0.20):
        merged = cv2.addWeighted(img, 1.0 - alpha, restored, alpha, 0)
        score = sharpness_score(cv2.cvtColor(merged, cv2.COLOR_RGB2GRAY))
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha


def spatial_refine(img: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(bgr, 5, 40, 40)
    return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)


def unsharp_enhance(img: np.ndarray, amount: float = 0.3, sigma: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    enhanced = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def detail_super_resolve(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if scale > 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    return unsharp_enhance(img, amount=0.35, sigma=1.0)


def mild_enhance_fallback(img: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    enhanced = cv2.detailEnhance(bgr, sigma_s=8, sigma_r=0.12)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return unsharp_enhance(rgb, amount=0.15, sigma=0.8)


def temporal_fuse(frames: list[np.ndarray]) -> np.ndarray:
    """视频多帧时域融合，降低单帧去模糊噪声。"""
    if len(frames) == 1:
        return frames[0]
    stack = np.stack(frames, axis=0).astype(np.float64)
    return np.clip(np.median(stack, axis=0), 0, 255).astype(np.uint8)


def deblur_image(
    img: np.ndarray,
    *,
    kernel: np.ndarray | None = None,
    kernel_length: int | None = None,
    kernel_angle: float | None = None,
    estimate_kernel: bool = True,
    rl_iterations: int = 12,
    blend: float | None = None,
    bilateral: bool = True,
    enhance: bool = True,
    super_resolve: bool = False,
    scale: float = 1.0,
    fallback: bool = True,
) -> DeblurResult:
    input_sharp = sharpness_score(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    if kernel is None:
        if estimate_kernel:
            kernel, est_len, est_angle = estimate_motion_kernel(img, search_iter=6)
            kernel_length = est_len
            kernel_angle = est_angle
        else:
            length = kernel_length or 15
            angle = kernel_angle if kernel_angle is not None else 0.0
            kernel = motion_psf(length, angle)
            kernel_length = length
            kernel_angle = angle
    else:
        kernel_length = kernel_length or kernel.shape[0]
        kernel_angle = kernel_angle if kernel_angle is not None else 0.0

    restored = rl_deconv(img, kernel, iterations=rl_iterations)
    alpha = auto_blend_ratio(img, restored) if blend is None else float(blend)
    merged = cv2.addWeighted(img, 1.0 - alpha, restored, alpha, 0)

    if bilateral:
        merged = spatial_refine(merged)
    if enhance:
        merged = unsharp_enhance(merged)
    if super_resolve:
        merged = detail_super_resolve(merged, scale=scale)

    if fallback:
        out_sharp = sharpness_score(cv2.cvtColor(merged, cv2.COLOR_RGB2GRAY))
        if out_sharp < input_sharp * 0.995:
            merged = mild_enhance_fallback(img)
            alpha = 0.0

    return DeblurResult(
        image=merged,
        kernel_length=int(kernel_length),
        kernel_angle=float(kernel_angle),
        kernel=kernel,
        blend=alpha,
    )
