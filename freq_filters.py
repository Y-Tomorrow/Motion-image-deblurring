"""FFT 频域滤波：理想滤波器与巴特沃斯滤波器（低通 / 高通）。"""

from __future__ import annotations

import numpy as np

FilterKind = str  # ideal_lpf | ideal_hpf | butterworth_lpf | butterworth_hpf

FILTER_CHOICES = ("ideal_lpf", "ideal_hpf", "butterworth_lpf", "butterworth_hpf")


def distance_grid(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    cy, cx = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    return np.sqrt((y - cy) ** 2 + (x - cx) ** 2)


def ideal_lowpass(shape: tuple[int, int], cutoff: float) -> np.ndarray:
    d = distance_grid(shape)
    return (d <= cutoff).astype(np.float64)


def ideal_highpass(shape: tuple[int, int], cutoff: float) -> np.ndarray:
    return 1.0 - ideal_lowpass(shape, cutoff)


def butterworth_lowpass(shape: tuple[int, int], cutoff: float, order: int = 2) -> np.ndarray:
    d = distance_grid(shape)
    d = np.maximum(d, 1e-8)
    return 1.0 / (1.0 + (d / cutoff) ** (2 * order))


def butterworth_highpass(shape: tuple[int, int], cutoff: float, order: int = 2) -> np.ndarray:
    d = distance_grid(shape)
    d = np.maximum(d, 1e-8)
    return 1.0 / (1.0 + (cutoff / d) ** (2 * order))


def build_filter(
    kind: FilterKind,
    shape: tuple[int, int],
    cutoff: float,
    order: int = 2,
) -> np.ndarray:
    if kind == "ideal_lpf":
        return ideal_lowpass(shape, cutoff)
    if kind == "ideal_hpf":
        return ideal_highpass(shape, cutoff)
    if kind == "butterworth_lpf":
        return butterworth_lowpass(shape, cutoff, order)
    if kind == "butterworth_hpf":
        return butterworth_highpass(shape, cutoff, order)
    raise ValueError(f"unknown filter kind: {kind}")


def fft_filter_channel(channel: np.ndarray, h: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(channel.astype(np.float64))
    fshift = np.fft.fftshift(f)
    filtered = np.fft.ifft2(np.fft.ifftshift(fshift * h))
    return np.real(filtered)


def apply_freq_filter(img: np.ndarray, h: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img, dtype=np.float64)
    for c in range(img.shape[2]):
        out[:, :, c] = fft_filter_channel(img[:, :, c], h)
    return np.clip(out, 0, 255).astype(np.uint8)


def filter_image(
    img: np.ndarray,
    kind: FilterKind,
    cutoff: float,
    order: int = 2,
    h_cache: dict[tuple, np.ndarray] | None = None,
) -> np.ndarray:
    shape = img.shape[:2]
    key = (kind, shape, cutoff, order)
    if h_cache is not None and key in h_cache:
        h = h_cache[key]
    else:
        h = build_filter(kind, shape, cutoff, order)
        if h_cache is not None:
            h_cache[key] = h
    return apply_freq_filter(img, h)
