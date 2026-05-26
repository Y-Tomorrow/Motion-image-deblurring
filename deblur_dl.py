"""
NRC-HybridNet：深度学习 + 无参考置信度融合 + 频域残差注入

创新点：
1. MIMO-UNet 多尺度深度去模糊（GPU 加速，GoPro 预训练）
2. NR 置信度自适应融合：根据锐度/梯度一致性，自动混合 DL 输出与原图
3. 频域残差注入：保留 DL 结果的同时，从原图回补高频纹理
4. 分块重叠推理：大图/视频帧 tile 并行，兼顾速度与边界质量
5. 视频时序 EMA：相邻帧结果指数滑动平均，抑制闪烁
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from models.mimo_unet import MIMOUNet

MODEL_PATH = Path(__file__).resolve().parent / "models" / "MIMO-UNet.pkl"
# MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pth"
_TILE = 512
_OVERLAP = 32


@dataclass
class DLDeblurResult:
    image: np.ndarray
    blend: float
    confidence: float
    used_tiles: bool


_model: MIMOUNet | None = None
_device: torch.device | None = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_model() -> MIMOUNet:
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"未找到权重 {MODEL_PATH}，请先运行: "
            f"D:\\anaconda3\\envs\\yolov8\\python.exe -m gdown 1EQJoQj3YMLFfzrbgzWMD3Xj96RqLdIlx -O {MODEL_PATH}"
        )
    net = MIMOUNet().to(get_device())
    state = torch.load(MODEL_PATH, map_location=get_device(), weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    net.load_state_dict(state)
    net.eval()
    _model = net
    return net


def _rgb_to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    return t.unsqueeze(0).to(get_device())


def _tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    return (arr.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def _infer_tensor(net: MIMOUNet, tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if get_device().type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = net(tensor)[-1]
        else:
            out = net(tensor)[-1]
    return out.clamp(0, 1)


def infer_full(net: MIMOUNet, img: np.ndarray) -> np.ndarray:
    # MIMO-UNet 返回 [1/4, 1/2, 1.0] 三个尺度输出，推理时只取最后一个全分辨率结果
    return _tensor_to_rgb(_infer_tensor(net, _rgb_to_tensor(img)))


def _hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(max(h, 2))[:h]
    wx = np.hanning(max(w, 2))[:w]
    return np.outer(wy, wx).astype(np.float32)


def infer_tiled(net: MIMOUNet, img: np.ndarray, tile: int = _TILE, overlap: int = _OVERLAP) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= tile and w <= tile:
        return infer_full(net, img)

    acc = np.zeros((h, w, 3), dtype=np.float64)
    weight = np.zeros((h, w, 1), dtype=np.float64)
    step = tile - overlap
    win = _hann2d(tile, tile)[:, :, None]

    for y in range(0, h, step):
        for x in range(0, w, step):
            y1, x1 = min(y + tile, h), min(x + tile, w)
            patch = img[y:y1, x:x1]
            ph, pw = patch.shape[:2]
            if ph < tile or pw < tile:
                pad = np.zeros((tile, tile, 3), dtype=np.uint8)
                pad[:ph, :pw] = patch
                patch_out = infer_full(net, pad)[:ph, :pw]
            else:
                patch_out = infer_full(net, patch)
            wpatch = win[:ph, :pw]
            acc[y:y1, x:x1] += patch_out * wpatch
            weight[y:y1, x:x1] += wpatch
    return np.clip(acc / np.maximum(weight, 1e-6), 0, 255).astype(np.uint8)


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var())


def gradient_energy(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def nr_confidence(blur: np.ndarray, restored: np.ndarray) -> float:
    """无参考置信度：锐度提升 + 梯度方向一致性。"""
    g0 = gradient_energy(cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY))
    g1 = gradient_energy(cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY))
    s0, s1 = sharpness(g0), sharpness(g1)
    sharp_gain = max(0.0, (s1 - s0) / (s0 + 1e-6))
    corr = float(np.mean(np.minimum(g1, g0 * 1.5) / (g0 + 1e-3)))
    return float(np.clip(0.55 * sharp_gain + 0.45 * corr, 0.0, 1.0))


def adaptive_blend(blur: np.ndarray, restored: np.ndarray, confidence: float) -> tuple[np.ndarray, float]:
    alpha = float(np.clip(0.55 + 0.35 * confidence, 0.55, 0.95))
    merged = cv2.addWeighted(blur, 1.0 - alpha, restored, alpha, 0)
    return merged, alpha


def freq_residual_inject(blur: np.ndarray, merged: np.ndarray, strength: float = 0.12) -> np.ndarray:
    """从原图提取高频残差，注入到融合结果，减轻过平滑。"""
    blur_f = blur.astype(np.float32)
    merged_f = merged.astype(np.float32)
    low = cv2.GaussianBlur(blur_f, (0, 0), 1.2)
    high = blur_f - low
    out = merged_f + strength * high
    return np.clip(out, 0, 255).astype(np.uint8)


def temporal_ema(prev: np.ndarray | None, curr: np.ndarray, beta: float = 0.75) -> np.ndarray:
    if prev is None:
        return curr
    out = beta * curr.astype(np.float32) + (1.0 - beta) * prev.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def deblur_image_dl(
    img: np.ndarray,
    *,
    use_tiles: bool = True,
    tile: int = _TILE,
    freq_strength: float = 0.12,
    skip_fusion: bool = False,
) -> DLDeblurResult:
    net = load_model()
    h, w = img.shape[:2]
    used_tiles = use_tiles and (h > tile or w > tile)
    restored = infer_tiled(net, img, tile=tile) if used_tiles else infer_full(net, img)

    if skip_fusion:
        return DLDeblurResult(restored, 1.0, 1.0, used_tiles)

    conf = nr_confidence(img, restored)
    merged, alpha = adaptive_blend(img, restored, conf)
    if conf > 0.15:
        merged = freq_residual_inject(img, merged, strength=freq_strength * conf)
    return DLDeblurResult(merged, alpha, conf, used_tiles)
