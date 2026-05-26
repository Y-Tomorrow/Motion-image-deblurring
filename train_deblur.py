"""
NRC-HybridNet 训练脚本：在 GOPRO_Large 上 fine-tune MIMO-UNet / MIMO-UNet+。

用法（yolov8 环境）：
  D:\anaconda3\envs\yolov8\python.exe train_deblur.py `
  --data-dir d:\work\sight\GOPRO_Large `
  --model-name MIMO-UNet `
  --epochs 20 `
  --batch-size 4 `
  --lr 1e-4

训练完成后，权重保存在 results/checkpoints/ 下，可直接用于 deblur_dl.py 的推理。
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from models.mimo_unet import MIMOUNet, MIMOUNetPlus

# -------------------------- 数据增强 --------------------------

class PairRandomCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img, label):
        w, h = img.size
        th, tw = self.size, self.size
        if w == tw and h == th:
            return img, label
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return img.crop((j, i, j + tw, i + th)), label.crop((j, i, j + tw, i + th))


class PairRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)
        return img, label


class PairToTensor:
    def __call__(self, img, label):
        return F.to_tensor(img), F.to_tensor(label)


class PairCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


# -------------------------- 数据集 --------------------------

class GOPRODeblurDataset(Dataset):
    """支持 GOPRO_Large 结构：train/<seq>/blur_gamma/*.png + train/<seq>/sharp/*.png"""

    def __init__(self, root: Path, transform=None, is_test: bool = False):
        self.root = Path(root)
        self.transform = transform
        self.is_test = is_test
        self.pairs: list[tuple[Path, Path, str]] = []

        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue
            blur_dir = seq_dir / "blur_gamma"
            sharp_dir = seq_dir / "sharp"
            if not blur_dir.is_dir() or not sharp_dir.is_dir():
                continue
            for blur_path in sorted(blur_dir.glob("*.png")):
                sharp_path = sharp_dir / blur_path.name
                if sharp_path.is_file():
                    self.pairs.append((blur_path, sharp_path, seq_dir.name))

        if not self.pairs:
            raise RuntimeError(f"在 {root} 下未找到 blur_gamma/sharp 配对")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path, _ = self.pairs[idx]
        blur = Image.open(blur_path).convert("RGB")
        sharp = Image.open(sharp_path).convert("RGB")

        if self.transform:
            blur, sharp = self.transform(blur, sharp)
        else:
            blur = F.to_tensor(blur)
            sharp = F.to_tensor(sharp)

        if self.is_test:
            return blur, sharp, self.pairs[idx][0].name
        return blur, sharp


def get_train_transform(crop_size: int = 256):
    return PairCompose([
        PairRandomCrop(crop_size),
        PairRandomHorizontalFlip(),
        PairToTensor(),
    ])


# -------------------------- 损失函数 --------------------------

class CharbonnierLoss(nn.Module):
    """L1 的平滑版本，训练更稳定"""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


# -------------------------- 工具函数 --------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.clamp(pred, 0, 1)
    mse = torch.mean((pred - target) ** 2)
    return float(10 * torch.log10(1.0 / mse)) if mse > 0 else 100.0


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    psnr_sum = 0.0
    for blur, sharp in loader:
        blur, sharp = blur.to(device), sharp.to(device)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred = model(blur)[-1]
        psnr_sum += compute_psnr(pred, sharp)
    return psnr_sum / len(loader)


def save_checkpoint(state: dict, is_best: bool, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_dir / "latest.pth")
    if is_best:
        torch.save(state, save_dir / "best.pth")


# -------------------------- 主训练流程 --------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据
    train_set = GOPRODeblurDataset(
        args.data_dir / "train",
        transform=get_train_transform(args.crop_size),
    )
    val_set = GOPRODeblurDataset(
        args.data_dir / "train",  # 简单起见，用 train 的一部分做 val（实际可拆分 valid）
        transform=PairToTensor(),
    )
    # 为了简单，这里 val 用 train 的前 100 张，实际项目建议拆分
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    # 模型
    if args.model_name == "MIMO-UNetPlus":
        model = MIMOUNetPlus(num_res=args.num_res).to(device)
    else:
        model = MIMOUNet(num_res=args.num_res).to(device)

    # 继续训练
    start_epoch = 0
    best_psnr = 0.0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resume from epoch {start_epoch}, best PSNR {best_psnr:.2f}")

    # 优化器 & 调度
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    criterion = CharbonnierLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Start training {args.model_name} for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for i, (blur, sharp) in enumerate(train_loader):
            blur, sharp = blur.to(device), sharp.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(blur)
                # MIMO-UNet 返回 3 个不同尺度输出 (1/4, 1/2, 1.0)，需要对应缩放 GT
                scales = [0.25, 0.5, 1.0]
                losses = []
                for p, s in zip(preds, scales):
                    if s < 1.0:
                        gt = nnF.interpolate(sharp, scale_factor=s, mode='bilinear', align_corners=False)
                    else:
                        gt = sharp
                    losses.append(criterion(p, gt))
                loss = sum(losses) / len(losses)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Iter [{i+1}/{len(train_loader)}] "
                      f"Loss {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        val_psnr = validate(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss {avg_loss:.4f}  Val PSNR {val_psnr:.2f}  LR {lr:.6f}  "
              f"Time {(time.time()-t0)/60:.1f} min")

        # 保存 checkpoint
        is_best = val_psnr > best_psnr
        best_psnr = max(best_psnr, val_psnr)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "args": args,
        }
        save_checkpoint(state, is_best, save_dir)

        if is_best:
            print(f"  >> New best PSNR: {best_psnr:.2f}  (saved best.pth)")

    print(f"Training finished. Best Val PSNR: {best_psnr:.2f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIMO-UNet on GOPRO")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "GOPRO_Large",
                        help="GOPRO_Large 根目录")
    parser.add_argument("--model-name", choices=["MIMO-UNet", "MIMO-UNetPlus"], default="MIMO-UNet")
    parser.add_argument("--num-res", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=200, help="每多少 epoch 学习率 * 0.5")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-dir", type=Path, default=Path(__file__).parent / "models")
    parser.add_argument("--resume", type=str, default="", help="checkpoint 路径，继续训练")
    args = parser.parse_args()

    train(args)
