"""
train_unet.py

Trains a lightweight U-Net model (UNetSmall) for binary kelp stipe segmentation
from underwater RGB images using corresponding binary masks.

Expected directory structure (relative to this script / project root):
- images/ : RGB input images (.jpg/.jpeg/.png)
- masks/  : binary segmentation masks (.png), where foreground > 0

Outputs saved to:
- runs/best_unet.pt     : best model checkpoint (by validation Dice)
- runs/loss_curve.png   : training and validation loss per epoch
- runs/val_metrics.png  : validation Dice and IoU per epoch
- runs/preds/           : predicted masks and qualitative panels for the test set
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from tqdm import tqdm
import matplotlib.pyplot as plt


# ----------------------------
# Configuration
# ----------------------------
ROOT = Path(".")
IMAGES_DIR = ROOT / "images"
MASKS_DIR = ROOT / "masks"
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True, parents=True)

SEED = 42
IMG_SIZE = 384          # square resize (CPU-friendly)
BATCH_SIZE = 2          # CPU-friendly
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-5

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# File utilities
# ----------------------------
def list_image_files() -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts]
    files.sort()
    return files


def find_mask_for_image(img_path: Path) -> Path:
    """
    Find the mask file corresponding to an image.

    Supports:
      image: kelp97.jpg -> mask: kelp97.png  OR  kelp97_mask.png
    """
    stem = img_path.stem

    cand1 = MASKS_DIR / f"{stem}.png"
    if cand1.exists():
        return cand1

    cand2 = MASKS_DIR / f"{stem}_mask.png"
    if cand2.exists():
        return cand2

    if stem.endswith("_mask"):
        cand3 = MASKS_DIR / f"{stem.replace('_mask', '')}.png"
        if cand3.exists():
            return cand3

    raise FileNotFoundError(f"No mask found for image: {img_path.name}")


def load_mask_as_binary(mask_path: Path) -> Image.Image:
    """
    Load a mask and convert it to a binary (0/255) single-channel image.

    This handles CVAT-exported masks that may be:
      - black/white masks
      - indexed/palette masks
    """
    m = Image.open(mask_path).convert("L")
    arr = np.array(m)
    arr = (arr > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


# ----------------------------
# Dataset
# ----------------------------
class KelpSegDataset(Dataset):
    def __init__(self, image_paths: list[Path], augment: bool = False):
        self.image_paths = image_paths
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = find_mask_for_image(img_path)

        img = Image.open(img_path).convert("RGB")
        mask = load_mask_as_binary(mask_path)

        img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            if random.random() < 0.3:
                b = 0.8 + 0.4 * random.random()
                img = TF.adjust_brightness(img, b)

        img_t = TF.to_tensor(img)  # (3,H,W), float in [0,1]
        mask_t = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)  # (1,H,W)

        return img_t, mask_t, img_path.name


# ----------------------------
# Model: U-Net (small, CPU-friendly)
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)  # logits


# ----------------------------
# Metrics and outputs
# ----------------------------
def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds_f = preds.view(preds.size(0), -1)
    t_f = targets.view(targets.size(0), -1)

    inter = (preds_f * t_f).sum(dim=1)
    union = preds_f.sum(dim=1) + t_f.sum(dim=1)

    dice = (2 * inter + eps) / (union + eps)
    iou = (inter + eps) / (preds_f.sum(dim=1) + t_f.sum(dim=1) - inter + eps)

    return dice.mean().item(), iou.mean().item()


def save_overlay(img_t: torch.Tensor, mask_t: torch.Tensor, pred_t: torch.Tensor, out_path: Path) -> None:
    """
    Save a qualitative comparison panel: image, ground-truth mask, predicted mask.
    """
    img = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt = (mask_t.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    pr = (pred_t.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].imshow(img); ax[0].set_title("Image"); ax[0].axis("off")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title("Ground truth"); ax[1].axis("off")
    ax[2].imshow(pr, cmap="gray"); ax[2].set_title("Prediction"); ax[2].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Training loop
# ----------------------------
def train() -> None:
    set_seed(SEED)

    files = list_image_files()
    if len(files) == 0:
        raise RuntimeError(f"No images found in {IMAGES_DIR.resolve()}")

    random.shuffle(files)
    n = len(files)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    print(f"Using device: {DEVICE}")
    print(f"Total images: {n}")
    print(f"Train/Val/Test: {len(train_files)}/{len(val_files)}/{len(test_files)}")

    train_ds = KelpSegDataset(train_files, augment=True)
    val_ds = KelpSegDataset(val_files, augment=False)
    test_ds = KelpSegDataset(test_files, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = UNetSmall().to(DEVICE)

    bce = nn.BCEWithLogitsLoss()

    def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_f = probs.view(probs.size(0), -1)
        t_f = targets.view(targets.size(0), -1)
        inter = (probs_f * t_f).sum(dim=1)
        union = probs_f.sum(dim=1) + t_f.sum(dim=1)
        dice = (2 * inter + eps) / (union + eps)
        return 1 - dice.mean()

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_dice = -1.0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad()
            logits = model(x)
            loss = 0.5 * bce(logits, y) + 0.5 * dice_loss(logits, y)
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        model.eval()
        val_losses, dices, ious = [], [], []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = model(x)
                loss = 0.5 * bce(logits, y) + 0.5 * dice_loss(logits, y)
                val_losses.append(loss.item())

                d, i = dice_iou_from_logits(logits, y)
                dices.append(d)
                ious.append(i)

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_dice = float(np.mean(dices)) if dices else 0.0
        val_iou = float(np.mean(ious)) if ious else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f} "
            f"val_iou={val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), RUNS_DIR / "best_unet.pt")

    # Save curves
    fig = plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    fig.savefig(RUNS_DIR / "loss_curve.png", dpi=200)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(history["val_dice"], label="val_dice")
    plt.plot(history["val_iou"], label="val_iou")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.tight_layout()
    fig.savefig(RUNS_DIR / "val_metrics.png", dpi=200)
    plt.close(fig)

    # Test set evaluation + qualitative examples
    model.load_state_dict(torch.load(RUNS_DIR / "best_unet.pt", map_location=DEVICE))
    model.eval()

    test_dices, test_ious = [], []

    preds_dir = RUNS_DIR / "preds"
    preds_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for x, y, name in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            d, i = dice_iou_from_logits(logits, y)
            test_dices.append(d)
            test_ious.append(i)

            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8) * 255

            pred_path = preds_dir / f"{Path(name[0]).stem}_pred.png"
            Image.fromarray(pred, mode="L").save(pred_path)

            overlay_path = preds_dir / f"{Path(name[0]).stem}_qual.png"
            pred_t = torch.from_numpy((pred > 0).astype(np.float32)).unsqueeze(0)
            save_overlay(x[0].cpu(), y[0].cpu(), pred_t, overlay_path)

    print(f"Test Dice: {np.mean(test_dices):.4f} ± {np.std(test_dices):.4f}")
    print(f"Test IoU : {np.mean(test_ious):.4f} ± {np.std(test_ious):.4f}")
    print(f"Outputs saved to: {RUNS_DIR.resolve()}")


if __name__ == "__main__":
    train()
