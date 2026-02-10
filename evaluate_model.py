import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

import cv2
import pandas as pd


# =========================
# Configuration
# =========================
ROOT = Path(".")
IMAGES_DIR = ROOT / "images"
MASKS_DIR = ROOT / "masks"
RUNS_DIR = ROOT / "runs"
PRED_DIR = RUNS_DIR / "preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RUNS_DIR / "best_unet_FINAL.pt"   # locked model checkpoint
SEED = 42
IMG_SIZE = 384                                 # must match training resize
DEVICE = "cpu"
THRESH = 0.5
MIN_AREA = 120                                 # connected component area threshold


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# Data I/O
# =========================
def list_image_files() -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts]
    files.sort()
    return files


def find_mask_for_image(img_path: Path) -> Path:
    stem = img_path.stem

    cand1 = MASKS_DIR / f"{stem}.png"
    if cand1.exists():
        return cand1

    cand2 = MASKS_DIR / f"{stem}_mask.png"
    if cand2.exists():
        return cand2

    raise FileNotFoundError(f"No mask found for image: {img_path.name}")


def load_mask_as_binary(mask_path: Path) -> Image.Image:
    """Load a mask and convert to binary (0/255) image."""
    m = Image.open(mask_path).convert("L")
    arr = np.array(m)
    arr = (arr > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


class KelpSegDataset(Dataset):
    """Segmentation dataset (image, binary mask) resized to a fixed resolution."""

    def __init__(self, image_paths: list[Path]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = find_mask_for_image(img_path)

        img = Image.open(img_path).convert("RGB")
        mask = load_mask_as_binary(mask_path)

        # Resize must match the training preprocessing
        img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)

        img_t = TF.to_tensor(img)  # (3,H,W) in [0,1]
        mask_t = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)  # (1,H,W)

        return img_t, mask_t, img_path.name


# =========================
# Model definition (must match training)
# =========================
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


# =========================
# Metrics
# =========================
def dice_iou(pred01: np.ndarray, gt01: np.ndarray, eps: float = 1e-7) -> tuple[float, float]:
    pred = pred01.astype(np.float32).ravel()
    gt = gt01.astype(np.float32).ravel()

    inter = (pred * gt).sum()
    dice = (2 * inter + eps) / (pred.sum() + gt.sum() + eps)
    iou = (inter + eps) / (pred.sum() + gt.sum() - inter + eps)
    return float(dice), float(iou)


def precision_recall_f1(pred01: np.ndarray, gt01: np.ndarray, eps: float = 1e-7) -> tuple[float, float, float]:
    pred = pred01.astype(np.uint8).ravel()
    gt = gt01.astype(np.uint8).ravel()

    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(precision), float(recall), float(f1)


def count_stipes_from_mask(mask01: np.ndarray) -> int:
    """Connected-component based stipe count from a binary mask."""
    binary = (mask01 > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            count += 1
    return count


def main() -> None:
    set_seed(SEED)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH.resolve()}")

    files = list_image_files()
    if not files:
        raise RuntimeError(f"No images found in: {IMAGES_DIR.resolve()}")

    # Deterministic 70/15/15 split (seeded shuffle)
    random.shuffle(files)
    n = len(files)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    test_files = files[n_train + n_val :]

    print(f"Using model: {MODEL_PATH}")
    print(f"Total images: {n} | Test set: {len(test_files)}")

    test_loader = DataLoader(KelpSegDataset(test_files), batch_size=1, shuffle=False, num_workers=0)

    model = UNetSmall().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    rows = []
    dices, ious, precs, recs, f1s = [], [], [], [], []

    with torch.no_grad():
        for x, y, name in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred01 = (prob > THRESH).astype(np.uint8)

            gt01 = y[0, 0].cpu().numpy().astype(np.uint8)

            d, i = dice_iou(pred01, gt01)
            p, r, f = precision_recall_f1(pred01, gt01)

            dices.append(d)
            ious.append(i)
            precs.append(p)
            recs.append(r)
            f1s.append(f)

            # Save predicted mask for record
            pred_png = (pred01 * 255).astype(np.uint8)
            Image.fromarray(pred_png, mode="L").save(PRED_DIR / f"{Path(name[0]).stem}_pred.png")

            # Stipe counts from connected components
            gt_count = count_stipes_from_mask(gt01)
            pr_count = count_stipes_from_mask(pred01)

            rows.append(
                {
                    "image": name[0],
                    "dice": d,
                    "iou": i,
                    "precision": p,
                    "recall": r,
                    "f1": f,
                    "gt_stipes": gt_count,
                    "pred_stipes": pr_count,
                    "abs_count_error": abs(gt_count - pr_count),
                }
            )

    df = pd.DataFrame(rows)

    # Additional count statistics
    df["signed_error"] = df["pred_stipes"] - df["gt_stipes"]
    mean_gt = df["gt_stipes"].mean()
    mean_pr = df["pred_stipes"].mean()
    mean_signed = df["signed_error"].mean()
    mae = df["abs_count_error"].mean()

    corr = float("nan")
    if len(df) > 1:
        corr = df[["gt_stipes", "pred_stipes"]].corr().iloc[0, 1]

    out_csv = RUNS_DIR / "evaluation_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Pixel-level segmentation metrics (test set) ===")
    print(f"Dice      : {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"IoU       : {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Precision : {np.mean(precs):.4f}")
    print(f"Recall    : {np.mean(recs):.4f}")
    print(f"F1-score  : {np.mean(f1s):.4f}")

    print("\n=== Stipe count summary (test set) ===")
    print(f"Mean GT stipes        : {mean_gt:.2f}")
    print(f"Mean predicted stipes : {mean_pr:.2f}")
    print(f"Mean signed error     : {mean_signed:.2f}  (positive = overcount)")
    print(f"Count correlation (r) : {corr:.3f}")
    print(f"MAE                   : {mae:.3f}")
    print(f"Saved table to        : {out_csv.resolve()}")


if __name__ == "__main__":
    main()
