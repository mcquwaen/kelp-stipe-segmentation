"""
make_qual_figures.py

Creates qualitative figure panels for segmentation results:
- 4-panel: Image | Ground truth | Prediction | Overlay
- 3-panel: Image | Ground truth | Prediction
- Optional grid summary (multiple examples in one figure)

Expected structure:
- images/ (original RGB images)
- masks/  (ground truth masks)
- runs/preds/ (predicted masks named like <stem>_pred.png)
Outputs:
- runs/figures/
"""

from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(".")
IMAGES_DIR = BASE_DIR / "images"
MASKS_DIR = BASE_DIR / "masks"
PREDS_DIR = BASE_DIR / "runs" / "preds"
OUT_DIR = BASE_DIR / "runs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_EXAMPLES = 6
DPI = 300


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask01(path: Path) -> np.ndarray:
    """Loads a mask and returns a binary 0/1 array."""
    m = Image.open(path).convert("L")
    arr = np.array(m)
    return (arr > 0).astype(np.uint8)


def find_mask_for_image(img_path: Path) -> Path:
    stem = img_path.stem
    cand1 = MASKS_DIR / f"{stem}.png"
    if cand1.exists():
        return cand1
    cand2 = MASKS_DIR / f"{stem}_mask.png"
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"No ground-truth mask found for: {img_path.name}")


def find_pred_for_image(img_path: Path) -> Path:
    stem = img_path.stem
    cand = PREDS_DIR / f"{stem}_pred.png"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"No prediction found for: {img_path.name} (expected {cand.name})")

def resize_to_hw(arr: np.ndarray, h: int, w: int, is_mask: bool = False) -> np.ndarray:
    """Resize an array to (h, w). Use nearest for masks, bilinear for RGB."""
    if arr.ndim == 2:
        im = Image.fromarray((arr * 255).astype(np.uint8)) if arr.max() <= 1 else Image.fromarray(arr.astype(np.uint8))
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        im = im.resize((w, h), resample=resample)
        out = np.array(im)
        return (out > 0).astype(np.uint8) if is_mask else out

    # RGB
    im = Image.fromarray(arr.astype(np.uint8))
    im = im.resize((w, h), resample=Image.BILINEAR)
    return np.array(im)

def make_overlay(rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """White overlay where mask==1 (expects rgb and mask have same HxW)."""
    overlay = rgb.astype(np.float32).copy()
    white = np.ones_like(overlay) * 255.0
    m = mask01.astype(bool)
    overlay[m] = (1 - alpha) * overlay[m] + alpha * white[m]
    return overlay.clip(0, 255).astype(np.uint8)

def save_panel_4(img_path: Path) -> None:
    rgb = load_rgb(img_path)

    gt_path = find_mask_for_image(img_path)
    pr_path = find_pred_for_image(img_path)

    gt = load_mask01(gt_path)
    pr = load_mask01(pr_path)

    # Use prediction size as the display size (matches model output)
    h, w = pr.shape[:2]

    rgb = resize_to_hw(rgb, h, w, is_mask=False)
    gt  = resize_to_hw(gt,  h, w, is_mask=True)
    pr  = resize_to_hw(pr,  h, w, is_mask=True)

    ov = make_overlay(rgb, pr)

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.5))
    ax[0].imshow(rgb); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title("Ground truth"); ax[1].axis("off")
    ax[2].imshow(pr, cmap="gray"); ax[2].set_title("Prediction"); ax[2].axis("off")
    ax[3].imshow(ov); ax[3].set_title("Overlay"); ax[3].axis("off")

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{img_path.stem}_qual_4panel.png", dpi=DPI)
    plt.close(fig)

def save_panel_3(img_path: Path) -> None:
    rgb = load_rgb(img_path)

    gt_path = find_mask_for_image(img_path)
    pr_path = find_pred_for_image(img_path)

    gt = load_mask01(gt_path)
    pr = load_mask01(pr_path)

    # Use prediction size as the display size
    h, w = pr.shape[:2]

    rgb = resize_to_hw(rgb, h, w, is_mask=False)
    gt  = resize_to_hw(gt,  h, w, is_mask=True)
    pr  = resize_to_hw(pr,  h, w, is_mask=True)

    fig, ax = plt.subplots(1, 3, figsize=(9, 3.5))
    ax[0].imshow(rgb); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title("Ground truth"); ax[1].axis("off")
    ax[2].imshow(pr, cmap="gray"); ax[2].set_title("Prediction"); ax[2].axis("off")

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{img_path.stem}_qual_3panel.png", dpi=DPI)
    plt.close(fig)

def save_grid(img_paths: list[Path]) -> None:
    """Creates a grid with 3 panels per row: Original | GT | Pred."""
    rows = len(img_paths)
    fig, ax = plt.subplots(rows, 3, figsize=(10, 3.2 * rows))

    if rows == 1:
        ax = np.expand_dims(ax, axis=0)

    for r, img_path in enumerate(img_paths):
        rgb = load_rgb(img_path)
        gt = load_mask01(find_mask_for_image(img_path))
        pr = load_mask01(find_pred_for_image(img_path))

        # Use prediction size as display size (ensures consistency)
        h, w = pr.shape[:2]
        rgb = resize_to_hw(rgb, h, w, is_mask=False)
        gt  = resize_to_hw(gt,  h, w, is_mask=True)
        pr  = resize_to_hw(pr,  h, w, is_mask=True)

        ax[r, 0].imshow(rgb); ax[r, 0].axis("off"); ax[r, 0].set_title("Original" if r == 0 else "")
        ax[r, 1].imshow(gt, cmap="gray"); ax[r, 1].axis("off"); ax[r, 1].set_title("Ground truth" if r == 0 else "")
        ax[r, 2].imshow(pr, cmap="gray"); ax[r, 2].axis("off"); ax[r, 2].set_title("Prediction" if r == 0 else "")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "qualitative_grid.png", dpi=DPI)
    plt.close(fig)

def main() -> None:
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts])

    if not images:
        raise RuntimeError(f"No images found in: {IMAGES_DIR.resolve()}")

    valid = []
    for p in images:
        try:
            _ = find_mask_for_image(p)
            _ = find_pred_for_image(p)
            valid.append(p)
        except Exception:
            continue

    if not valid:
        raise RuntimeError("No valid examples found with BOTH ground-truth masks and predictions.")

    examples = valid[: min(N_EXAMPLES, len(valid))]

    for p in examples:
        save_panel_3(p)
        save_panel_4(p)

    save_grid(examples)

    print(f"Saved qualitative figures to: {OUT_DIR.resolve()}")
    print(f"Exported {len(examples)} examples + a grid summary.")


if __name__ == "__main__":
    main()
