"""
Inference and visualization for ROI-aware U-Net baseline.

Class mapping:
0 = background
1 = benign
2 = in_situ
3 = invasive
255 = ignore_index, ROI outside area

綠色 = benign
黃色 = in_situ
紅色 = invasive
黑色 = background
灰色 = ignore / ROI 外

This script:
1. Loads trained U-Net checkpoint.
2. Samples patches from patch_index.csv.
3. Predicts segmentation masks.
4. Saves H&E image, GT overlay, prediction overlay, color masks.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm

from dataset import BreastTumorPatchDataset, IGNORE_INDEX


# =========================
# Settings
# =========================

PATCH_INDEX_CSV = "patch_index.csv"

# Choose "val" or "test"
SPLIT = "val"

# Use best_dice.pth or best_loss.pth
CKPT_PATH = "outputs/unet_resnet34_roi/best_dice.pth"

OUT_DIR = Path("outputs/infer_visualize_val")

NUM_CLASSES = 4
PATCH_SIZE = 512
NUM_SAMPLES = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: "background",
    1: "benign",
    2: "in_situ",
    3: "invasive",
    255: "ignore",
}

# RGB colors
CLASS_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),          # background: black
    1: np.array([0, 255, 0], dtype=np.uint8),        # benign: green
    2: np.array([255, 255, 0], dtype=np.uint8),      # in_situ: yellow
    3: np.array([255, 0, 0], dtype=np.uint8),        # invasive: red
    255: np.array([160, 160, 160], dtype=np.uint8),  # ignore: gray
}


# =========================
# Helper functions
# =========================

def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalized image tensor back to uint8 RGB image.

    input:
        image_tensor: [3, H, W], normalized by ImageNet mean/std

    output:
        image: [H, W, 3], uint8
    """
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = image * std + mean
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)

    return image


def colorize_mask(mask: np.ndarray, show_ignore: bool = True) -> np.ndarray:
    """
    Convert class-id mask to RGB color mask.

    mask:
        [H, W], values 0/1/2/3/255

    output:
        [H, W, 3], uint8
    """
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, cls_color in CLASS_COLORS.items():
        if cls_id == 255 and not show_ignore:
            continue
        color[mask == cls_id] = cls_color

    return color


def make_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    include_background: bool = False,
    include_ignore: bool = False,
) -> np.ndarray:
    """
    Overlay mask on image.

    image:
        [H, W, 3], uint8
    mask:
        [H, W], class id
    """
    overlay = image.copy()
    color_mask = colorize_mask(mask, show_ignore=include_ignore)

    if include_background:
        foreground = mask != IGNORE_INDEX
    else:
        foreground = (mask > 0) & (mask != IGNORE_INDEX)

    if not include_ignore:
        foreground = foreground & (mask != IGNORE_INDEX)

    overlay[foreground] = (
        image[foreground] * (1.0 - alpha) + color_mask[foreground] * alpha
    ).astype(np.uint8)

    return overlay


def save_side_by_side(
    image: np.ndarray,
    gt_overlay: np.ndarray,
    pred_overlay: np.ndarray,
    out_path: Path,
):
    """
    Save image / GT overlay / prediction overlay as one horizontal panel.
    """
    h, w, _ = image.shape

    canvas = np.ones((h, w * 3, 3), dtype=np.uint8) * 255

    canvas[:, 0:w] = image
    canvas[:, w:2 * w] = gt_overlay
    canvas[:, 2 * w:3 * w] = pred_overlay

    Image.fromarray(canvas).save(out_path)


def compute_patch_iou_dice(pred: np.ndarray, target: np.ndarray):
    """
    Compute IoU and Dice for one patch, ignoring target == 255.
    Return dictionaries.
    """
    valid = target != IGNORE_INDEX

    ious = {}
    dices = {}

    for c in range(NUM_CLASSES):
        pred_c = (pred == c) & valid
        target_c = (target == c) & valid

        tp = np.logical_and(pred_c, target_c).sum()
        fp = np.logical_and(pred_c, ~target_c & valid).sum()
        fn = np.logical_and(~pred_c & valid, target_c).sum()

        union = tp + fp + fn
        denom = 2 * tp + fp + fn

        if union == 0:
            iou = np.nan
        else:
            iou = tp / union

        if denom == 0:
            dice = np.nan
        else:
            dice = (2 * tp) / denom

        ious[c] = iou
        dices[c] = dice

    return ious, dices


def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    return model


# =========================
# Main
# =========================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Inference visualization")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Split:", SPLIT)
    print("Checkpoint:", CKPT_PATH)
    print("Output dir:", OUT_DIR)

    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # -------------------------
    # Dataset
    # -------------------------

    dataset = BreastTumorPatchDataset(
        patch_index_csv=PATCH_INDEX_CSV,
        split=SPLIT,
        patch_size=PATCH_SIZE,
        augment=False,
        normalize=True,
    )

    print("Dataset samples:", len(dataset))

    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    indices = indices[:min(NUM_SAMPLES, len(indices))]

    # -------------------------
    # Model
    # -------------------------

    model = build_model().to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("Loaded model state from ckpt['model']")
    else:
        model.load_state_dict(ckpt)
        print("Loaded raw state_dict checkpoint")

    model.eval()

    # -------------------------
    # Visualization loop
    # -------------------------

    summary_rows = []

    for rank, idx in enumerate(tqdm(indices, desc="Infer")):
        image_tensor, target_tensor = dataset[idx]

        image = denormalize_image(image_tensor)
        target = target_tensor.numpy().astype(np.uint8)

        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        # For visualization, do not show prediction outside valid ROI.
        # GT target has 255 outside ROI.
        pred_vis = pred.copy()
        pred_vis[target == IGNORE_INDEX] = IGNORE_INDEX

        gt_overlay = make_overlay(
            image=image,
            mask=target,
            alpha=0.45,
            include_background=False,
            include_ignore=False,
        )

        pred_overlay = make_overlay(
            image=image,
            mask=pred_vis,
            alpha=0.45,
            include_background=False,
            include_ignore=False,
        )

        gt_color = colorize_mask(target, show_ignore=True)
        pred_color = colorize_mask(pred_vis, show_ignore=True)

        ious, dices = compute_patch_iou_dice(pred=pred_vis, target=target)

        row = dataset.df.iloc[idx]

        slide_id = str(row["slide_id"])
        hospital = str(row["hospital"])
        patch_type = str(row.get("patch_type", "unknown"))
        dominant_class = int(row.get("dominant_class", -1))
        x = int(row["x"])
        y = int(row["y"])

        safe_slide_id = (
            slide_id.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(",", "_")
        )

        base = f"{rank:03d}_{SPLIT}_{hospital}_{patch_type}_dom{dominant_class}_{safe_slide_id}_x{x}_y{y}"

        Image.fromarray(image).save(OUT_DIR / f"{base}_image.png")
        Image.fromarray(gt_color).save(OUT_DIR / f"{base}_gt_mask.png")
        Image.fromarray(pred_color).save(OUT_DIR / f"{base}_pred_mask.png")
        Image.fromarray(gt_overlay).save(OUT_DIR / f"{base}_gt_overlay.png")
        Image.fromarray(pred_overlay).save(OUT_DIR / f"{base}_pred_overlay.png")

        save_side_by_side(
            image=image,
            gt_overlay=gt_overlay,
            pred_overlay=pred_overlay,
            out_path=OUT_DIR / f"{base}_panel.png",
        )

        summary_rows.append({
            "rank": rank,
            "dataset_index": idx,
            "split": SPLIT,
            "hospital": hospital,
            "slide_id": slide_id,
            "x": x,
            "y": y,
            "patch_type": patch_type,
            "dominant_class": dominant_class,
            "iou_background": ious[0],
            "iou_benign": ious[1],
            "iou_in_situ": ious[2],
            "iou_invasive": ious[3],
            "dice_background": dices[0],
            "dice_benign": dices[1],
            "dice_in_situ": dices[2],
            "dice_invasive": dices[3],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / f"summary_{SPLIT}.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved visualizations to:", OUT_DIR)
    print("Saved summary:", summary_path)

    print("\nMean patch metrics:")
    metric_cols = [
        "iou_background",
        "iou_benign",
        "iou_in_situ",
        "iou_invasive",
        "dice_background",
        "dice_benign",
        "dice_in_situ",
        "dice_invasive",
    ]

    print(summary_df[metric_cols].mean(numeric_only=True))


if __name__ == "__main__":
    main()