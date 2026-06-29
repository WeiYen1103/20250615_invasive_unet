"""
Train U-Net baseline for breast H&E multi-class segmentation.

Class mapping:
0 = background
1 = benign
2 = in_situ
3 = invasive
255 = ignore_index, ROI outside area

Input:
H&E RGB patch, shape [B, 3, 512, 512]

Target:
mask patch, shape [B, 512, 512]
values = 0, 1, 2, 3, or 255
"""

from pathlib import Path
import csv
import json
import random
from collections import Counter, defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from tqdm import tqdm

from dataset import BreastTumorPatchDataset, IGNORE_INDEX
from wandb_utils import finish_wandb, log_wandb_metrics, setup_wandb


# =========================
# Basic settings
# =========================

SEED = 42

PATCH_INDEX_CSV = "patch_index.csv"

NUM_CLASSES = 4
PATCH_SIZE = 512

BATCH_SIZE = 16
NUM_WORKERS = 8

EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
# 退回弱正則化
DECODER_DROPOUT = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path("outputs/unet_resnet34_roi_stain_PlusWeakregularization")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Utility functions
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_confusion_matrix(pred, target, num_classes=4, ignore_index=255):
    """
    pred:   [H, W], numpy int
    target: [H, W], numpy int

    Return:
    confusion matrix, shape [num_classes, num_classes]
    rows = ground truth
    cols = prediction
    """
    valid = target != ignore_index

    pred = pred[valid]
    target = target[valid]

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(target.reshape(-1), pred.reshape(-1)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    return cm


def metrics_from_confusion_matrix(cm):
    """
    cm: rows = ground truth, cols = prediction
    """
    eps = 1e-7

    ious = []
    dices = []

    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        iou = tp / (tp + fp + fn + eps)
        dice = (2 * tp) / (2 * tp + fp + fn + eps)

        ious.append(iou)
        dices.append(dice)

    return np.array(ious), np.array(dices)


def format_metrics(values, name):
    class_names = ["background", "benign", "in_situ", "invasive"]

    msg = [f"{name}:"]
    for cls_name, value in zip(class_names, values):
        msg.append(f"  {cls_name:10s}: {value:.4f}")

    msg.append(f"  macro_mean : {np.nanmean(values):.4f}")
    return "\n".join(msg)


def build_split_audit_report(patch_index_csv: str):
    path = Path(patch_index_csv)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Patch index is empty: {patch_index_csv}")

    required_fields = {"split", "he_path", "hospital", "dominant_class", "patch_type"}
    missing_fields = required_fields - set(rows[0].keys())
    if missing_fields:
        raise RuntimeError(
            f"Patch index missing audit columns: {sorted(missing_fields)}"
        )

    split_counts = Counter(row["split"] for row in rows)
    split_to_slides = defaultdict(set)
    split_to_slide_patch_counts = defaultdict(Counter)

    for row in rows:
        split = row["split"]
        he_path = row["he_path"]
        split_to_slides[split].add(he_path)
        split_to_slide_patch_counts[split][he_path] += 1

    overlaps = []
    splits = sorted(split_counts)
    for idx, split_a in enumerate(splits):
        for split_b in splits[idx + 1:]:
            shared_slides = sorted(split_to_slides[split_a] & split_to_slides[split_b])
            if shared_slides:
                overlaps.append(
                    {
                        "split_a": split_a,
                        "split_b": split_b,
                        "count": len(shared_slides),
                        "examples": shared_slides[:3],
                    }
                )

    distributions = {}
    warnings = []
    warning_gap_threshold = 0.07

    for field in ("hospital", "dominant_class", "patch_type"):
        by_split = defaultdict(Counter)
        for row in rows:
            by_split[row["split"]][str(row[field])] += 1

        values = sorted({str(row[field]) for row in rows})
        field_summary = {}

        for split, counter in by_split.items():
            total = sum(counter.values())
            field_summary[split] = {
                value: {
                    "count": counter.get(value, 0),
                    "ratio": counter.get(value, 0) / max(total, 1),
                }
                for value in values
            }

        distributions[field] = field_summary

        if "train" in field_summary and "val" in field_summary:
            for value in values:
                train_ratio = field_summary["train"][value]["ratio"]
                val_ratio = field_summary["val"][value]["ratio"]
                ratio_gap = abs(train_ratio - val_ratio)
                if ratio_gap >= warning_gap_threshold:
                    warnings.append(
                        f"{field}={value} train/val ratio gap={ratio_gap:.3f} "
                        f"(train={train_ratio:.3f}, val={val_ratio:.3f})"
                    )

    per_split_slide_stats = {}
    for split, counter in split_to_slide_patch_counts.items():
        counts = np.array(list(counter.values()), dtype=np.int64)
        per_split_slide_stats[split] = {
            "slides": int(len(counts)),
            "min_patches": int(counts.min()),
            "median_patches": float(np.median(counts)),
            "max_patches": int(counts.max()),
            "mean_patches": float(counts.mean()),
        }

    return {
        "patch_index_csv": str(path),
        "num_rows": len(rows),
        "split_counts": dict(split_counts),
        "overlaps": overlaps,
        "distributions": distributions,
        "per_split_slide_stats": per_split_slide_stats,
        "warnings": warnings,
    }


def print_split_audit_report(report):
    print("\n" + "=" * 60)
    print("Patch Index Split Audit")
    print("=" * 60)
    print("Rows:", report["num_rows"])
    print("Split counts:", report["split_counts"])

    for split, stats in sorted(report["per_split_slide_stats"].items()):
        print(
            f"{split} slide stats: "
            f"slides={stats['slides']}, "
            f"patches_per_slide(min/median/max/mean)="
            f"{stats['min_patches']}/{stats['median_patches']:.1f}/"
            f"{stats['max_patches']}/{stats['mean_patches']:.1f}"
        )

    if report["overlaps"]:
        print("Split overlap detected:")
        for overlap in report["overlaps"]:
            print(
                f"  {overlap['split_a']} vs {overlap['split_b']}: "
                f"{overlap['count']} shared slides"
            )
            print(f"    examples: {overlap['examples']}")
    else:
        print("Split overlap detected: none")

    for field, field_summary in report["distributions"].items():
        print(f"\n{field} distribution:")
        values = sorted(next(iter(field_summary.values())).keys())
        for value in values:
            parts = []
            for split in sorted(field_summary):
                count = field_summary[split][value]["count"]
                ratio = field_summary[split][value]["ratio"]
                parts.append(f"{split}={count} ({ratio:.3f})")
            print(f"  {value}: " + ", ".join(parts))

    if report["warnings"]:
        print("\nAudit warnings:")
        for warning in report["warnings"]:
            print("  -", warning)
    else:
        print("\nAudit warnings: none")


def add_decoder_dropout(model, dropout_p: float):
    """
    Add light Dropout2d after each SMP U-Net decoder block.
    Forward hooks keep this independent of the exact DecoderBlock signature.
    """
    if dropout_p <= 0:
        return

    decoder_blocks = getattr(getattr(model, "decoder", None), "blocks", None)
    if decoder_blocks is None:
        raise RuntimeError(
            "Could not find model.decoder.blocks to attach decoder dropout."
        )

    dropout_layers = nn.ModuleList(
        [nn.Dropout2d(p=dropout_p) for _ in range(len(decoder_blocks))]
    )
    model.decoder.decoder_dropout_layers = dropout_layers
    model.decoder.decoder_dropout_hook_handles = []

    for block, dropout in zip(decoder_blocks, dropout_layers):
        def _dropout_hook(_module, _inputs, output, dropout_layer=dropout):
            return dropout_layer(output)

        handle = block.register_forward_hook(_dropout_hook)
        model.decoder.decoder_dropout_hook_handles.append(handle)

    print(f"Decoder dropout: p={dropout_p:.2f}, blocks={len(decoder_blocks)}")


# =========================
# Training / validation
# =========================

def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()

    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for images, targets in pbar:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    num_batches = 0

    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    pbar = tqdm(loader, desc="Val", leave=False)

    for images, targets in pbar:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item()
        num_batches += 1

        preds = torch.argmax(logits, dim=1)

        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        for pred, target in zip(preds_np, targets_np):
            cm = compute_confusion_matrix(
                pred=pred,
                target=target,
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
            )
            total_cm += cm

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    val_loss = running_loss / max(num_batches, 1)

    ious, dices = metrics_from_confusion_matrix(total_cm)

    return val_loss, ious, dices, total_cm


# =========================
# Main
# =========================

def main():
    set_seed(SEED)

    print("=" * 60)
    print("U-Net baseline training")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Patch index:", PATCH_INDEX_CSV)
    print("Output dir:", OUT_DIR)
    print("Ignore index:", IGNORE_INDEX)
    print("Num classes:", NUM_CLASSES)

    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    split_audit_report = build_split_audit_report(PATCH_INDEX_CSV)
    print_split_audit_report(split_audit_report)

    audit_path = OUT_DIR / "split_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(split_audit_report, f, indent=2)
    print("Saved split audit:", audit_path)

    if split_audit_report["overlaps"]:
        raise RuntimeError(
            "Split leakage detected in patch_index.csv. "
            "Please fix the split before training."
        )

    # -------------------------
    # Dataset
    # -------------------------

    train_ds = BreastTumorPatchDataset(
        patch_index_csv=PATCH_INDEX_CSV,
        split="train",
        patch_size=PATCH_SIZE,
        augment=True,
        normalize=True,
    )

    val_ds = BreastTumorPatchDataset(
        patch_index_csv=PATCH_INDEX_CSV,
        split="val",
        patch_size=PATCH_SIZE,
        augment=False,
        normalize=True,
    )

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    wandb_run = setup_wandb(
        out_dir=OUT_DIR,
        config={
            "seed": SEED,
            "patch_index_csv": PATCH_INDEX_CSV,
            "num_classes": NUM_CLASSES,
            "patch_size": PATCH_SIZE,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "decoder_dropout": DECODER_DROPOUT,
            "device": DEVICE,
            "encoder_name": "resnet34",
            "encoder_weights": "imagenet",
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "output_dir": str(OUT_DIR),
            "split_audit_json": str(audit_path),
        },
        tags=["breast", "unet", "4class", "roi"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
        drop_last=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
        drop_last=False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    # -------------------------
    # Model
    # -------------------------

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    add_decoder_dropout(model, DECODER_DROPOUT)

    model = model.to(DEVICE)

    # -------------------------
    # Loss / optimizer
    # -------------------------

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

    # -------------------------
    # Training loop
    # -------------------------

    best_val_loss = float("inf")
    best_macro_dice = -1.0
    # ---- early stopping 設定 ----
    EARLY_STOP_PATIENCE = 5   # val loss 連續幾個 epoch 沒改善就停
    epochs_no_improve = 0     # 計數器

    log_path = OUT_DIR / "train_log.csv"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,val_loss,"
            "iou_background,iou_benign,iou_in_situ,iou_invasive,mean_iou,"
            "dice_background,dice_benign,dice_in_situ,dice_invasive,mean_dice,lr\n"
        )

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("=" * 60)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
        )

        val_loss, val_ious, val_dices, val_cm = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        mean_iou = float(np.nanmean(val_ious))
        mean_dice = float(np.nanmean(val_dices))

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")
        print(f"LR:         {current_lr:.6e}")

        print(format_metrics(val_ious, "Val IoU"))
        print(format_metrics(val_dices, "Val Dice"))

        print("Confusion matrix rows=GT, cols=Pred:")
        print(val_cm)

        # Save last checkpoint
        last_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ious": val_ious,
            "val_dices": val_dices,
            "val_confusion_matrix": val_cm,
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
        }

        torch.save(last_ckpt, OUT_DIR / "last.pth")

        # Save best by val loss + early stopping 計數
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0          # 有進步,計數器歸零
            torch.save(last_ckpt, OUT_DIR / "best_loss.pth")
            print("Saved best_loss.pth")
        else:
            epochs_no_improve += 1         # 沒進步,計數器 +1
            print(f"No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs")

        # Save best by macro dice
        if mean_dice > best_macro_dice:
            best_macro_dice = mean_dice
            torch.save(last_ckpt, OUT_DIR / "best_dice.pth")
            print("Saved best_dice.pth")

        log_wandb_metrics(
            wandb_run,
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/lr": current_lr,
                "val/loss": val_loss,
                "val/mean_iou": mean_iou,
                "val/mean_dice": mean_dice,
                "val/iou_background": float(val_ious[0]),
                "val/iou_benign": float(val_ious[1]),
                "val/iou_in_situ": float(val_ious[2]),
                "val/iou_invasive": float(val_ious[3]),
                "val/dice_background": float(val_dices[0]),
                "val/dice_benign": float(val_dices[1]),
                "val/dice_in_situ": float(val_dices[2]),
                "val/dice_invasive": float(val_dices[3]),
                "best/val_loss": best_val_loss,
                "best/macro_dice": best_macro_dice,
            },
            step=epoch,
        )

        # Append log
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                f"{val_ious[0]:.6f},{val_ious[1]:.6f},{val_ious[2]:.6f},{val_ious[3]:.6f},{mean_iou:.6f},"
                f"{val_dices[0]:.6f},{val_dices[1]:.6f},{val_dices[2]:.6f},{val_dices[3]:.6f},{mean_dice:.6f},"
                f"{current_lr:.8e}\n"
            )
        # ---- early stopping 檢查 ----
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(val loss 連續 {EARLY_STOP_PATIENCE} epoch 沒改善)")
            break

    print("\nTraining finished.")
    print("Best val loss:", best_val_loss)
    print("Best macro dice:", best_macro_dice)
    print("Checkpoints saved to:", OUT_DIR)
    finish_wandb(
        wandb_run,
        summary={
            "best_val_loss": best_val_loss,
            "best_macro_dice": best_macro_dice,
            "output_dir": str(OUT_DIR),
            "train_log_csv": str(log_path),
        },
    )


if __name__ == "__main__":
    main()
