"""
Fast ROI-aware patch index generation.

Purpose:
- Generate patch_index.csv for U-Net training / validation / test.
- Instead of scanning the whole WSI by sliding window,
  this script samples patch centers from ROI-valid class pixels.

Class mapping:
0 = background
1 = benign
2 = in_situ
3 = invasive

ROI mapping:
0 = outside ROI, unlabeled
1 = inside ROI, valid annotation region

Output:
patch_index.csv
"""

import argparse
from collections import defaultdict
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm


# =========================
# Basic settings
# =========================

SLIDES_CSV = "slides.csv"
OUTPUT_CSV = "patch_index.csv"

PATCH_SIZE = 512

# A patch is kept only if at least this proportion is inside ROI.
# 0.25 means at least 25% of the patch pixels must be inside ROI.
ROI_RATIO_THRESHOLD = 0.25

# If a foreground class occupies at least this proportion inside ROI,
# the patch is considered to contain that class.
FOREGROUND_RATIO_THRESHOLD = 0.01

# Maximum number of patches per slide and per patch type.
# These numbers control sampling balance.
MAX_PATCHES_PER_TYPE_PER_SLIDE = {
    "background": 200,
    "benign_only": 500,
    "in_situ_only": 500,
    "invasive_only": 500,
    "mixed": 800,
}

# Maximum number of attempts per slide.
# If a slide has rare classes or no mixed patches, this prevents infinite loops.
MAX_ATTEMPTS_PER_SLIDE = 8000

# Class-aware sampling probability.
# Foreground classes are sampled more often than background.
SAMPLE_CLASS_WEIGHTS = {
    0: 1.0,
    1: 2.5,
    2: 2.5,
    3: 2.5,
}

VALID_CLASSES = [0, 1, 2, 3]

RANDOM_SEED = 42


# =========================
# Helper functions
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def read_level0(path: str) -> np.ndarray:
    """
    Read level-0 image from pyramid TIFF.
    For this script, we only read mask and ROI, not H&E.
    """
    with tifffile.TiffFile(path) as tif:
        arr = tif.pages[0].asarray()
    return arr


def check_mask_and_roi(mask: np.ndarray, roi: np.ndarray, mask_path: str, roi_path: str):
    """
    Validate mask and ROI format.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask should be 2D, got {mask.shape}, path={mask_path}")

    if roi.ndim != 2:
        raise ValueError(f"ROI should be 2D, got {roi.shape}, path={roi_path}")

    if mask.shape != roi.shape:
        raise ValueError(
            f"Mask and ROI shape mismatch:\n"
            f"mask: {mask.shape}, path={mask_path}\n"
            f"roi:  {roi.shape}, path={roi_path}"
        )

    mask_values = set(np.unique(mask).tolist())
    invalid_mask_values = mask_values - set(VALID_CLASSES)
    if invalid_mask_values:
        raise ValueError(
            f"Invalid mask values found: {invalid_mask_values}, path={mask_path}"
        )

    roi_values = set(np.unique(roi).tolist())
    invalid_roi_values = roi_values - {0, 1}
    if invalid_roi_values:
        raise ValueError(
            f"Invalid ROI values found: {invalid_roi_values}, path={roi_path}"
        )


def get_patch_info(mask_patch: np.ndarray, roi_patch: np.ndarray):
    """
    Compute patch type and class ratios only inside ROI.

    Input:
        mask_patch: H x W, values 0/1/2/3
        roi_patch:  H x W, values 0/1

    Return:
        keep_patch
        dominant_class
        patch_type
        ratios
        has_class
        roi_ratio
    """
    roi_valid = roi_patch > 0
    roi_ratio = roi_valid.mean()

    if roi_ratio < ROI_RATIO_THRESHOLD:
        return False, 0, "low_roi", None, None, roi_ratio

    valid_values = mask_patch[roi_valid]

    if valid_values.size == 0:
        return False, 0, "empty_roi", None, None, roi_ratio

    values, counts = np.unique(valid_values, return_counts=True)
    count_dict = dict(zip(values.tolist(), counts.tolist()))

    total_valid = valid_values.size

    ratios = {
        c: count_dict.get(c, 0) / total_valid
        for c in VALID_CLASSES
    }

    has_class = {
        1: ratios[1] >= FOREGROUND_RATIO_THRESHOLD,
        2: ratios[2] >= FOREGROUND_RATIO_THRESHOLD,
        3: ratios[3] >= FOREGROUND_RATIO_THRESHOLD,
    }

    present_classes = [c for c, has in has_class.items() if has]

    if len(present_classes) == 0:
        dominant_class = 0
        patch_type = "background"
    else:
        dominant_class = max(present_classes, key=lambda c: ratios[c])

        if len(present_classes) >= 2:
            patch_type = "mixed"
        elif dominant_class == 1:
            patch_type = "benign_only"
        elif dominant_class == 2:
            patch_type = "in_situ_only"
        elif dominant_class == 3:
            patch_type = "invasive_only"
        else:
            patch_type = "unknown"

    return True, dominant_class, patch_type, ratios, has_class, roi_ratio


def choose_random_pixel(coords):
    """
    coords is output from np.where, i.e. (ys, xs).
    Return:
        (x, y) or None
    """
    ys, xs = coords
    n = len(xs)

    if n == 0:
        return None

    idx = random.randint(0, n - 1)
    return int(xs[idx]), int(ys[idx])


def pixel_to_patch_xy(cx: int, cy: int, W: int, H: int, patch_size: int):
    """
    Convert sampled pixel coordinate to patch top-left coordinate.
    Add random jitter so patches are not always centered exactly at sampled pixel.
    """
    jitter_x = random.randint(-patch_size // 4, patch_size // 4)
    jitter_y = random.randint(-patch_size // 4, patch_size // 4)

    x = cx - patch_size // 2 + jitter_x
    y = cy - patch_size // 2 + jitter_y

    x = max(0, min(x, W - patch_size))
    y = max(0, min(y, H - patch_size))

    return int(x), int(y)


def is_patch_type_full(type_to_patches, patch_type: str) -> bool:
    return len(type_to_patches[patch_type]) >= MAX_PATCHES_PER_TYPE_PER_SLIDE[patch_type]


def all_patch_types_full(type_to_patches) -> bool:
    for patch_type, max_count in MAX_PATCHES_PER_TYPE_PER_SLIDE.items():
        if len(type_to_patches[patch_type]) < max_count:
            return False
    return True


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ROI-aware patch index for selected splits."
    )
    parser.add_argument(
        "--slides-csv",
        default=SLIDES_CSV,
        help=f"Input slide CSV. Default: {SLIDES_CSV}",
    )
    parser.add_argument(
        "--output-csv",
        default=OUTPUT_CSV,
        help=f"Output patch index CSV. Default: {OUTPUT_CSV}",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process. Example: --splits train val test",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append completed slides to output CSV immediately and resume from "
            "existing output CSV if present."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)
    output_path = Path(args.output_csv)

    slides = pd.read_csv(args.slides_csv)
    process_splits = args.splits

    slides = slides[slides["split"].isin(process_splits)].reset_index(drop=True)

    print("Slides CSV:", args.slides_csv)
    print("Processing splits:", process_splits)
    print("Slides after split filtering:", len(slides))

    required_columns = {
        "slide_id",
        "hospital",
        "he_path",
        "mask_path",
        "roi_path",
        "split",
    }

    missing_columns = required_columns - set(slides.columns)
    if missing_columns:
        raise RuntimeError(
            f"slides.csv missing columns: {missing_columns}\n"
            f"Please rerun updated create_slide_csv.py first."
        )

    all_rows = []
    existing_df = None
    processed_slide_ids = set()

    if args.resume and output_path.exists():
        existing_df = pd.read_csv(output_path)
        if "slide_id" not in existing_df.columns:
            raise RuntimeError(
                f"Existing resume file missing slide_id column: {output_path}"
            )
        processed_slide_ids = set(existing_df["slide_id"].astype(str).tolist())
        print(f"Resume mode: found existing output {output_path}")
        print(f"Already completed slides: {len(processed_slide_ids)}")

    print("Loaded slides:", len(slides))
    print(slides.groupby(["hospital", "split"]).size())

    for _, slide_row in tqdm(slides.iterrows(), total=len(slides), desc="Slides"):
        slide_id = slide_row["slide_id"]
        hospital = slide_row["hospital"]
        he_path = slide_row["he_path"]
        mask_path = slide_row["mask_path"]
        roi_path = slide_row["roi_path"]
        split = slide_row["split"]

        if args.resume and str(slide_id) in processed_slide_ids:
            print(f"\nSkipping completed slide: {hospital} | {split} | {slide_id}")
            continue

        print(f"\nProcessing: {hospital} | {split} | {slide_id}")

        mask = read_level0(mask_path)
        roi = read_level0(roi_path)

        check_mask_and_roi(mask, roi, mask_path, roi_path)

        H, W = mask.shape

        print(f"  mask shape: {mask.shape}, unique: {np.unique(mask).tolist()}")
        print(f"  roi shape:  {roi.shape}, unique: {np.unique(roi).tolist()}")
        print(f"  ROI overall ratio: {(roi > 0).mean():.4f}")

        roi_valid = roi > 0

        # Build pixel pools inside ROI.
        # Each pool contains pixel coordinates belonging to one class.
        pixel_pools = {
            0: np.where(roi_valid & (mask == 0)),
            1: np.where(roi_valid & (mask == 1)),
            2: np.where(roi_valid & (mask == 2)),
            3: np.where(roi_valid & (mask == 3)),
        }

        for cls_id in VALID_CLASSES:
            print(f"  class {cls_id} pixels in ROI: {len(pixel_pools[cls_id][0])}")

        # Only sample from classes that actually exist in this slide.
        available_classes = [
            c for c in VALID_CLASSES
            if len(pixel_pools[c][0]) > 0
        ]

        if len(available_classes) == 0:
            print("  No valid ROI pixels found. Skip this slide.")
            continue

        available_weights = [
            SAMPLE_CLASS_WEIGHTS[c]
            for c in available_classes
        ]

        type_to_patches = defaultdict(list)
        seen_xy = set()

        attempts = 0

        while attempts < MAX_ATTEMPTS_PER_SLIDE:
            attempts += 1

            if all_patch_types_full(type_to_patches):
                break

            sample_class = random.choices(
                population=available_classes,
                weights=available_weights,
                k=1,
            )[0]

            sampled_pixel = choose_random_pixel(pixel_pools[sample_class])

            if sampled_pixel is None:
                continue

            cx, cy = sampled_pixel
            x, y = pixel_to_patch_xy(cx, cy, W, H, PATCH_SIZE)

            if (x, y) in seen_xy:
                continue

            seen_xy.add((x, y))

            mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            roi_patch = roi[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            keep, dominant_class, patch_type, ratios, has_class, roi_ratio = get_patch_info(
                mask_patch=mask_patch,
                roi_patch=roi_patch,
            )

            if not keep:
                continue

            if patch_type not in MAX_PATCHES_PER_TYPE_PER_SLIDE:
                continue

            if is_patch_type_full(type_to_patches, patch_type):
                continue

            row = {
                "slide_id": slide_id,
                "hospital": hospital,
                "he_path": he_path,
                "mask_path": mask_path,
                "roi_path": roi_path,
                "x": x,
                "y": y,
                "patch_size": PATCH_SIZE,
                "split": split,

                "dominant_class": dominant_class,
                "patch_type": patch_type,
                "roi_ratio": roi_ratio,

                # Class ratios are computed only inside ROI.
                "ratio_background": ratios[0],
                "ratio_benign": ratios[1],
                "ratio_in_situ": ratios[2],
                "ratio_invasive": ratios[3],

                "has_benign": int(has_class[1]),
                "has_in_situ": int(has_class[2]),
                "has_invasive": int(has_class[3]),
            }

            type_to_patches[patch_type].append(row)

        slide_rows = []

        print(f"  attempts: {attempts}")

        for patch_type in MAX_PATCHES_PER_TYPE_PER_SLIDE:
            patches = type_to_patches[patch_type]
            slide_rows.extend(patches)

            print(
                f"  {patch_type}: kept={len(patches)} / "
                f"max={MAX_PATCHES_PER_TYPE_PER_SLIDE[patch_type]}"
            )

        print(f"  total kept in this slide: {len(slide_rows)}")

        if args.resume:
            slide_df = pd.DataFrame(slide_rows)
            write_header = not output_path.exists()
            slide_df.to_csv(
                output_path,
                mode="a" if output_path.exists() else "w",
                header=write_header,
                index=False,
            )
            processed_slide_ids.add(str(slide_id))
            print(f"  Appended {len(slide_rows)} patches to {output_path}")
        else:
            all_rows.extend(slide_rows)

    if args.resume:
        if not output_path.exists():
            raise RuntimeError(
                "No patches were generated. Try lowering ROI_RATIO_THRESHOLD "
                "or increasing MAX_ATTEMPTS_PER_SLIDE."
            )
        patch_df = pd.read_csv(output_path)
    else:
        patch_df = pd.DataFrame(all_rows)

        if len(patch_df) == 0:
            raise RuntimeError(
                "No patches were generated. Try lowering ROI_RATIO_THRESHOLD "
                "or increasing MAX_ATTEMPTS_PER_SLIDE."
            )

        patch_df = patch_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        patch_df.to_csv(args.output_csv, index=False)

    print("\nSaved:", args.output_csv)
    print("Total patches:", len(patch_df))

    print("\nSplit distribution:")
    print(patch_df["split"].value_counts())

    print("\nPatch type distribution by split:")
    print(patch_df.groupby(["split", "patch_type"]).size())

    print("\nDominant class distribution by split:")
    print(patch_df.groupby(["split", "dominant_class"]).size())

    print("\nHospital distribution by split:")
    print(patch_df.groupby(["hospital", "split"]).size())

    print("\nROI ratio summary:")
    print(patch_df["roi_ratio"].describe())

    print("\nClass ratio summary:")
    print(patch_df[[
        "ratio_background",
        "ratio_benign",
        "ratio_in_situ",
        "ratio_invasive",
    ]].describe())


if __name__ == "__main__":
    main()
