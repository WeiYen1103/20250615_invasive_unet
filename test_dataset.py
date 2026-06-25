import argparse
from pathlib import Path
import random

import pandas as pd
import torch

from dataset import BreastTumorPatchDataset


DEFAULT_PATCH_INDEX = "patch_index.csv"
DEFAULT_TEST_PATCH_INDEX = "patch_index_test.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity-check dataset loading and augmentation behavior."
    )
    parser.add_argument(
        "--patch-index-csv",
        default=None,
        help=(
            "Patch index CSV to inspect. If omitted, use patch_index.csv when it "
            "exists, otherwise fall back to patch_index_test.csv."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional splits to test. Default: use all splits found in the CSV.",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=5,
        help="How many samples to inspect per split. Default: 5",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable dataset augmentation when loading samples.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable ImageNet normalization. Default: off",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sample selection. Default: 42",
    )
    return parser.parse_args()


def resolve_patch_index_csv(arg_value: str | None) -> Path:
    if arg_value is not None:
        return Path(arg_value)

    default_path = Path(DEFAULT_PATCH_INDEX)
    if default_path.exists():
        return default_path

    return Path(DEFAULT_TEST_PATCH_INDEX)


def get_available_splits(csv_path: Path):
    df = pd.read_csv(csv_path, usecols=["split"])
    return sorted(df["split"].dropna().unique().tolist())


def test_split(csv_path: Path, split: str, samples_per_split: int, augment: bool, normalize: bool):
    print("=" * 60)
    print(f"Testing split: {split}")
    print(f"Patch index: {csv_path}")
    print(f"Augment: {augment}")
    print(f"Normalize: {normalize}")
    print("=" * 60)

    ds = BreastTumorPatchDataset(
        patch_index_csv=str(csv_path),
        split=split,
        patch_size=512,
        augment=augment,
        normalize=normalize,
    )

    print("Dataset length:", len(ds))

    indices = random.sample(range(len(ds)), min(samples_per_split, len(ds)))

    for sample_num, idx in enumerate(indices):
        image, target = ds[idx]

        print(f"\nSample {sample_num} (idx={idx})")
        print("Image shape:", image.shape)
        print("Target shape:", target.shape)
        print("Image dtype:", image.dtype)
        print("Target dtype:", target.dtype)
        print("Image min/max:", float(image.min()), float(image.max()))
        print("Target unique values:", target.unique())

        if image.shape != torch.Size([3, 512, 512]):
            raise RuntimeError(f"Bad image shape at idx={idx}: {image.shape}")

        if target.shape != torch.Size([512, 512]):
            raise RuntimeError(f"Bad target shape at idx={idx}: {target.shape}")

        valid_values = set(target.unique().tolist())
        invalid_values = valid_values - {0, 1, 2, 3, 255}
        if invalid_values:
            raise RuntimeError(f"Invalid target values at idx={idx}: {invalid_values}")


def main():
    args = parse_args()
    random.seed(args.seed)

    csv_path = resolve_patch_index_csv(args.patch_index_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Patch index CSV not found: {csv_path}")

    available_splits = get_available_splits(csv_path)
    if not available_splits:
        raise RuntimeError(f"No splits found in {csv_path}")

    if args.splits is None:
        splits_to_test = available_splits
    else:
        missing_splits = sorted(set(args.splits) - set(available_splits))
        if missing_splits:
            raise RuntimeError(
                f"Requested splits not found in {csv_path}: {missing_splits}. "
                f"Available splits: {available_splits}"
            )
        splits_to_test = args.splits

    print("Available splits:", available_splits)
    print("Testing splits:", splits_to_test)

    for split in splits_to_test:
        test_split(
            csv_path=csv_path,
            split=split,
            samples_per_split=args.samples_per_split,
            augment=args.augment,
            normalize=args.normalize,
        )


if __name__ == "__main__":
    main()
