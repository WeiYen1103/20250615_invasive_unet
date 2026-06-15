from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path("/work/u5839081/H2G/dataset/Breast_Tumors")

HOSPITALS = ["CGMH", "Mackey", "TCVGH"]

OUTPUT_CSV = "slides.csv"


def main():
    rows = []

    for hospital in HOSPITALS:
        hospital_dir = ROOT / hospital
        mask_dir = hospital_dir / "masks"
        roi_dir = hospital_dir / "rois"

        if not hospital_dir.exists():
            print(f"Skip missing hospital folder: {hospital_dir}")
            continue

        if not mask_dir.exists():
            print(f"Skip {hospital}: missing masks folder: {mask_dir}")
            continue

        if not roi_dir.exists():
            print(f"Skip {hospital}: missing rois folder: {roi_dir}")
            continue

        # 只掃 hospital_dir 底下的 HE .tif，不掃 masks / rois 裡面的 tif
        he_paths = sorted([
            p for p in hospital_dir.glob("*.tif")
            if p.is_file()
        ])

        for he_path in he_paths:
            mask_path = mask_dir / he_path.name
            roi_path = roi_dir / he_path.name

            if not mask_path.exists():
                print(f"Mask not found for: {hospital} / {he_path.name}")
                continue

            if not roi_path.exists():
                print(f"ROI not found for: {hospital} / {he_path.name}")
                continue

            slide_id = he_path.stem

            # TCVGH 當 external test
            if hospital == "TCVGH":
                split = "test"
            else:
                split = "train"

            rows.append({
                "slide_id": slide_id,
                "hospital": hospital,
                "he_path": str(he_path),
                "mask_path": str(mask_path),
                "roi_path": str(roi_path),
                "split": split,
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No valid HE / mask / ROI pairs found.")

    # CGMH + Mackey 內部分 train / val
    train_df = df[df["split"] == "train"].copy()

    if len(train_df) > 1:
        stratify_col = train_df["hospital"] if train_df["hospital"].nunique() > 1 else None

        train_idx, val_idx = train_test_split(
            train_df.index,
            test_size=0.2,
            random_state=42,
            stratify=stratify_col,
        )

        df.loc[val_idx, "split"] = "val"

    df = df.reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(df)
    print(f"\nSaved: {OUTPUT_CSV}")

    print("\nSplit count:")
    print(df["split"].value_counts())

    print("\nHospital count:")
    print(df.groupby(["hospital", "split"]).size())

    print("\nColumns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()