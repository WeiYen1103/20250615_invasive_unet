from dataset import BreastTumorPatchDataset

def main():
    for split in ["train", "val"]:
        print("=" * 60)
        print(f"Testing split: {split}")
        print("=" * 60)

        ds = BreastTumorPatchDataset(
            patch_index_csv="patch_index.csv",
            split=split,
            patch_size=512,
            augment=False,
            normalize=True,
        )

        print("Dataset length:", len(ds))

        for i in range(min(5, len(ds))):
            image, target = ds[i]

            print(f"\nSample {i}")
            print("Image shape:", image.shape)
            print("Target shape:", target.shape)
            print("Image dtype:", image.dtype)
            print("Target dtype:", target.dtype)
            print("Target unique values:", target.unique())

if __name__ == "__main__":
    main()