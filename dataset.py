"""
1. 從 H&E 讀取 512×512 RGB patch
2. 從 mask 讀取同座標的 512×512 class mask
3. 從 ROI 讀取同座標的 512×512 ROI mask
4. 把 ROI 外的 target 設成 255
5. 後續 CrossEntropyLoss(ignore_index=255) 就不會計算 ROI 外區域
---
當 DataLoader 要第 i 筆資料時，根據 patch_index.csv 的第 i 列，
去 H&E、mask、ROI 中切出對應 patch，轉成 PyTorch tensor 回傳。
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import tifffile
import zarr
from skimage.color import rgb2hed, hed2rgb

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

# 把roi外的pixel target 設成225，讓loss不計算roi外的區域
IGNORE_INDEX = 255

# 標準 H&E stain separation 矩陣 (Ruifrok & Johnston, 2001)。
# 用來在不依賴額外套件的情況下做輕量 stain augmentation。
# _HED_FROM_RGB = np.array([
#     [1.87798274, -1.00767869, -0.55611582],
#     [-0.06590806, 1.13473037, -0.1357552],
#     [-0.60190736, -0.48041419, 1.57358807],
# ], dtype=np.float32)
# _RGB_FROM_HED = np.linalg.inv(_HED_FROM_RGB).astype(np.float32)

STAIN_AUG_PROB = 0.8
STAIN_SCALE_RANGE = (0.85, 1.15)
STAIN_BIAS_RANGE = (-0.05, 0.05)


class BreastTumorPatchDataset(Dataset):
    """
    ROI-aware patch dataset for breast tumor multi-class segmentation.

    Class mapping:
        0 = background
        1 = benign
        2 = in_situ
        3 = invasive
        255 = ignore, ROI 外不計算 loss

    Input:
        H&E patch: [3, H, W], float32, normalized to [0, 1]

    Target:
        mask patch: [H, W], int64
        values in {0,1,2,3,255}
    """

    def __init__(
        self,
        patch_index_csv: str,
        split: str = "train",
        patch_size: int = 512,
        augment: bool = False,
        normalize: bool = True,
    ):
        self.df = pd.read_csv(patch_index_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(f"No samples found for split={split}")

        required_cols = {
            "he_path",
            "mask_path",
            "roi_path",
            "x",
            "y",
            "patch_size",
            "split",
        }

        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise RuntimeError(
                f"patch_index.csv missing columns: {missing_cols}\n"
                f"Please rerun ROI-aware create_patch_index.py."
            )

        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        self.normalize = normalize

        if not HAS_OPENSLIDE:
            raise ImportError(
                "openslide-python is required for patch-level WSI reading.\n"
                "Install it with:\n"
                "  conda install -c conda-forge openslide-python openslide -y\n"
                "or:\n"
                "  pip install openslide-python\n"
            )

        # 每個 worker 會各自有一份 cache。
        # 這裡 cache OpenSlide object，不會把整張 WSI 讀進 RAM。
        self.slide_cache = {}

        print(f"[Dataset] split={split}, samples={len(self.df)}")

    def __len__(self):
        return len(self.df)

    def _get_slide(self, path: str):
        """
        Cache slide reader.

        Priority:
        1. OpenSlide
        2. tifffile + zarr fallback for BigTIFF or unsupported TIFF
        """
        if path in self.slide_cache:
            return self.slide_cache[path]

        try:
            slide = openslide.OpenSlide(path)
            self.slide_cache[path] = {
                "type": "openslide",
                "slide": slide,
            }
            return self.slide_cache[path]

        except Exception as e:
            print("[Dataset] OpenSlide failed, use tifffile+zarr fallback")
            print(f"  path: {path}")
            print(f"  error: {repr(e)}")

            tif = tifffile.TiffFile(path)
            store = tif.series[0].aszarr()
            arr = zarr.open(store, mode="r")

            self.slide_cache[path] = {
                "type": "zarr",
                "tif": tif,      # keep TiffFile object alive
                "arr": arr,
            }

            print(f"  zarr shape: {arr.shape}, dtype: {arr.dtype}")

            return self.slide_cache[path]

    def _read_region_rgb(self, path: str, x: int, y: int, size: int) -> np.ndarray:
        """
        Read H&E RGB patch.

        Return:
            image: H x W x 3, uint8
        """
        reader = self._get_slide(path)

        if reader["type"] == "openslide":
            slide = reader["slide"]

            patch = slide.read_region(
                location=(x, y),
                level=0,
                size=(size, size),
            )

            patch = patch.convert("RGB")
            patch = np.array(patch, dtype=np.uint8)
            return patch

        elif reader["type"] == "zarr":
            arr = reader["arr"]

            patch = arr[y:y + size, x:x + size]
            patch = np.asarray(patch)

            if patch.ndim == 2:
                patch = np.stack([patch, patch, patch], axis=-1)

            elif patch.ndim == 3:
                if patch.shape[2] >= 3:
                    patch = patch[:, :, :3]
                elif patch.shape[2] == 1:
                    patch = np.repeat(patch, 3, axis=2)
                else:
                    raise ValueError(
                        f"Unsupported RGB patch shape: {patch.shape}, path={path}"
                    )
            else:
                raise ValueError(
                    f"Unsupported RGB patch ndim: {patch.ndim}, shape={patch.shape}, path={path}"
                )

            patch = patch.astype(np.uint8)
            return patch

        else:
            raise RuntimeError(f"Unknown reader type: {reader['type']}")

    def _read_region_gray(self, path: str, x: int, y: int, size: int) -> np.ndarray:
        """
        Read mask / ROI patch.

        Return:
            gray: H x W, uint8
        """
        reader = self._get_slide(path)

        if reader["type"] == "openslide":
            slide = reader["slide"]

            patch = slide.read_region(
                location=(x, y),
                level=0,
                size=(size, size),
            )

            patch = np.array(patch, dtype=np.uint8)

            if patch.ndim == 2:
                gray = patch
            else:
                gray = patch[:, :, 0]

            return gray.astype(np.uint8)

        elif reader["type"] == "zarr":
            arr = reader["arr"]

            patch = arr[y:y + size, x:x + size]
            patch = np.asarray(patch)

            if patch.ndim == 2:
                gray = patch
            elif patch.ndim == 3:
                gray = patch[:, :, 0]
            else:
                raise ValueError(
                    f"Unsupported gray patch shape: {patch.shape}, path={path}"
                )

            return gray.astype(np.uint8)

        else:
            raise RuntimeError(f"Unknown reader type: {reader['type']}")

    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray, roi: np.ndarray):
        """
        Apply simple spatial augmentations.
        image: H x W x 3
        mask: H x W
        roi: H x W
        """
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            roi = np.flip(roi, axis=1).copy()

        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
            roi = np.flip(roi, axis=0).copy()


        # 0, 90, 180, 270 degree rotation
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
            roi = np.rot90(roi, k, axes=(0, 1)).copy()

        return image, mask, roi 

    def _apply_stain_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Stain augmentation (Tellez et al. 2019 style) using skimage HED.
        只動 image,mask/roi 不變。
        """
        if random.random() >= STAIN_AUG_PROB:
            return image

        rgb = image.astype(np.float32) / 255.0
        hed = rgb2hed(rgb)  # [H,W,3], channel 0=H, 1=E, 2=DAB

        # 對 H 和 E 兩個 channel 做 alpha*x + beta 擾動
        for stain_idx in (0, 1):
            alpha = random.uniform(*STAIN_SCALE_RANGE)
            beta = random.uniform(*STAIN_BIAS_RANGE)
            hed[..., stain_idx] = hed[..., stain_idx] * alpha + beta

        rgb_aug = hed2rgb(hed)
        rgb_aug = np.clip(rgb_aug, 0.0, 1.0)
        return (rgb_aug * 255.0).astype(np.uint8)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        he_path = row["he_path"]
        mask_path = row["mask_path"]
        roi_path = row["roi_path"]

        x = int(row["x"])
        y = int(row["y"])
        size = int(row["patch_size"])

        # 保險：如果 patch_index 裡的 patch_size 與 dataset 設定不同，以 CSV 為準。
        if size != self.patch_size:
            size = self.patch_size

        image = self._read_region_rgb(he_path, x, y, size)
        mask = self._read_region_gray(mask_path, x, y, size)
        roi = self._read_region_gray(roi_path, x, y, size)

        # 防呆檢查
        if image.shape[:2] != (size, size):
            raise ValueError(f"Bad image shape: {image.shape}, idx={idx}")

        if mask.shape != (size, size):
            raise ValueError(f"Bad mask shape: {mask.shape}, idx={idx}")

        if roi.shape != (size, size):
            raise ValueError(f"Bad roi shape: {roi.shape}, idx={idx}")

        # mask 只允許 0/1/2/3
        # ROI 只允許 0/1，但 OpenSlide 讀出來有時候可能會有 255，
        # 所以這裡用 roi > 0 當有效區域。
        mask = mask.astype(np.uint8)
        roi = roi.astype(np.uint8)

        # ROI 外設成 ignore_index，避免把未標註區域當 background
        target = mask.copy()
        target[roi == 0] = IGNORE_INDEX

        # 若 mask 有非 0/1/2/3 的值，除了 IGNORE_INDEX 外，都視為錯誤
        valid_values = {0, 1, 2, 3, IGNORE_INDEX}
        unique_values = set(np.unique(target).tolist())
        invalid_values = unique_values - valid_values
        if invalid_values:
            raise ValueError(
                f"Invalid target values {invalid_values} at idx={idx}, "
                f"mask_path={mask_path}, roi_path={roi_path}, x={x}, y={y}"
            )

        # 資料增強
        if self.augment:
            image, target, roi = self._apply_augmentation(image, target, roi)
            image = self._apply_stain_augmentation(image)

        # image: H,W,3 -> 3,H,W
        # 影像正規化:轉成 float32 並 normalize 到 [0, 1]
        image = image.astype(np.float32) / 255.0

        if self.normalize:
            # ImageNet normalization, suitable for pretrained ResNet encoder
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))

        image_tensor = torch.tensor(image, dtype=torch.float32)
        target_tensor = torch.tensor(target.astype(np.int64), dtype=torch.long)

        return image_tensor, target_tensor
