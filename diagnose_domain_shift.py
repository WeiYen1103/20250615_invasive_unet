"""
Domain shift diagnostic for the ROI-aware U-Net baseline.

目的:
用「分布級」證據確認 train 醫院 (CGMH/Mackey) 與外部醫院 (TCVGH) 之間
是否存在 domain shift,而不是靠單張 patch 的個案。

輸出兩組證據:
1. 輸出端 (confidence):模型在 in-domain (val) 與 out-domain (TCVGH test) 上
   預測的 softmax max-probability 分布直方圖。
   - domain shift 的典型簽名:out-domain 上信心沒有明顯下降,卻全錯
     (= 模型「自信地錯」)。
2. 輸入端 (stain):train 與 TCVGH 的 H&E 染色分布對照
   - 在 RGB 與 HED (Haematoxylin-Eosin-DAB) 兩個色彩空間各畫一組,
     HED 對 H&E 染色差異更敏感,是 domain shift 在輸入端最直接的證據。

用法:
    python diagnose_domain_shift.py

需要修改下方 CONFIG 區的三個路徑:
    INDOMAIN_CSV  : 含 CGMH/Mackey 的 patch_index (train + val)
    OUTDOMAIN_CSV : 含 TCVGH 的 patch_index (你 infer 時用的 test csv)
    CKPT_PATH     : 訓練好的 checkpoint

說明:
- confidence 只在 ROI 內 (target != 255) 的 pixel 上統計,
  ROI 外的 ignore 區不納入,才不會被大片背景稀釋。
- 染色統計只取 ROI 內的組織 pixel,排除白色背景,
  否則兩邊都被一大片白色拉到接近,看不出差異。
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import BreastTumorPatchDataset, IGNORE_INDEX


# =========================
# CONFIG  —— 改這裡
# =========================

INDOMAIN_CSV = "patch_index.csv"        # CGMH/Mackey (train + val)
OUTDOMAIN_CSV = "patch_index_test.csv"  # TCVGH (你 infer 用的那個)

CKPT_PATH = "outputs/unet_resnet34_roi/best_dice.pth"

OUT_DIR = Path("outputs/domain_shift_diag")

NUM_CLASSES = 4
PATCH_SIZE = 512

# 每個 domain 抽幾個 patch 做統計。
# confidence 跟 stain 都用這個數量;太多會慢,500~1000 已足夠看出分布。
N_PATCHES_PER_DOMAIN = 600

# in-domain 用哪個 split 當代表 (val 最能對應「同分布但模型沒看過的 slide」)
INDOMAIN_SPLIT = "val"
OUTDOMAIN_SPLIT = "test"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# RGB -> HED 色彩轉換
# =========================
# 標準 H&E stain separation 矩陣 (Ruifrok & Johnston, 2001)。
# 不依賴 skimage,直接用矩陣做,避免額外安裝。

_HED_FROM_RGB = np.array([
    [1.87798274, -1.00767869, -0.55611582],
    [-0.06590806, 1.13473037, -0.1357552],
    [-0.60190736, -0.48041419, 1.57358807],
], dtype=np.float64)


def rgb_to_hed(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    rgb_uint8: [..., 3], 0-255
    回傳 HED: [..., 3] (Haematoxylin, Eosin, DAB)
    """
    rgb = rgb_uint8.astype(np.float64) / 255.0
    np.maximum(rgb, 1e-6, out=rgb)  # 避免 log(0)
    od = -np.log(rgb)               # optical density
    hed = od @ _HED_FROM_RGB.T
    return hed


# =========================
# 載入 dataset (不 normalize,才能拿回真實顏色做染色統計)
# =========================

def load_dataset(csv, split, normalize):
    return BreastTumorPatchDataset(
        patch_index_csv=csv,
        split=split,
        patch_size=PATCH_SIZE,
        augment=False,
        normalize=normalize,
    )


def pick_indices(ds, n):
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    return idx[:min(n, len(idx))]


def denormalize(image_tensor):
    """把 ImageNet-normalized tensor 還原成 uint8 RGB [H,W,3]。"""
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = np.clip(img * std + mean, 0, 1)
    return (img * 255).astype(np.uint8)


# =========================
# 1. Confidence 分布
# =========================

@torch.no_grad()
def collect_confidence(model, ds, indices):
    """
    回傳一個 1D array:所有 ROI 內 pixel 的 softmax max-probability。
    """
    model.eval()
    confs = []
    for i in tqdm(indices, desc="  confidence"):
        image_tensor, target_tensor = ds[i]
        target = target_tensor.numpy()
        valid = target != IGNORE_INDEX
        if valid.sum() == 0:
            continue

        x = image_tensor.unsqueeze(0).to(DEVICE)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()  # [C,H,W]
        maxprob = prob.max(axis=0)                            # [H,W]

        confs.append(maxprob[valid])
    return np.concatenate(confs) if confs else np.array([])


# =========================
# 2. 染色分布
# =========================

def collect_stain(ds, indices):
    """
    回傳 dict,含 ROI 內、非白背景的 pixel 在 RGB 與 HED 的取樣。
    為了省記憶體,每個 patch 只隨機抽一部分 pixel。
    """
    rgb_samples = []
    hed_samples = []
    per_patch_cap = 2000  # 每張 patch 最多取這麼多 pixel

    for i in tqdm(indices, desc="  stain"):
        image_tensor, target_tensor = ds[i]   # 注意:這個 ds 要 normalize=False
        rgb = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        rgb = (rgb * 255).astype(np.uint8)    # normalize=False 時 dataset 已是 /255 的 float
        target = target_tensor.numpy()

        # 只取 ROI 內
        valid = target != IGNORE_INDEX
        # 排除白色背景 (三通道都很亮的 pixel)
        not_white = ~np.all(rgb > 220, axis=-1)
        mask = valid & not_white
        if mask.sum() == 0:
            continue

        px = rgb[mask]  # [N,3]
        if len(px) > per_patch_cap:
            sel = np.random.RandomState(SEED + i).choice(len(px), per_patch_cap, replace=False)
            px = px[sel]

        rgb_samples.append(px)
        hed_samples.append(rgb_to_hed(px))

    rgb_all = np.concatenate(rgb_samples) if rgb_samples else np.zeros((0, 3))
    hed_all = np.concatenate(hed_samples) if hed_samples else np.zeros((0, 3))
    return {"rgb": rgb_all, "hed": hed_all}


# =========================
# 繪圖
# =========================

def plot_confidence(conf_in, conf_out, out_path):
    plt.figure(figsize=(7, 4.5))
    bins = np.linspace(0.25, 1.0, 60)
    plt.hist(conf_in, bins=bins, density=True, alpha=0.55,
             label=f"in-domain (CGMH/Mackey {INDOMAIN_SPLIT})  mean={conf_in.mean():.3f}")
    plt.hist(conf_out, bins=bins, density=True, alpha=0.55,
             label=f"out-domain (TCVGH {OUTDOMAIN_SPLIT})  mean={conf_out.mean():.3f}")
    plt.xlabel("predicted softmax max-probability (per ROI pixel)")
    plt.ylabel("density")
    plt.title("Prediction confidence: in-domain vs out-domain")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_stain(stain_in, stain_out, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    rgb_names = ["R", "G", "B"]
    hed_names = ["Haematoxylin", "Eosin", "DAB"]

    # RGB row
    for c in range(3):
        ax = axes[0, c]
        ax.hist(stain_in["rgb"][:, c], bins=60, range=(0, 255),
                density=True, alpha=0.55, label="train (CGMH/Mackey)")
        ax.hist(stain_out["rgb"][:, c], bins=60, range=(0, 255),
                density=True, alpha=0.55, label="TCVGH")
        ax.set_title(f"RGB · {rgb_names[c]}")
        if c == 0:
            ax.legend(fontsize=8)

    # HED row
    for c in range(3):
        ax = axes[1, c]
        lo = min(stain_in["hed"][:, c].min(), stain_out["hed"][:, c].min())
        hi = max(stain_in["hed"][:, c].max(), stain_out["hed"][:, c].max())
        ax.hist(stain_in["hed"][:, c], bins=60, range=(lo, hi),
                density=True, alpha=0.55, label="train")
        ax.hist(stain_out["hed"][:, c], bins=60, range=(lo, hi),
                density=True, alpha=0.55, label="TCVGH")
        ax.set_title(f"HED · {hed_names[c]}")

    fig.suptitle("Stain distribution: train (CGMH/Mackey) vs TCVGH", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def summarize_stain(stain_in, stain_out):
    """印出每個通道的 mean±std,給一個量化對照表。"""
    rows = []
    for space in ["rgb", "hed"]:
        names = ["R", "G", "B"] if space == "rgb" else ["H", "E", "D"]
        for c in range(3):
            a = stain_in[space][:, c]
            b = stain_out[space][:, c]
            rows.append({
                "space": space,
                "channel": names[c],
                "train_mean": float(a.mean()),
                "train_std": float(a.std()),
                "tcvgh_mean": float(b.mean()),
                "tcvgh_std": float(b.std()),
                "mean_diff": float(b.mean() - a.mean()),
            })
    return pd.DataFrame(rows)


# =========================
# Main
# =========================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("Domain shift diagnostic")
    print("=" * 60)
    print("Device:", DEVICE)
    print("In-domain  CSV:", INDOMAIN_CSV, "split:", INDOMAIN_SPLIT)
    print("Out-domain CSV:", OUTDOMAIN_CSV, "split:", OUTDOMAIN_SPLIT)
    print("Checkpoint:", CKPT_PATH)
    print("Output dir:", OUT_DIR)

    # ---- datasets ----
    # confidence 要 normalize=True (餵模型);stain 要 normalize=False (看真實顏色)
    ds_in_norm = load_dataset(INDOMAIN_CSV, INDOMAIN_SPLIT, normalize=True)
    ds_out_norm = load_dataset(OUTDOMAIN_CSV, OUTDOMAIN_SPLIT, normalize=True)

    # in-domain 的染色用 train split (更能代表模型訓練時看到的染色分布)
    ds_in_raw = load_dataset(INDOMAIN_CSV, "train", normalize=False)
    ds_out_raw = load_dataset(OUTDOMAIN_CSV, OUTDOMAIN_SPLIT, normalize=False)

    idx_in = pick_indices(ds_in_norm, N_PATCHES_PER_DOMAIN)
    idx_out = pick_indices(ds_out_norm, N_PATCHES_PER_DOMAIN)
    idx_in_raw = pick_indices(ds_in_raw, N_PATCHES_PER_DOMAIN)
    idx_out_raw = pick_indices(ds_out_raw, N_PATCHES_PER_DOMAIN)

    # ---- model ----
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES, activation=None).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    # ---- 1. confidence ----
    print("\n[1/2] Collecting prediction confidence ...")
    conf_in = collect_confidence(model, ds_in_norm, idx_in)
    conf_out = collect_confidence(model, ds_out_norm, idx_out)
    plot_confidence(conf_in, conf_out, OUT_DIR / "confidence_hist.png")

    # ---- 2. stain ----
    print("\n[2/2] Collecting stain distribution ...")
    stain_in = collect_stain(ds_in_raw, idx_in_raw)
    stain_out = collect_stain(ds_out_raw, idx_out_raw)
    plot_stain(stain_in, stain_out, OUT_DIR / "stain_hist.png")

    stain_table = summarize_stain(stain_in, stain_out)
    stain_table.to_csv(OUT_DIR / "stain_summary.csv", index=False)

    # ---- 文字總結 ----
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Confidence mean  in-domain : {conf_in.mean():.4f}")
    print(f"Confidence mean  out-domain: {conf_out.mean():.4f}")
    print(f"Confidence drop            : {conf_in.mean() - conf_out.mean():+.4f}")
    print("  解讀:若 drop 很小 (例如 < 0.05) 但前面已知 TCVGH 分數崩")
    print("        => 模型『自信地錯』,是 domain shift 的強訊號。")
    print()
    print("Stain channel summary (train vs TCVGH):")
    print(stain_table.to_string(index=False))
    print()
    print("  解讀:某通道 mean_diff 大 (尤其 HED 的 H / E),")
    print("        代表 TCVGH 染色明顯偏移 => domain shift 在輸入端的直接證據。")
    print()
    print("Saved:")
    print(" ", OUT_DIR / "confidence_hist.png")
    print(" ", OUT_DIR / "stain_hist.png")
    print(" ", OUT_DIR / "stain_summary.csv")


if __name__ == "__main__":
    main()