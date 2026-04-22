import os
import random
import shutil
import scipy.io
import cv2
import numpy as np
from tqdm import tqdm

# --- 路徑設定 ---
base_path = "./PlaTiF Dataset"
output_path = "yolo_dataset"

# label=7 代表 Normal（無骨折），1~6 為 Schatzker 骨折分型
NORMAL_LABEL = 7
YOLO_CLASS_ID = 0  # YOLO 只用一個類別：Fracture

# 切分比例
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
# TEST_RATIO  = 0.1（剩餘）

RANDOM_SEED = 42


def normalize_image(img):
    img_float = img.astype(np.float32)
    img_norm = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX)
    img_8u = img_norm.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_8u)


def mask_to_yolo_bbox(mask, img_h, img_w):
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0:
        return None
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return x_center, y_center, w, h


def collect_all_samples():
    """讀取所有 .mat 檔，回傳 (patient_id, im_key, part_path) 的清單"""
    samples = []
    part_dirs = sorted([
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and "Patient Data" in d
    ])
    print(f"找到 {len(part_dirs)} 個資料夾：{part_dirs}\n")

    for part in part_dirs:
        part_path = os.path.join(base_path, part)
        for f in sorted(os.listdir(part_path)):
            if not f.endswith(".mat"):
                continue
            pid = f.replace(".mat", "")
            try:
                mat_data = scipy.io.loadmat(os.path.join(part_path, f))
                patient_data = mat_data[pid]
                im_fields = [k for k in patient_data.dtype.names if k.startswith("im")]
                for im_key in im_fields:
                    samples.append((pid, im_key, part_path))
            except Exception as e:
                print(f"  Error reading {pid}: {e}")

    return samples


def save_sample(pid, im_key, part_path, split):
    """處理單張影像並存到對應的 split 資料夾"""
    img_dir = os.path.join(output_path, "images", split)
    lbl_dir = os.path.join(output_path, "labels", split)

    mat_data = scipy.io.loadmat(os.path.join(part_path, f"{pid}.mat"))
    patient_data = mat_data[pid]
    im_data = patient_data[0, 0][im_key][0, 0]

    img = im_data["OriginalImage"]
    mask = im_data["BW"]
    label = int(im_data["label"][0, 0])

    img_h, img_w = img.shape
    img_final = normalize_image(img)

    filename = f"{pid}_{im_key}"
    cv2.imwrite(os.path.join(img_dir, f"{filename}.jpg"), img_final)

    label_path = os.path.join(lbl_dir, f"{filename}.txt")
    if label == NORMAL_LABEL:
        open(label_path, "w").close()
    else:
        bbox = mask_to_yolo_bbox(mask, img_h, img_w)
        with open(label_path, "w") as f:
            if bbox:
                x_c, y_c, w, h = bbox
                f.write(f"{YOLO_CLASS_ID} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


# --- 建立輸出資料夾 ---
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_path, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", split), exist_ok=True)

# --- 收集所有樣本並打亂 ---
all_samples = collect_all_samples()
random.seed(RANDOM_SEED)
random.shuffle(all_samples)

total = len(all_samples)
n_train = int(total * TRAIN_RATIO)
n_val   = int(total * VAL_RATIO)

train_samples = all_samples[:n_train]
val_samples   = all_samples[n_train:n_train + n_val]
test_samples  = all_samples[n_train + n_val:]

print(f"總影像數：{total}")
print(f"Train：{len(train_samples)}  Val：{len(val_samples)}  Test：{len(test_samples)}\n")

# --- 處理並存檔 ---
for split, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
    print(f"處理 {split}...")
    for pid, im_key, part_path in tqdm(samples):
        try:
            save_sample(pid, im_key, part_path, split)
        except Exception as e:
            print(f"  Error {pid} {im_key}: {e}")

print("\n完成！資料夾結構：")
print("  yolo_dataset/images/train/")
print("  yolo_dataset/images/val/")
print("  yolo_dataset/images/test/")
