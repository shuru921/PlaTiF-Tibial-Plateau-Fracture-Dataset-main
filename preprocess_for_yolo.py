import os
import random
import scipy.io
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# --- 路徑設定 ---
base_path = "./PlaTiF Dataset"
output_path = "yolo_dataset"

# label=7 代表 Normal（無骨折），1~6 為 Schatzker 骨折分型
NORMAL_LABEL = 7
YOLO_CLASS_ID = 0  # YOLO 只用一個類別：Fracture

# 切分比例（以病人為單位，分層切分）
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


def collect_patients_by_label():
    """
    掃描所有 Part，依照 label 分組，回傳：
    { label: [(patient_id, part_path), ...], ... }
    """
    part_dirs = sorted([
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and "Patient Data" in d
    ])
    print(f"找到 {len(part_dirs)} 個資料夾：{part_dirs}\n")

    groups = defaultdict(list)

    for part in part_dirs:
        part_path = os.path.join(base_path, part)
        for f in sorted(os.listdir(part_path)):
            if not f.endswith(".mat"):
                continue
            pid = f.replace(".mat", "")
            try:
                mat_data = scipy.io.loadmat(os.path.join(part_path, f))
                patient_data = mat_data[pid]
                # 取第一張影像的 label 代表該病人的類別
                first_im = [k for k in patient_data.dtype.names if k.startswith("im")][0]
                label = int(patient_data[0, 0][first_im][0, 0]["label"][0, 0])
                groups[label].append((pid, part_path))
            except Exception as e:
                print(f"  Error reading {pid}: {e}")

    return groups


def stratified_split(groups):
    """
    對每個 label 群組各自做 80/10/10 切分，
    確保每個 split 都有每種類別的資料
    """
    train, val, test = [], [], []
    random.seed(RANDOM_SEED)

    label_names = {1: "Type1", 2: "Type2", 3: "Type3",
                   4: "Type4", 5: "Type5", 6: "Type6", 7: "Normal"}

    print("各類別切分結果：")
    print(f"{'類別':<10} {'總數':>4} {'train':>6} {'val':>4} {'test':>4}")
    print("-" * 32)

    for label in sorted(groups.keys()):
        patients = groups[label].copy()
        random.shuffle(patients)

        n = len(patients)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        # 確保 val 和 test 至少各有 1 人（當總數 >= 3 時）
        if n >= 3:
            n_val  = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            n_train = n - n_val - n_test
        else:
            n_test = n - n_train - n_val

        train += patients[:n_train]
        val   += patients[n_train:n_train + n_val]
        test  += patients[n_train + n_val:]

        name = label_names.get(label, f"Label{label}")
        print(f"{name:<10} {n:>4} {n_train:>6} {n_val:>4} {n - n_train - n_val:>4}")

    print("-" * 32)
    print(f"{'合計':<10} {len(train)+len(val)+len(test):>4} {len(train):>6} {len(val):>4} {len(test):>4}\n")

    return train, val, test


def process_patient(pid, part_path, split):
    """處理一個病人的所有影像，存到對應的 split 資料夾"""
    img_dir = os.path.join(output_path, "images", split)
    lbl_dir = os.path.join(output_path, "labels", split)

    mat_data = scipy.io.loadmat(os.path.join(part_path, f"{pid}.mat"))
    patient_data = mat_data[pid]
    im_fields = [k for k in patient_data.dtype.names if k.startswith("im")]

    for im_key in im_fields:
        try:
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
        except Exception as e:
            print(f"  Error {pid} {im_key}: {e}")


# --- 建立輸出資料夾 ---
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_path, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", split), exist_ok=True)

# --- 依 label 分組並做分層切分 ---
groups = collect_patients_by_label()
train_patients, val_patients, test_patients = stratified_split(groups)

# --- 處理並存檔 ---
for split, patients in [("train", train_patients), ("val", val_patients), ("test", test_patients)]:
    print(f"處理 {split}...")
    for pid, part_path in tqdm(patients):
        process_patient(pid, part_path, split)

# --- 統計影像數量 ---
print()
for split in ["train", "val", "test"]:
    n = len(os.listdir(os.path.join(output_path, "images", split)))
    print(f"{split}：{n} 張影像")

print("\n完成！每個 split 都保有各類別的比例。")
