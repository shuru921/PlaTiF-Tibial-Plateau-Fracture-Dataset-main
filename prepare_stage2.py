import os
import cv2
import numpy as np
import scipy.io
from tqdm import tqdm

# --- 路徑設定 ---
base_path = "./PlaTiF Dataset"
yolo_img_path = "./yolo_dataset/images"
output_path = "./stage2_dataset"

NORMAL_LABEL = 7


def normalize_image(img):
    img_float = img.astype(np.float32)
    img_norm = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX)
    img_8u = img_norm.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_8u)


def crop_to_content(img, padding=20):
    """裁切掉黑色邊框，只保留骨頭區域"""
    non_zero = np.where(img > 0)
    if len(non_zero[0]) == 0:
        return img
    y_min, y_max = non_zero[0].min(), non_zero[0].max()
    x_min, x_max = non_zero[1].min(), non_zero[1].max()
    y_min = max(0, y_min - padding)
    y_max = min(img.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(img.shape[1], x_max + padding)
    return img[y_min:y_max, x_min:x_max]


def get_patient_splits():
    """從現有 yolo_dataset 的檔名反推每個 split 有哪些病人"""
    splits = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(yolo_img_path, split)
        pids = set()
        for f in os.listdir(split_dir):
            # 檔名格式：Patient_ID_001_im0.jpg → 取前三段
            parts = f.replace(".jpg", "").rsplit("_im", 1)
            if len(parts) == 2:
                pids.add(parts[0])  # e.g. Patient_ID_001
        splits[split] = sorted(pids)
        print(f"{split}: {len(pids)} 位病人")
    return splits


def find_mat_file(pid):
    """在各 Part 資料夾中找到對應病人的 .mat 檔"""
    part_dirs = sorted([
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and "Patient Data" in d
    ])
    for part in part_dirs:
        mat_path = os.path.join(base_path, part, f"{pid}.mat")
        if os.path.exists(mat_path):
            return mat_path
    return None


def process_patient(pid, split):
    mat_path = find_mat_file(pid)
    if mat_path is None:
        print(f"  找不到 {pid}.mat，跳過")
        return

    mat_data = scipy.io.loadmat(mat_path)
    patient_data = mat_data[pid]
    im_fields = [k for k in patient_data.dtype.names if k.startswith("im")]

    for im_key in im_fields:
        try:
            im_data = patient_data[0, 0][im_key][0, 0]
            masked_img = im_data["maskedImage"]
            label = int(im_data["label"][0, 0])

            category = "normal" if label == NORMAL_LABEL else "fracture"
            save_dir = os.path.join(output_path, split, category)

            img_processed = normalize_image(masked_img)
            img_cropped = crop_to_content(img_processed)

            filename = f"{pid}_{im_key}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), img_cropped)

        except Exception as e:
            print(f"  Error {pid} {im_key}: {e}")


# --- 建立輸出資料夾 ---
for split in ["train", "val", "test"]:
    for category in ["fracture", "normal"]:
        os.makedirs(os.path.join(output_path, split, category), exist_ok=True)

# --- 取得切分 ---
print("讀取病人切分...")
splits = get_patient_splits()

# --- 處理每位病人 ---
for split, pids in splits.items():
    print(f"\n處理 {split}...")
    for pid in tqdm(pids):
        process_patient(pid, split)

# --- 統計結果 ---
print("\n=== 完成！資料集統計 ===")
print(f"{'':10} {'fracture':>10} {'normal':>10} {'total':>8}")
print("-" * 42)
for split in ["train", "val", "test"]:
    f_count = len(os.listdir(os.path.join(output_path, split, "fracture")))
    n_count = len(os.listdir(os.path.join(output_path, split, "normal")))
    print(f"{split:<10} {f_count:>10} {n_count:>10} {f_count+n_count:>8}")
