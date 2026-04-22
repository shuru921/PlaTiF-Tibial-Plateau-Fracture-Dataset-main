import os
import random
import shutil
from pathlib import Path

# --- 設定路徑 ---
dataset_path = "yolo_dataset"
images_dir = os.path.join(dataset_path, "images/train") # 假設原本全都在 train 資料夾
labels_dir = os.path.join(dataset_path, "labels/train")

# 我們要輸出的資料夾結構
output_images_train = os.path.join(dataset_path, "images/train_split")
output_images_val = os.path.join(dataset_path, "images/val")
output_images_test = os.path.join(dataset_path, "images/test")

output_labels_train = os.path.join(dataset_path, "labels/train_split")
output_labels_val = os.path.join(dataset_path, "labels/val")
output_labels_test = os.path.join(dataset_path, "labels/test")

# 建立目標資料夾
for folder in [output_images_train, output_images_val, output_images_test,
               output_labels_train, output_labels_val, output_labels_test]:
    os.makedirs(folder, exist_ok=True)

# 取得所有影像檔名 (不含副檔名)
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
file_basenames = [os.path.splitext(f)[0] for f in image_files]

# 隨機打亂資料
random.seed(42)
random.shuffle(file_basenames)

# 設定切分比例 (例如 Train: 80%, Val: 10%, Test: 10%)
total_count = len(file_basenames)
train_split = int(total_count * 0.8)
val_split = int(total_count * 0.9)

train_files = file_basenames[:train_split]
val_files = file_basenames[train_split:val_split]
test_files = file_basenames[val_split:]

def move_files(file_list, target_img_dir, target_lbl_dir):
    for f in file_list:
        # Move image
        src_img = os.path.join(images_dir, f + ".jpg")
        dst_img = os.path.join(target_img_dir, f + ".jpg")
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)

        # Move label
        src_lbl = os.path.join(labels_dir, f + ".txt")
        dst_lbl = os.path.join(target_lbl_dir, f + ".txt")
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)

# 複製檔案到對應的資料夾
print(f"總資料數: {total_count}")
print("開始分配檔案...")
move_files(train_files, output_images_train, output_labels_train)
print(f"Train 資料完成: {len(train_files)} 筆")

move_files(val_files, output_images_val, output_labels_val)
print(f"Val 資料完成: {len(val_files)} 筆")

move_files(test_files, output_images_test, output_labels_test)
print(f"Test 資料完成: {len(test_files)} 筆")

print("切分完成！請記得更新您的 data.yaml 路徑。")
