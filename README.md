PlaTiF Dataset: Code Usage Guide 💻🦴

Welcome to the PlaTiF (Tibial Plateau Fracture) Dataset repository\! 🎉 This guide will help you navigate and utilize the MATLAB and Python scripts we've provided to explore the fascinating world of tibial plateau fracture images and their segmentation masks.

🗺️ Your PlaTiF Journey: A Quick Roadmap

1.  **Clone This Repository**: Get all the goodies onto your machine\! 📥
2.  **Download the Dataset:** Grab the actual patient data ([Access Link to PlaTiF Dataset](https://doi.org/10.5281/zenodo.18007397)). 💾
3.  **Choose Your Path:** Pythonista or MATLAB guru? Pick your preferred language\! 🐍📊
4.  **Install Dependencies:** A few quick installs to get everything running smoothly. 🛠️
5.  **Run & Visualize:** Read and plot X-ray image, Tibia mask, and Schatzker classification label. 🖼️

## 🚧 General Setup: Getting Started is Easy\!

First things first, let's get your environment ready.

### Step 1: Clone the Repository 📥

Open your terminal or command prompt and run this command:

```bash
git clone https://github.com/ali-kazemi8/PlaTiF-Tibial-Plateau-Fracture-Dataset.git
cd PlaTiF-Tibial-Plateau-Fracture-Dataset
```

Voila\! You now have all the scripts locally. ✨

### Step 2: Download the PlaTiF Dataset 💾

Our scripts need data to shine\!

  * Make sure you have downloaded the actual PlaTiF dataset.
  * Keep the `.mat` files in a readily accessible location.
  * **Important:** You'll need to point our scripts to the specific patient `.mat` files you want to visualize.

### Data Structure at a Glance 🧩

Each `.mat` file is a treasure trove of information, organized neatly into a structured variable (e.g., `Patient_ID_001`) containing:

📁 Patient_ID_XXX

├── 🧾 im0:
🖼️ OriginalImage → Original X-ray image
│ ⚫ BW → Binary mask of segmented tibial plateau
│ 🖼️ maskedImage → X-ray masked with segmentation
│ 🏷️ label → Class label (1–7) for Schatzker fracture type

├── 🧾 im1:
…

└── 📐 Coronal_CT (optional) → Associated CT image if available

-----

## 🐍 Python Power-Up: For the Python Enthusiasts\!

Our Python scripts are located in the `Python/` directory. Get ready to `import` and explore\!

### Step 1: Set Up Your Python Playground 🏞️

We highly recommend using a virtual environment to keep your project dependencies clean and tidy.

```bash
# Navigate into the Python directory
cd Python

# Create a virtual environment (if you don't have one already)
python -m venv venv

# Activate your new environment!

# On macOS/Linux:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate
```

You'll see `(venv)` appear in your terminal prompt, indicating your environment is active\! 🎉

### Step 2: Install Essential Libraries 📚

We've made it super easy\! We've provided a `requirements.txt` file with everything you need.

```
# Python/requirements.txt
numpy
scipy
matplotlib
```

With your virtual environment active, simply run:

```bash
pip install -r requirements.txt
```

And just like that, you're ready\! ✅

### Step 3: Run Our Sample Visualization Script\! 🚀

We've prepared a friendly script, `Read_Data_PythonCode.py`, to help you visualize a patient's X-ray image and tibia bone mask.

*(If you haven't already, create `Python/Read_Data_PythonCode.py` and paste the following content into it.)*

```python
import os
import scipy.io
import matplotlib.pyplot as plt

# --- 1. Setup: Define file path and patient ID ---
base_path = r'C:\Users\Ali\Desktop\PlaTiF Dataset'
patient_id = 'Patient_ID_001'
file_path = os.path.join(base_path, f'{patient_id}.mat')

# --- 2. Load Data ---
mat_data = scipy.io.loadmat(file_path)
patient_data = mat_data[patient_id]

# --- 3. Access Image Data from the 'im0' struct ---
im0_data = patient_data[0, 0]['im0']
original_img = im0_data[0, 0]['OriginalImage']
bw_mask = im0_data[0, 0]['BW']
masked_img = im0_data[0, 0]['maskedImage']
Schatzker_label = im0_data[0, 0]['label'][0, 0]

# --- 4. Visualize the Data (with conditional plotting) ---

# Check if Coronal_CT exists to determine subplot layout
if 'Coronal_CT' in patient_data.dtype.names:
    num_cols = 4
    fig, axes = plt.subplots(1, num_cols, figsize=(20, 5)) # Wider figure for 4 plots
    coronal_ct_image = patient_data[0, 0]['Coronal_CT']
else:
    num_cols = 3
    fig, axes = plt.subplots(1, num_cols, figsize=(18, 6)) # Original figure size

fig.suptitle(f'Data for {patient_id}', fontsize=16)

# Plot the common images
axes[0].imshow(original_img, cmap='gray')
axes[0].set_title(f'Schatzker Classification Label: {Schatzker_label}')
axes[0].axis('off')

axes[1].imshow(bw_mask, cmap='gray')
axes[1].set_title('Tibia Bone Plateau Mask')
axes[1].axis('off')

axes[2].imshow(masked_img, cmap='gray')
axes[2].set_title('Segmented Tibia Bone')
axes[2].axis('off')

# Plot the optional fourth image only if it exists
if num_cols == 4:
    axes[3].imshow(coronal_ct_image, cmap='gray')
    axes[3].set_title('A Coronal CT Slice')
    axes[3].axis('off')

plt.tight_layout()
plt.show()
```

**How to Run It:**

Navigate to the `Python/` directory in your terminal (with your `venv` active) and try these commands:

  * **To see a specific X-ray knee image (e.g., im0 in Patient_ID_001):**
    ```bash
    python Read_Data_PythonCode.py --file /path/to/your/dataset/Patient_ID_001.mat --im0
    ```
    You'll see a beautiful side-by-side comparison\! 🖼️
  * **To explore the X-ray image:**
    ```bash
    python Read_Data_PythonCode.py --file /path/to/your/dataset/Patient_ID_001.mat
    ```
-----

## 📊 MATLAB Magic: For the MATLAB Aficionados\!

Our MATLAB scripts reside in the `MATLAB/` directory. Let's get them running\!

### Step 1: MATLAB Environment Prep ⚙️

1.  **Open MATLAB:** Launch your MATLAB application.
2.  **Toolbox Check:** Ensure you have the **Image Processing Toolbox** installed. It's crucial for functions like `imshow` and `visboundaries` to display images and outlines beautifully. If not, you might need to install it via the Add-On Explorer in MATLAB. 📥

### Step 2: Unleash the Visualization Script\! ✨

We've crafted `Read_Data_MatlabCode.m` to make visualizing your data a breeze.

*(If you haven't already, create `MATLAB/Read_Data_MatlabCode.m` and paste the following content into it.)*

```matlab
clc
clear
close all

% --- 1. Setup: Define file path and patient ID ---
basePath = 'C:\Users\Ali\Desktop\PlaTiF Dataset';
patientID = 'Patient_ID_001';
filePath = fullfile(basePath, [patientID, '.mat']);

% --- 2. Load Data ---
dataStruct = load(filePath);
patientData = dataStruct.(patientID);

% --- 3. Access Image Data from the 'im0' struct ---
im0_data = patientData.im0;
originalImg = im0_data.OriginalImage;
bwMask = im0_data.BW;
maskedImg = im0_data.maskedImage;
SchatzkerLabel = im0_data.label;

% --- 4. Visualize the Data (with conditional plotting) ---
figure('Name', ['Data for ', patientID]);

% Check if Coronal_CT exists to determine subplot layout
if isfield(patientData, 'Coronal_CT')
    numCols = 4;
    coronalCT_image = patientData.Coronal_CT;
else
    numCols = 3;
end

% Subplot 1: Original Image
subplot(1, numCols, 1);
imshow(originalImg, []);
title(['Schatzker Classification Label: ', num2str(SchatzkerLabel)]);

% Subplot 2: Tibia Mask (BW)
subplot(1, numCols, 2);
imshow(bwMask);
title('Tibia Bone Plateau Mask');

% Subplot 3: Masked Image
subplot(1, numCols, 3);
imshow(maskedImg, []);
title('Segmented Tibia Bone');

% Plot the optional fourth image only if it exists
if numCols == 4
    subplot(1, numCols, 4);
    imshow(coronalCT_image, []);
    title('A Coronal CT Slice');
end
```

**How to Run It:**

1.  In MATLAB, navigate to the `MATLAB/` directory in the Current Folder browser.

2.  In the MATLAB Command Window, run the script with your file path and desired patient:

-----

Finally, the knee X-ray image is displayed along with the annotation images 🤩

<img width="1366" height="444" alt="Figure_1" src="https://github.com/user-attachments/assets/0680a192-107a-4f49-9fe9-d2b9e63ca831" />


Schatzker Classification System:
<div align="center">
  <img width="592" height="474" alt="inbox_2650414_8f817d7d3f89c2dc9b99b7b5960b2cb1_FotoJet" src="https://github.com/user-attachments/assets/26b4588c-37b6-4241-8766-67ddcfd80b06" />
</div>

-----

---

## 專題實作紀錄

### 注意事項

> **原始 `.mat` 資料檔未包含在此 repo 中**（檔案過大）。
> 若要重新執行前處理，請先至 Zenodo 下載原始資料集：
> [https://doi.org/10.5281/zenodo.18007397](https://doi.org/10.5281/zenodo.18007397)
>
> 下載後將各 Part 資料夾放置於 `PlaTiF Dataset/` 底下，依序執行：
> ```bash
> python preprocess_for_yolo.py   # 產生 Stage 1 資料集
> python prepare_stage2.py        # 產生 Stage 2 資料集
> ```

---

### 資料集概況

- 總病人數：186 人（Part 1~5）
- 類別分布：Normal（無骨折）58 人、骨折（Schatzker Type 1~6）128 人
- 每個病人有 1~N 張影像（im0, im1, im2...），共 421 張

| Schatzker 分類 | 病人數 | 影像數 |
|---------------|--------|--------|
| Normal        | 58     | 139    |
| Type 1        | 26     | 65     |
| Type 2        | 34     | 70     |
| Type 3        | 12     | 26     |
| Type 4        | 10     | 17     |
| Type 5        | 13     | 22     |
| Type 6        | 33     | 82     |

> `.mat` 檔中 label=7 代表 Normal，label=1~6 對應 Schatzker 分型

---

### 專案檔案說明

```
PlaTiF-Tibial-Plateau-Fracture-Dataset-main/
├── PlaTiF Dataset/          # 原始 .mat 資料（Part 1~5，186 個病人）
├── yolo_seg_dataset/        # Stage 1 資料集（YOLO Segmentation）
│   ├── images/train|val|test/
│   └── labels/train|val|test/
├── stage2_dataset/          # Stage 2 資料集（精細二元判斷）
│   ├── train/fracture|normal/
│   ├── val/fracture|normal/
│   └── test/fracture|normal/
├── data.yaml                # YOLO 設定檔
├── preprocess_for_yolo.py   # Stage 1 前處理腳本
├── prepare_stage2.py        # Stage 2 前處理腳本
├── Read_Data_PythonCode.py  # 原始資料視覺化（Python）
├── Read_Data_MatlabCode.m   # 原始資料視覺化（MATLAB）
├── PROJECT_ARCHITECTURE.md  # 完整系統架構文件
└── requirements.txt         # 套件依賴清單
```

---

### 前處理說明

#### Stage 1（`preprocess_for_yolo.py`）

從 `.mat` 讀取 `OriginalImage`，經 Min-Max 正規化與 CLAHE 對比增強後，將脛骨 `BW` mask 的輪廓頂點轉換為 **YOLO Segmentation 多邊形格式**。正常影像（label=7）產生空白 `.txt` 作為負樣本。

**資料切分：以病人為單位，分層抽樣（80/10/10），固定 seed=42**

| Split | 影像數（有骨折） | 影像數（正常） | 總計 |
|-------|----------------|--------------|------|
| train | 223 | 113 | 336 |
| val   | 24  | 13  | 37  |
| test  | 30  | 18  | 48  |

**訓練指令：**
```bash
yolo segment train data=data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
```

#### Stage 2（`prepare_stage2.py`）

與 Stage 1 使用相同病人切分，從 `.mat` 讀取 `maskedImage`（去背景純脛骨影像），經 CLAHE 增強與黑邊裁切後，依 label 存放至 `fracture/` 或 `normal/` 資料夾。

---

### 系統架構

本專案採兩階段 AI 輔助診斷架構，目標是幫助急診醫師判斷是否有骨折：

```
輸入：X 光影像
  ↓
[Stage 1] YOLO Segmentation
  → 有無骨折 + 脛骨輪廓 Mask + 信心分數
  ↓
信心分數高（> 0.7）→ 直接輸出結果
信心分數低（困難案例）
  → 用 Mask 去背得到純脛骨影像 → [Stage 2]
  ↓
[Stage 2] ResNet / EfficientNet 精細二元判斷
  → 輸入：純脛骨影像（去除背景干擾）
  → 輸出：有骨折 / 無骨折（更精確）
```

> 本系統只做有無骨折的判斷，不做 Schatzker 分型（分型為骨科醫師職責）。
> 詳細架構請參考 `PROJECT_ARCHITECTURE.md`。

---

## 💡 Explore and Innovate\!

This code is just a starting point! Feel free to:

 * Modify the Visualization: Experiment with different colormaps (cmap), transparency levels (alpha), or try overlaying the mask contour instead of the full mask. 🎨

 * Batch Processing: Adapt the scripts to automatically process and save visualizations for your entire dataset. 🔄

 * Feature Extraction: Use the segmentation masks to automatically calculate features like fracture area, bone dimensions, or other key morphological measurements. 🔬

 * Data Augmentation: Apply transformations (rotation, scaling, flipping) to your 2D images and masks to expand your dataset for deep learning. 🤖

 * Train a Model: Use this dataset to train a U-Net or other semantic segmentation model to automatically segment the tibia in new X-ray images! 🧠
