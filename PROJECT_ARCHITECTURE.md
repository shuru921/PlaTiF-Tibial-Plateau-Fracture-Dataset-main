# 脛骨平台骨折 AI 輔助診斷系統 — 專案架構

## 專案目標

輸入一張急診 X 光影像，自動判斷是否有脛骨平台骨折、標出位置並給出信心分數。
針對模型不確定的「困難案例」，透過第二階段模型進行更精細的判斷，輔助急診醫師降低漏診率。

> Schatzker 骨折分型屬於骨科醫師的判斷範疇，本系統專注在「有無骨折」的二元判斷。

---

## 資料集

**來源：** PlaTiF Dataset（`.mat` 格式，186 位病人，421 張影像）

### .mat 欄位說明

| 欄位 | 說明 | 用途 |
|------|------|------|
| `OriginalImage` | 原始 X 光灰階影像 | Stage 1 輸入 |
| `BW` | 脛骨平台輪廓 mask（整塊脛骨，非骨折線） | 轉換為 YOLO Segmentation 多邊形 label |
| `maskedImage` | 去除背景後的純脛骨影像（OriginalImage × BW） | Stage 2 訓練輸入 |
| `label` | 1~6 = Schatzker 骨折分型，7 = 正常 | 訓練 label |

> `BW` 是整個脛骨的輪廓，不是骨折線的標註。
> Stage 1 用 `BW` 訓練脛骨 segmentation，推論時即時產生 mask 給 Stage 2 使用。

### 資料統計

| 類別 | 病人數 | 影像數 |
|------|--------|--------|
| Type 1 | 26 | 65 |
| Type 2 | 34 | 70 |
| Type 3 | 12 | 26 |
| Type 4 | 10 | 17 |
| Type 5 | 13 | 22 |
| Type 6 | 33 | 82 |
| Normal | 58 | 139 |
| **合計** | **186** | **421** |

### 資料切分策略

- **單位：** 以病人為單位切分（避免同一病人出現在不同 split）
- **方法：** 分層抽樣（每個 split 保有各 Schatzker Type）
- **比例：** Train 80% / Val 10% / Test 10%，固定 seed=42
- **兩個階段使用完全相同的切分**（避免 Data Leakage）

| Split | 病人數 | 影像數（有骨折） | 影像數（正常） | 總計 |
|-------|--------|----------------|--------------|------|
| Train | ~149 | 223 | 113 | 336 |
| Val | ~18 | 24 | 13 | 37 |
| Test | ~19 | 30 | 18 | 48 |

---

## 系統架構

```
輸入：X 光影像（OriginalImage）
        ↓
┌───────────────────────────────────────┐
│  Stage 1：YOLO Segmentation           │
│  模型：YOLOv8n-seg                    │
│  輸入：OriginalImage（640×640）        │
│  輸出：                                │
│    - 有無骨折                          │
│    - 脛骨輪廓 Mask（BW 輪廓）          │
│    - 信心分數（Confidence Score）      │
└───────────────────────────────────────┘
        ↓
   信心分數判斷
        ├── 高信心（> 0.7）─────────────────→ 輸出結果給急診醫師
        │
        └── 低信心（0.3~0.7，困難案例）
                ↓
        用 YOLO 產生的 Mask 去背 → 純脛骨影像
                ↓
┌──────────────────────────────────────────┐
│  Stage 2：精細二元判斷                    │
│  模型：ResNet18 / EfficientNet-B0         │
│  輸入：純脛骨影像（224×224）              │
│  目的：排除背景干擾，專注脛骨細節         │
│  輸出：有骨折 / 無骨折（更精確的判斷）   │
└──────────────────────────────────────────┘
        ↓
   最終輸出給急診醫師
```

---

## 前處理流程

### Stage 1 前處理（`preprocess_for_yolo.py`）✅ 已完成

```
.mat 原始資料
    ↓
1. 讀取 OriginalImage
2. Min-Max 正規化（縮放至 0~255）
3. CLAHE 對比增強（clipLimit=2.0, tileGridSize=8×8）
4. 讀取 BW mask → 取輪廓頂點 → 正規化 → YOLO Segmentation 格式
   （正常影像 label=7 → 空白 .txt 負樣本）
    ↓
yolo_seg_dataset/
├── images/train|val|test/*.jpg   （OriginalImage 處理後）
└── labels/train|val|test/*.txt   （YOLO 多邊形格式 或 空白）
```

**YOLO Segmentation label 格式：**
```
class_id  x1 y1  x2 y2  x3 y3  ...（脛骨輪廓頂點，正規化 0~1）
0         0.283 0.539  0.278 0.543  0.277 0.546  ...
```
- class_id = 0（唯一類別：Fracture）
- 多邊形頂點為脛骨 BW 輪廓，已用 `approxPolyDP` 簡化
- 正常影像：空白 .txt

### Stage 2 前處理（`prepare_stage2.py`）✅ 已完成

```
.mat 原始資料（與 Stage 1 相同病人切分）
    ↓
1. 讀取 maskedImage（已去背景的純脛骨影像）
2. Min-Max 正規化 + CLAHE（參數同 Stage 1）
3. 裁切黑色邊框（只保留骨頭區域，減少無效資訊）
4. 依 label 存放（fracture: label 1~6 / normal: label 7）
    ↓
stage2_dataset/
├── train/fracture|normal/*.jpg
├── val/fracture|normal/*.jpg
└── test/fracture|normal/*.jpg
```

**Stage 2 的 label：**
- `fracture/`：label 1~6（任何 Schatzker Type 都歸為有骨折）
- `normal/`：label 7
- 資料夾名稱即為 label，不需要額外標註檔

---

## 訓練流程

### Step 1：訓練 Stage 1（YOLO Segmentation）

| 設定 | 內容 |
|------|------|
| 模型 | YOLOv8n-seg |
| 預訓練 | COCO pretrained weights |
| 輸入解析度 | 640×640 |
| 類別數 | 1（Fracture） |
| 訓練指令 | `yolo segment train data=data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640` |
| 評估指標 | Precision、Recall、mAP@0.5、Mask mAP |

### Step 2：分析困難案例

- 對 test 集跑 inference
- 找出 confidence 落在 0.3~0.7 的影像（困難案例）
- 對照 `.mat` label 分析是哪些 Schatzker Type
- 作為了解模型弱點的依據（分析用，不作為輸出）

### Step 3：訓練 Stage 2（精細二元判斷模型）

| 設定 | 內容 |
|------|------|
| 模型 | ResNet18 / EfficientNet-B0 |
| 預訓練 | ImageNet pretrained（Transfer Learning） |
| 輸入解析度 | 224×224 |
| 類別數 | 2（fracture / normal） |
| 訓練策略 | 先凍結前幾層，訓練穩定後再全部微調 |
| Data Augmentation | 旋轉、翻轉、銳化（針對資料量少） |
| 評估指標 | Accuracy、Precision、Recall、AUC |

### Step 4：串接兩個模型，整體評估

- Stage 1 confidence 低於門檻 → 用 YOLO 產生的 mask 去背 → 觸發 Stage 2
- 比較加入 Stage 2 前後的整體 Recall
- 重點評估：困難案例的改善幅度

---

## 關鍵設計決策

| 問題 | 決策 | 原因 |
|------|------|------|
| Detection vs Segmentation | **Segmentation** | 同時偵測骨折與割出脛骨輪廓，供 Stage 2 使用 |
| Stage 2 輸入來源 | YOLO 即時產生的 mask | 真實部署時沒有 `.mat`，需從 X 光直接產生 |
| 為何不做骨折線 Segmentation | 無標註資料 | `.mat` 只有脛骨輪廓 mask，無骨折線像素標註 |
| 為何不做 Schatzker 分型 | 不在系統輸出 | 分型是骨科醫師的職責，本系統服務急診 |
| 為何用 Transfer Learning | 資料量少（421 張） | 從頭訓練會過擬合，ImageNet 特徵可遷移 |

---

## 專案檔案結構

```
PlaTiF-Tibial-Plateau-Fracture-Dataset-main/
├── PlaTiF Dataset/              # 原始 .mat 資料（5 個 Part）
├── yolo_seg_dataset/            # Stage 1 資料集（YOLO Segmentation）✅
│   ├── images/train|val|test/
│   └── labels/train|val|test/
├── stage2_dataset/              # Stage 2 資料集（分類模型）✅
│   ├── train/fracture|normal/
│   ├── val/fracture|normal/
│   └── test/fracture|normal/
├── data.yaml                    # YOLO 設定檔
├── preprocess_for_yolo.py       # Stage 1 前處理腳本
├── prepare_stage2.py            # Stage 2 前處理腳本
├── Read_Data_PythonCode.py      # 原始資料視覺化
├── Read_Data_MatlabCode.m       # 原始資料視覺化（MATLAB）
└── PROJECT_ARCHITECTURE.md      # 本文件
```

---

## 評估重點

| 指標 | 說明 | 為什麼重要 |
|------|------|-----------|
| **Recall（敏感度）** | 骨折有多少被抓到 | 最重要，漏診的危害遠大於誤診 |
| Precision | 預測骨折中真正是骨折的比例 | 避免不必要的 CT 檢查 |
| mAP@0.5 | YOLO 整體偵測表現 | Stage 1 主要指標 |
| Mask mAP | 脛骨輪廓分割精準度 | Stage 1 分割品質指標 |
| AUC | 分類模型綜合表現 | Stage 2 主要指標 |

---

## 里程碑

- [x] Stage 1 資料集前處理完成（`yolo_seg_dataset/`）
- [x] Stage 2 資料集前處理完成（`stage2_dataset/`）
- [ ] Stage 1 YOLO Segmentation 訓練完成，baseline 結果確認
- [ ] 困難案例分析（哪些 Schatzker Type 較難偵測）
- [ ] Stage 2 分類模型訓練完成
- [ ] 兩階段串接，整體系統評估
