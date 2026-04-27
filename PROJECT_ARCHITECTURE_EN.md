# Tibial Plateau Fracture AI-Assisted Diagnosis System — Project Architecture

## Project Goal

Given a single emergency department (ED) X-ray image, automatically determine whether a tibial plateau fracture is present, localize it with a bounding region, and provide a confidence score.
For cases where the model is uncertain ("hard cases"), a second-stage model performs a more precise re-evaluation to help ED physicians reduce missed diagnoses.

> Schatzker fracture classification is the responsibility of orthopedic surgeons. This system focuses solely on binary fracture detection: **fracture vs. normal**.

---

## Dataset

**Source:** PlaTiF Dataset (`.mat` format, 186 patients, 421 images)

### .mat Field Description

| Field | Description | Usage |
|-------|-------------|-------|
| `OriginalImage` | Raw grayscale X-ray image | Stage 1 input |
| `BW` | Tibia plateau contour mask (whole tibia, not fracture line) | Converted to YOLO Segmentation polygon labels |
| `maskedImage` | Background-removed tibia image (OriginalImage × BW) | Stage 2 training input |
| `label` | 1~6 = Schatzker fracture type, 7 = Normal | Training label |

> `BW` represents the entire tibia contour, not the fracture line annotation.
> Stage 1 uses `BW` to train tibia segmentation; at inference time, the generated mask is passed to Stage 2.

### Dataset Statistics

| Class | Patients | Images |
|-------|----------|--------|
| Type 1 | 26 | 65 |
| Type 2 | 34 | 70 |
| Type 3 | 12 | 26 |
| Type 4 | 10 | 17 |
| Type 5 | 13 | 22 |
| Type 6 | 33 | 82 |
| Normal | 58 | 139 |
| **Total** | **186** | **421** |

### Data Split Strategy

- **Unit:** Patient-level split (prevents the same patient from appearing in multiple splits)
- **Method:** Stratified sampling (each split maintains proportion of all Schatzker types)
- **Ratio:** Train 80% / Val 10% / Test 10%, fixed seed=42
- **Both stages use the identical split** (prevents data leakage)

| Split | Patients | Fracture Images | Normal Images | Total |
|-------|----------|----------------|---------------|-------|
| Train | ~149 | 223 | 113 | 336 |
| Val | ~18 | 24 | 13 | 37 |
| Test | ~19 | 30 | 18 | 48 |

---

## System Architecture

```
Input: X-ray image (OriginalImage)
        ↓
┌───────────────────────────────────────┐
│  Stage 1: YOLO Segmentation           │
│  Model: YOLOv8n-seg                   │
│  Input: OriginalImage (640×640)       │
│  Output:                              │
│    - Fracture present / absent        │
│    - Tibia contour Mask               │
│    - Confidence Score                 │
└───────────────────────────────────────┘
        ↓
   Confidence threshold check
        ├── High confidence (> 0.7) ──────────→ Output result to ED physician
        │
        └── Low confidence (0.3~0.7, hard case)
                ↓
        Apply YOLO mask → background-removed tibia image
                ↓
┌──────────────────────────────────────────┐
│  Stage 2: Refined Binary Classification  │
│  Model: ResNet18 / EfficientNet-B0       │
│  Input: Masked tibia image (224×224)     │
│  Goal: Remove background noise,          │
│        focus on bone detail              │
│  Output: Fracture / Normal (more precise)│
└──────────────────────────────────────────┘
        ↓
   Final output to ED physician
```

---

## Preprocessing Pipeline

### Stage 1 Preprocessing (`preprocess_for_yolo.py`) ✅ Done

```
Raw .mat data
    ↓
1. Read OriginalImage
2. Min-Max normalization (scale to 0~255)
3. CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=8×8)
4. Read BW mask → extract contour points → normalize → YOLO Segmentation format
   (Normal images: label=7 → empty .txt as negative sample)
    ↓
yolo_seg_dataset/
├── images/train|val|test/*.jpg   (processed OriginalImage)
└── labels/train|val|test/*.txt   (YOLO polygon format or empty)
```

**YOLO Segmentation label format:**
```
class_id  x1 y1  x2 y2  x3 y3  ... (normalized tibia contour vertices, 0~1)
0         0.283 0.539  0.278 0.543  0.277 0.546  ...
```
- class_id = 0 (single class: Fracture)
- Polygon vertices derived from tibia BW contour, simplified with `approxPolyDP`
- Normal images: empty `.txt`

### Stage 2 Preprocessing (`prepare_stage2.py`) ✅ Done

```
Raw .mat data (same patient split as Stage 1)
    ↓
1. Read maskedImage (background-removed tibia image)
2. Min-Max normalization + CLAHE (same parameters as Stage 1)
3. Crop black borders (keep only bone region)
4. Save by label (fracture: label 1~6 / normal: label 7)
    ↓
stage2_dataset/
├── train/fracture|normal/*.jpg
├── val/fracture|normal/*.jpg
└── test/fracture|normal/*.jpg
```

**Stage 2 labels:**
- `fracture/`: label 1~6 (any Schatzker type is treated as fracture)
- `normal/`: label 7
- Folder name serves as the label — no additional annotation files needed

---

## Training Pipeline

### Step 1: Train Stage 1 (YOLO Segmentation)

| Setting | Value |
|---------|-------|
| Model | YOLOv8n-seg |
| Pretrained weights | COCO |
| Input resolution | 640×640 |
| Number of classes | 1 (Fracture) |
| Training command | `yolo segment train data=data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640` |
| Evaluation metrics | Precision, Recall, mAP@0.5, Mask mAP |

### Step 2: Hard Case Analysis

- Run inference on the test set
- Identify images with confidence score in the range 0.3~0.7 (hard cases)
- Cross-reference with `.mat` labels to determine which Schatzker types are most difficult
- Used to understand model weaknesses (analysis only, not part of final output)

### Step 3: Train Stage 2 (Refined Binary Classifier)

| Setting | Value |
|---------|-------|
| Model | ResNet18 / EfficientNet-B0 |
| Pretrained weights | ImageNet (Transfer Learning) |
| Input resolution | 224×224 |
| Number of classes | 2 (fracture / normal) |
| Training strategy | Freeze early layers first, then fine-tune all layers |
| Data Augmentation | Rotation, flipping, sharpening (compensate for small dataset) |
| Evaluation metrics | Accuracy, Precision, Recall, AUC |

### Step 4: Pipeline Integration and Overall Evaluation

- Stage 1 confidence below threshold → apply YOLO mask → trigger Stage 2
- Compare overall Recall before and after adding Stage 2
- Key focus: improvement on hard cases

---

## Key Design Decisions

| Question | Decision | Reason |
|----------|----------|--------|
| Detection vs. Segmentation | **Segmentation** | Simultaneously detects fracture and generates tibia mask for Stage 2 |
| Stage 2 input source | YOLO-generated mask at inference time | Real deployment has no `.mat` files; mask must be derived from raw X-ray |
| Why not segment fracture lines | No annotation available | `.mat` only contains tibia outline mask, not fracture line pixel annotations |
| Why no Schatzker classification | Not part of system output | Classification is the orthopedic surgeon's role; this system serves ED physicians |
| Why Transfer Learning | Small dataset (421 images) | Training from scratch leads to overfitting; ImageNet features transfer well |

---

## Project File Structure

```
PlaTiF-Tibial-Plateau-Fracture-Dataset-main/
├── PlaTiF Dataset/              # Raw .mat data (5 parts)
├── yolo_seg_dataset/            # Stage 1 dataset (YOLO Segmentation) ✅
│   ├── images/train|val|test/
│   └── labels/train|val|test/
├── stage2_dataset/              # Stage 2 dataset (classifier) ✅
│   ├── train/fracture|normal/
│   ├── val/fracture|normal/
│   └── test/fracture|normal/
├── data.yaml                    # YOLO configuration file
├── preprocess_for_yolo.py       # Stage 1 preprocessing script
├── prepare_stage2.py            # Stage 2 preprocessing script
├── Read_Data_PythonCode.py      # Data visualization (Python)
├── Read_Data_MatlabCode.m       # Data visualization (MATLAB)
├── PROJECT_ARCHITECTURE.md      # This document (Chinese)
└── PROJECT_ARCHITECTURE_EN.md   # This document (English)
```

---

## Evaluation Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Recall (Sensitivity)** | Proportion of actual fractures correctly detected | Most critical — missed fractures are far more harmful than false alarms |
| Precision | Proportion of predicted fractures that are truly fractures | Avoids unnecessary CT referrals |
| mAP@0.5 | Overall YOLO detection performance | Primary Stage 1 metric |
| Mask mAP | Tibia contour segmentation accuracy | Stage 1 segmentation quality |
| AUC | Overall classifier performance | Primary Stage 2 metric |

---

## Milestones

- [x] Stage 1 dataset preprocessing complete (`yolo_seg_dataset/`)
- [x] Stage 2 dataset preprocessing complete (`stage2_dataset/`)
- [ ] Stage 1 YOLO Segmentation training complete, baseline confirmed
- [ ] Hard case analysis (identify which Schatzker types are hardest to detect)
- [ ] Stage 2 classifier training complete
- [ ] Two-stage pipeline integration and overall evaluation
