"""
YOLO Segmentation Label 視覺化驗證腳本
- 將 polygon mask 疊加在原圖上
- 輸出到 vis_output/ 資料夾，方便批量瀏覽
用法：
    python visualize_labels.py              # 預設 train split，最多 50 張
    python visualize_labels.py --split val  # 改看 val
    python visualize_labels.py --all        # 全部輸出（可能很多）
    python visualize_labels.py --show       # 邊輸出邊跳出視窗即時看
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent
DATASET = BASE / "yolo_seg_dataset"
OUTPUT = BASE / "vis_output"

# 每個 class 的顏色（目前只有 Fracture class 0）
CLASS_COLORS = {
    0: (0, 255, 100),   # 綠色
}
CLASS_NAMES = {0: "Fracture"}


def draw_seg_label(img: np.ndarray, label_path: Path) -> tuple[np.ndarray, int]:
    """將一個 label txt 的所有 polygon 畫到 img 上，回傳 (annotated_img, polygon_count)"""
    h, w = img.shape[:2]
    overlay = img.copy()
    count = 0

    if not label_path.exists() or label_path.stat().st_size == 0:
        # 空白 label = Normal（無骨折），加浮水印
        cv2.putText(img, "NORMAL (no label)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return img, 0

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))

        if len(coords) < 6:  # 至少 3 個點
            continue

        # 反正規化到像素座標
        pts = np.array(
            [[int(coords[i] * w), int(coords[i + 1] * h)]
             for i in range(0, len(coords), 2)],
            dtype=np.int32,
        )

        color = CLASS_COLORS.get(cls, (255, 80, 80))
        # 填色半透明
        cv2.fillPoly(overlay, [pts], color)
        # 輪廓線
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        # class 標籤
        label_str = CLASS_NAMES.get(cls, f"cls{cls}")
        cv2.putText(img, label_str, (pts[0][0], pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        count += 1

    # 半透明疊合
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    return img, count


def process_split(split: str, max_imgs: int | None, show: bool):
    img_dir = DATASET / "images" / split
    lbl_dir = DATASET / "labels" / split
    out_dir = OUTPUT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if max_imgs is not None:
        # 優先挑有 label 的
        labeled = [p for p in img_paths if (lbl_dir / (p.stem + ".txt")).stat().st_size > 0
                   if (lbl_dir / (p.stem + ".txt")).exists()]
        unlabeled = [p for p in img_paths if p not in labeled]
        # 取 max_imgs 張，盡量包含有標注的
        selected = labeled[:max_imgs]
        if len(selected) < max_imgs:
            selected += unlabeled[:max_imgs - len(selected)]
        img_paths = sorted(selected)

    print(f"\n[{split}] 處理 {len(img_paths)} 張影像 → {out_dir}")
    empty_count = poly_count = 0

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] 無法讀取 {img_path.name}")
            continue

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        img, n = draw_seg_label(img, lbl_path)

        if n == 0:
            empty_count += 1
        else:
            poly_count += n

        # 左上角顯示檔名
        cv2.putText(img, img_path.stem, (5, img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img)

        if show:
            cv2.imshow("Label Check  (press any key: next  |  q: quit)", img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()

    print(f"  有標注影像: {len(img_paths) - empty_count}  |  空白(Normal): {empty_count}  |  polygon 總數: {poly_count}")
    print(f"  輸出位置: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--max", type=int, default=50, help="最多輸出幾張（0 = 全部）")
    parser.add_argument("--all", action="store_true", help="輸出全部影像（忽略 --max）")
    parser.add_argument("--show", action="store_true", help="用 OpenCV 視窗即時顯示")
    args = parser.parse_args()

    max_imgs = None if args.all or args.max == 0 else args.max
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        process_split(split, max_imgs, args.show)

    print("\n完成！請打開 vis_output/ 資料夾查看結果。")


if __name__ == "__main__":
    main()
