import cv2
import numpy as np
from pathlib import Path
import os
import glob


def yolo_seg_to_mask(image_path, label_path):
    """YOLO分割标签 -> mask"""

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not label_path.exists():
        return mask

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        points = []
        for i in range(0, len(coords), 2):
            x = int(min(max(coords[i], 0), 1) * w)
            y = int(min(max(coords[i + 1], 0), 1) * h)
            points.append([x, y])

        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], class_id + 1)

    return mask


def save_overlay(image, mask, save_path):
    """保存叠加图"""

    overlay = image.copy()
    colors = {
        1: np.array([0, 255, 0]),  # class0
        2: np.array([0, 0, 255])  # class1
    }

    for cls, color in colors.items():
        overlay[mask == cls] = (
                overlay[mask == cls] * 0.5 + color * 0.5
        ).astype(np.uint8)

    cv2.imwrite(str(save_path), overlay)


def batch_convert(images_dir, labels_dir, output_dir, vis_dir):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    vis_dir = Path(vis_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
        image_files += list(images_dir.glob(ext))

    print(f"找到 {len(image_files)} 张图片")

    for img_path in image_files:

        base_name = img_path.stem  # ⭐ 文件名获取（简化点）
        label_path = labels_dir / f"{base_name}.txt"

        mask = yolo_seg_to_mask(img_path, label_path)
        if mask is None:
            continue

        # 保存mask
        mask_path = output_dir / f"{base_name}.png"
        cv2.imwrite(str(mask_path), mask)

        # 保存可视化
        image = cv2.imread(str(img_path))
        vis_path = vis_dir / img_path.name
        save_overlay(image, mask, vis_path)

        print(f"已处理: {img_path.name}")

    print("转换完成！")