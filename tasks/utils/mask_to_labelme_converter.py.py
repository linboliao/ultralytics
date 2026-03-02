import cv2
import json
import os
from glob import glob
import numpy as np


def convert_masks_to_labelme(
    mask_folder,
    image_folder,
    output_folder,
    class_map=None,
    min_area=20
):
    """
    将语义分割 mask 批量转换为 LabelMe polygon 格式

    Args:
        mask_folder: mask 文件夹
        image_folder: 原图文件夹
        output_folder: 输出 json 文件夹
        class_map: {1: "class_name"} 映射
        min_area: 过滤小轮廓
    """

    os.makedirs(output_folder, exist_ok=True)

    mask_files = sorted(
        glob(os.path.join(mask_folder, "*.png")) +
        glob(os.path.join(mask_folder, "*.jpg"))
    )

    print(f"共找到 {len(mask_files)} 个 mask 文件")

    for mask_path in mask_files:
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"读取失败: {mask_path}")
                continue

            h, w = mask.shape
            base_name = os.path.splitext(os.path.basename(mask_path))[0]

            # 尝试匹配原图
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(image_folder, base_name + ext)
                if os.path.exists(candidate):
                    img_path = os.path.basename(candidate)
                    break

            if img_path is None:
                img_path = os.path.basename(mask_path)  # fallback

            shapes = []
            class_ids = sorted([c for c in np.unique(mask) if c != 0])

            for cid in class_ids:
                label = class_map.get(cid, f"class_{cid}") if class_map else f"class_{cid}"

                binary = (mask == cid).astype(np.uint8) * 255

                contours, _ = cv2.findContours(
                    binary,
                    cv2.RETR_TREE,   # 保留层级
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_area:
                        continue

                    epsilon = 0.003 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    if len(approx) < 3:
                        continue

                    points = approx.reshape(-1, 2).astype(float).tolist()

                    shapes.append({
                        "label": label,
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })

            annotation = {
                "version": "4.5.9",
                "flags": {},
                "shapes": shapes,
                "imagePath": img_path,
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w
            }

            out_json = os.path.join(output_folder, base_name + ".json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=2)

            print(f"已转换: {base_name}")

        except Exception as e:
            print(f"处理失败 {mask_path}: {e}")


if __name__ == "__main__":

    CLASS_MAP = {
        1: "benign",
        2: "grade_3",
        3: "grade_4",
        4: "grade_5",
    }

    phase = "test"

    convert_masks_to_labelme(
        mask_folder=f"/NAS145/liaolinbo/Data/GlandSeg/gleason/dataset/{phase}/masks",
        image_folder=f"/NAS145/liaolinbo/Data/GlandSeg/gleason/dataset/{phase}/images",
        output_folder=f"/NAS145/liaolinbo/Data/GlandSeg/gleason/dataset/{phase}/labelme",
        class_map=CLASS_MAP,
        min_area=30
    )