import os
import json
import re
import uuid
from pathlib import Path

import cv2
from tqdm import tqdm
from loguru import logger

from segmentation.wsi import WSIOperator
from ultralytics import YOLO

import numpy as np
from geojson import Polygon, Feature, FeatureCollection


def yolo_seg_to_geojson(results, offsets):
    """
    将YOLO分割结果转换为GeoJSON格式
    :param results: YOLO推理结果对象
    :param filename_offset_map: 文件名到坐标偏移量的映射
    :return: GeoJSON FeatureCollection
    """
    features = []

    for result in results:
        # 解析文件名中的偏移量
        w_offset, h_offset = offsets
        if not result.masks:
            return None
        for i, mask in enumerate(result.masks.xy):
            # 获取类别信息
            class_id = int(result.boxes.cls[i])
            class_name = result.names[class_id]
            conf = float(result.boxes.conf[i])
            points = np.array(mask, np.int32)
            mask_img = np.zeros((2048, 2048), dtype=np.uint8)  # 创建空白画布
            cv2.fillPoly(mask_img, [points], color=255)  # 填充多边形区域
            # 从 Method 1 的 mask_img 中提取轮廓
            kernel_size = (2, 2)  # 根据连接点宽度调整
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            eroded = cv2.erode(mask_img, kernel, iterations=1)  # 腐蚀断开连接点
            dilated = cv2.dilate(eroded, kernel, iterations=1)  # 恢复主体形状
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 坐标转换：局部坐标 → 全局坐标
            for contour in contours:
                if cv2.contourArea(contour) < 2048*2048 / 20:
                    continue
                coords = contour.squeeze().tolist()
                global_coords = [[x * 4 + w_offset, y * 4 + h_offset] for point in coords for x, y in [point]]

                # 确保多边形闭合
                if global_coords[0] != global_coords[-1]:
                    global_coords.append(global_coords[0])

                # 创建GeoJSON特征
                features.append(Feature(
                    id=str(uuid.uuid4()),
                    geometry=Polygon([global_coords]),
                    properties={
                        "objectType": "annotation",
                        "classification": {
                            "name": class_name,
                            "color": [0, 255, 0]
                        }
                    }
                ))

    return FeatureCollection(features)


def process_svs_to_geojson(slide_path, model, level, patch_size, output_path):
    """
    处理SVS文件并生成GeoJSON
    :param slide_path: SVS文件路径
    :param model: YOLO分割模型
    :param level: 金字塔层级
    :param patch_size: 切片尺寸 (width, height)
    :param output_path: GeoJSON输出路径
    """
    slide = WSIOperator(slide_path)
    slide_id = Path(slide_path).stem
    level_dim = slide.level_dimensions[level]
    downsample = slide.wsi.level_downsamples[level]

    all_features = []

    # 计算网格数量
    grid_x = level_dim[0] // patch_size[0]
    grid_y = level_dim[1] // patch_size[1]

    for y in tqdm(range(0, grid_y * patch_size[1], patch_size[1])):
        for x in range(0, grid_x * patch_size[0], patch_size[0]):
            # 计算基准层坐标
            base_x = int(x * downsample)
            base_y = int(y * downsample)
            w = min(patch_size[0], level_dim[0] - x)
            h = min(patch_size[1], level_dim[1] - y)

            # 读取图像块
            patch = slide.read_region((x, y), level, (w, h), check_background=True)
            if not patch:
                logger.info(f'{slide_id} patch {x} {y} 为背景，跳过！')
                continue
            patch_rgb = patch.convert("RGB")

            # 推理并转换坐标
            results = model(patch_rgb, iou=0.7, conf=0.2)
            # for i, result in enumerate(results):
            #     result.save(filename=f"{slide_id}_{x}_{y}_{i}.jpg")  # save to disk
            features = yolo_seg_to_geojson(results, (base_x, base_y))
            if not features:
                logger.info(f'{slide_id} patch {x} {y} 未识别到目标，跳过！')
                continue
            all_features.extend(features["features"])

    # 保存最终GeoJSON
    with open(output_path, 'w') as f:
        json.dump(FeatureCollection(all_features), f, indent=2)


if __name__ == "__main__":
    # 初始化模型
    model = YOLO("/data2/lbliao/Code/ultralytics/runs/segment/seminal-2048/weights/best.pt")

    # 处理SVS文件
    slide_dir = f'/NAS2/Data1/lbliao/Data/MXB/seminal/slides/'
    for slide in os.listdir(slide_dir):
        slide_id = os.path.splitext(slide)[0]
        label_path = f'/NAS2/Data1/lbliao/Data/MXB/seminal/geojson/{slide_id}.geojson'
        if os.path.exists(label_path):
            continue
        process_svs_to_geojson(
            slide_path=f"/NAS2/Data1/lbliao/Data/MXB/seminal/slides/{slide}",
            model=model,
            level=2,
            patch_size=(2048, 2048),
            output_path=f"{slide_id}.geojson"
        )
