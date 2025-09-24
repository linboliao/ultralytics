import argparse
import glob
import os
import math
import json
import random
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import geopandas as gpd
import concurrent.futures
from shapely.geometry import Polygon, MultiPolygon

from tasks.wsi import WSIOperator


def tissue_ratio(img, threshold=30):
    """
    计算图像中组织区域的占比
    参数:
        img: PIL Image对象
        threshold: 阈值，用于区分组织与非组织区域
    返回:
        ratio: 组织区域占比
    """
    img_array = np.array(img)
    diff = np.ptp(img_array, axis=2)
    return (diff > threshold).mean()


class X2YOLO:
    """
    X格式到YOLO格式转换器的抽象基类
    """
    pass


class GeoJSON2YOLO(X2YOLO):
    """
    处理WSI图像和GeoJSON标注，生成掩膜并切割图像块的类
    """

    def __init__(self, level: int = 0, patch_size: int = 512):
        """
        初始化处理器

        参数:
            level: 金字塔层级
            patch_size: 切割块大小
        """
        self.level = level
        self.patch_size = patch_size
        self.wsi = None
        self.width = 0
        self.height = 0

        self.class_map = {
            'prostate': 1,
            'Negative': 1,
            'non-cancer': 1,
            'cancer': 2,
            'Positive': 2,
            'Tumor': 2,
            'lymphocyte': 3,
            'vessel': 3,
            'nerve': 3,
            '杂质': 3
        }

    def load_wsi(self, slide_path: str) -> None:
        """
        加载WSI图像

        参数:
            slide_path: WSI图像路径
        """
        print(slide_path)
        self.wsi = WSIOperator(slide_path)
        self.width, self.height = self.wsi.level_dimensions[self.level]

    def parse_geojson(self, geojson_path: str) -> List[Tuple[Polygon, int]]:
        """
        解析GeoJSON文件，提取几何形状和对应的类别ID

        参数:
            geojson_path: GeoJSON文件路径

        返回:
            包含(几何形状, 类别ID)元组的列表
        """
        gdf = gpd.read_file(geojson_path)
        shapes = []
        unknown_class = set()

        for idx, row in gdf.iterrows():

            geom = row.geometry

            fill_value = 2
            if 'classification' in row and row['classification']:
                classification = row['classification']

                if isinstance(classification, str):
                    classification = json.loads(classification)

                if isinstance(classification, dict) and 'name' in classification:
                    class_name = classification['name']

                    if class_name == 'Other':
                        # 需要人工检查 Other 到底是什么
                        unknown_class.add(class_name)
                        continue

                    if class_name in self.class_map:
                        fill_value = self.class_map[class_name]
                    else:
                        fill_value = 4
                        unknown_class.add(class_name)

            if geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    shapes.append((poly, fill_value))
            elif geom.geom_type == 'Polygon':
                shapes.append((geom, fill_value))

        if unknown_class:
            print(f"发现未知分类: {unknown_class}")

        return shapes

    def create_mask(self, shapes: List[Tuple[Polygon, int]]) -> np.ndarray:
        """
        创建掩膜图像

        参数:
            shapes: 包含(几何形状, 填充值)元组的列表

        返回:
            掩膜图像数组
        """
        from rasterio.features import rasterize
        w, h = self.wsi.level_dimensions[0]
        rasterized = rasterize(shapes, out_shape=(h, w), fill=0, all_touched=True)

        if self.level != 0:
            rasterized = cv2.resize(rasterized, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return rasterized

    def save_mask(self, image, mask, mask_path):
        color_map = {
            1: [0, 255, 0],
            2: [255, 0, 0],
            3: [0, 0, 255]
        }
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        for class_id, color in color_map.items():
            colored_mask[mask == class_id] = color
        image_rgb = np.array(image)
        if image_rgb.shape != colored_mask.shape:
            print(f'size 不匹配：image shape{image_rgb.shape}; color mask shape {colored_mask.shape}')
            colored_mask = cv2.resize(colored_mask, (image_rgb.shape[1], image_rgb.shape[0]))
        overlay = cv2.addWeighted(image_rgb, 1, colored_mask, 0.3, 0)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_path, overlay_bgr)

    def save_label(self, mask_patch, label_path):
        binary_masks = {}
        for mask_val, class_id in {1: 0, 2: 1, 3: 2}.items():
            binary_mask = np.uint8(mask_patch == mask_val) * 255
            binary_masks[class_id] = binary_mask
        width, height = mask_patch.shape[:2]
        with open(label_path, 'w') as f:
            for class_id, binary_mask in binary_masks.items():
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    normalized_coords = []
                    for point in contour:
                        x, y = point[0]
                        x_norm = x / width
                        y_norm = y / height
                        normalized_coords.extend([x_norm, y_norm])

                    if len(normalized_coords) >= 6:
                        coords_str = ' '.join([f"{coord:.6f}" for coord in normalized_coords])
                        f.write(f"{class_id} {coords_str}\n")

    def extract_patches(self, mask: np.ndarray, output_dir: Path) -> int:
        """
        提取图像和掩膜块

        参数:
            mask: 掩膜图像数组
            output_dir: 输出目录

        返回:
            提取的块数
        """
        slide_name = Path(self.wsi.filename).stem
        image_output_dir = output_dir / 'images'
        labels_output_dir = output_dir / 'labels'
        masks_output_dir = output_dir / 'masks'

        num_cols = math.ceil(self.width / self.patch_size)
        num_rows = math.ceil(self.height / self.patch_size)

        patch_count = 0

        print(f'开始切割 {slide_name}, 列：{num_cols}, 行：{num_rows}')
        for row in range(num_rows):
            for col in range(num_cols):
                x1 = col * self.patch_size
                y1 = row * self.patch_size
                x2 = min(x1 + self.patch_size, self.width)
                y2 = min(y1 + self.patch_size, self.height)

                img_patch = self.wsi.read_region((x1, y1), self.level, (x2 - x1, y2 - y1))
                img_patch = img_patch.convert('RGB')

                mask_patch = mask[y1:y2, x1:x2]
                raito = tissue_ratio(img_patch)
                keep_prob = max(0.1, raito * 0.5)  # 组织越多保留概率越高
                if np.any(mask_patch > 0):
                    pass
                elif random.random() > keep_prob:
                    # 过滤无标签数据且无组织数据
                    continue

                img_path = os.path.join(image_output_dir, f'{slide_name}_{x1}_{y1}.png')
                img_patch.save(img_path)
                # mask_path = os.path.join(masks_output_dir, f'{slide_name}_{x1}_{y1}.png')
                # self.save_mask(img_patch, mask_patch, mask_path)
                label_path = os.path.join(labels_output_dir, f'{slide_name}_{x1}_{y1}.txt')
                self.save_label(mask_patch, label_path)
                patch_count += 1

        return patch_count

    def process(self, geojson_path: str, slide_path: str, output_dir: Path) -> Dict[str, int]:
        """
        处理完整的流程

        参数:
            geojson_path: GeoJSON文件路径
            slide_path: WSI图像路径
            output_dir: 输出目录

        返回:
            包含处理统计信息的字典
        """
        self.load_wsi(slide_path)
        shapes = self.parse_geojson(geojson_path)
        mask = self.create_mask(shapes)
        patch_count = self.extract_patches(mask, output_dir)

        return {
            "total_shapes": len(shapes),
            "patch_count": patch_count,
            "mask_shape": mask.shape
        }


def find_all_wsi_paths(wsi_root, extentions):
    """
    find the full wsi path under data_root, return a dict {slide_id: full_path}
    """
    # to support more than one ext, e.g., support .svs and .mrxs
    result = {}
    for ext in extentions.split(';'):
        print('Process format:', ext)
        ext = ext[1:]
        all_paths = glob.glob(os.path.join(wsi_root, '**'), recursive=True)
        all_paths = [i for i in all_paths if i.split('.')[-1].lower() == ext.lower()]
        for h in all_paths:
            slide_name = os.path.split(h)[1]
            slide_id = os.path.splitext(slide_name)[0]
            result[slide_id] = h
    print("found {} wsi".format(len(result)))
    return result


def split_dataset(slide_paths, seed=42):
    """随机划分数据集"""
    ratio = [0.8, 0.1, 0.1]
    if not slide_paths:
        return {"train": [], "val": [], "test": []}

    random.seed(seed)
    random.shuffle(slide_paths)
    total = len(slide_paths)
    num_train = int(total * ratio[0])
    num_val = int(total * ratio[1])

    return {
        'train': slide_paths[:num_train],
        'val': slide_paths[num_train:num_train + num_val],
        'test': slide_paths[num_train + num_val:]
    }


parser = argparse.ArgumentParser()
parser.add_argument('--slide_dir', type=str, required=True)
parser.add_argument('--slide_ext', type=str, required=True, help='.要在里面 用;分隔')
parser.add_argument('--geojson_dir', type=str, required=True, help='')
parser.add_argument('--output_dir', type=str, required=True, help='')
parser.add_argument('--patch_size', type=int, default=512)
parser.add_argument('--patch_level', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

if __name__ == "__main__":
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_type in ["train", "val", "test"]:
        (output_dir / dataset_type / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / dataset_type / "labels").mkdir(parents=True, exist_ok=True)
        (output_dir / dataset_type / "masks").mkdir(parents=True, exist_ok=True)

    all_wsi_paths = find_all_wsi_paths(args.slide_dir, args.slide_ext)
    slide_paths, geojson_paths, slide_ids = [], [], []
    for slide_id, wsi_path in all_wsi_paths.items():
        geojson_path = os.path.join(args.geojson_dir, f'{slide_id}.geojson')
        if os.path.exists(geojson_path):
            slide_paths.append(wsi_path)
            geojson_paths.append(geojson_path)
            slide_ids.append(slide_id)

    dataset = split_dataset(slide_ids, args.seed)


    def process_single_slide(slide_id, typ):
        geojson_path = os.path.join(args.geojson_dir, f'{slide_id}.geojson')
        slide_path = all_wsi_paths.get(slide_id)
        processor = GeoJSON2YOLO(level=0, patch_size=512)
        result = processor.process(geojson_path, slide_path, output_dir / typ)
        return f"处理完成: {result}"


    for typ, slide_ids in dataset.items():
        print(f'Processing {typ} slides')
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for slide_id in slide_ids:
                future = executor.submit(process_single_slide, slide_id, typ)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    traceback.print_exc()
