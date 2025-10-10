import argparse
import glob
import os
import math
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString

from tasks.utils.wsi import WSIOperator
from concurrent.futures import ThreadPoolExecutor
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
from loguru import logger


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


def line2polygon(line):
    """
    将LineString转换为闭合的Polygon
    通过将第一个点添加到坐标序列的末尾实现闭合
    """
    if not isinstance(line, LineString):
        raise TypeError("输入必须是LineString类型")

    # 获取LineString的坐标序列
    coords = list(line.coords)

    # 如果LineString已经闭合（首尾点相同），则直接创建Polygon
    if coords[0] == coords[-1]:
        return Polygon(coords)

    # 否则，将第一个点添加到末尾使其闭合
    closed_coords = coords + [coords[0]]

    return Polygon(closed_coords)


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
            'Region*': 2,
            'lymphocyte': 3,
            'vessel': 3,
            'vessle': 3,
            'nerve': 3,
            '杂质': 3
        }

        self.color_map = {
            (0, 255, 0): 1,
            (255, 0, 0): 2,
            (255, 255, 0): 3,
        }

    def load_wsi(self, slide_path: str) -> None:
        """
        加载WSI图像

        参数:
            slide_path: WSI图像路径
        """
        logger.info(slide_path)
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
        unknown_color = set()
        unknown_shape = set()

        for idx, row in gdf.iterrows():

            geom = row.geometry

            fill_value = -1
            if row['classification'] is None:
                fill_value = 2
            elif 'classification' in row and row['classification']:
                cls = row['classification']
                if isinstance(cls, str):
                    cls = json.loads(cls)
                if not isinstance(cls, dict):
                    logger.info('解析 classification 失败')
                    continue
                cls_name = cls.get('name', 'unknown')
                fill_value = self.class_map.get(cls_name, -1)
                if fill_value == -1:
                    unknown_class.add(cls_name)
                    continue
            elif 'color' in row and row['color']:
                color = row['color']
                color = tuple(color) if isinstance(color, list) else color
                fill_value = self.color_map.get(color, -1)
                if fill_value == -1:
                    unknown_color.add(color)
                    continue

            if geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    shapes.append((poly, fill_value))
            elif geom.geom_type == 'Polygon':
                shapes.append((geom, fill_value))
            elif geom.geom_type == 'LineString':
                shape = line2polygon(geom)
                shapes.append((shape, fill_value))
            else:
                unknown_shape.add(geom.geom_type)

        if unknown_class or unknown_color or unknown_shape:
            filename = self.wsi.filename
            logger.warning(f"发现未知分类: {unknown_class}; 发现未知颜色：{unknown_color}; 发现未知geo type： {unknown_shape}； slide {filename}")

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

    def process_patch(self, args):
        """处理单个图像块的函数"""
        row, col, slide_name, mask, output_dir = args

        image_output_dir = output_dir / 'images'
        masks_output_dir = output_dir / 'masks'

        x1 = col * self.patch_size
        y1 = row * self.patch_size
        x2 = min(x1 + self.patch_size, self.width)
        y2 = min(y1 + self.patch_size, self.height)

        img_patch = self.wsi.read_region((x1, y1), self.level, (x2 - x1, y2 - y1))
        img_patch = img_patch.convert('RGB')

        mask_patch = mask[y1:y2, x1:x2]
        ratio = tissue_ratio(img_patch)
        keep_prob = max(0.1, ratio * 0.5)

        if np.any(mask_patch > 0):
            pass
        # elif random.random() > keep_prob:
        else:
            return None

        img_path = os.path.join(image_output_dir, f'{slide_name}_{x1}_{y1}.png')
        img_patch.save(img_path)
        mask_path = os.path.join(masks_output_dir, f'{slide_name}_{x1}_{y1}.png')
        cv2.imwrite(mask_path, mask_patch)

        return 1

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

        num_cols = math.ceil(self.width / self.patch_size)
        num_rows = math.ceil(self.height / self.patch_size)

        logger.info(f'开始切割 {slide_name}, 列：{num_cols}, 行：{num_rows}')
        tasks = []
        for row in range(num_rows):
            for col in range(num_cols):
                tasks.append((row, col, slide_name, mask, output_dir))

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(self.process_patch, tasks))
            patch_count = sum(result for result in results if result is not None)

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
    links_path = f'{wsi_root}/symlink_record.csv'
    if os.path.exists(links_path):
        links = pd.read_csv(links_path)
        all_paths = links['target'].tolist()
    else:
        all_paths = glob.glob(os.path.join(wsi_root, '**'), recursive=True)
    for ext in extentions.split(';'):
        logger.info(f'Process format:{ext}')
        paths = [i for i in all_paths if os.path.splitext(i)[1].lower() == ext.lower()]
        for h in paths:
            slide_name = os.path.split(h)[1]
            slide_id = os.path.splitext(slide_name)[0]
            result[slide_id] = h
    logger.info("found {} wsi".format(len(result)))
    return result


def split_dataset(slide_paths, seed=42):
    """随机划分数据集"""
    ratio = [0.6, 0.2, 0.2]
    if not slide_paths:
        return {"train": [], "val": [], "test": []}

    random.seed(seed)
    random.shuffle(slide_paths)
    total = len(slide_paths)
    num_train = int(total * ratio[0])
    num_val = max(int(total * ratio[1]),1)

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
        processor = GeoJSON2YOLO(level=0, patch_size=args.patch_size)
        result = processor.process(geojson_path, slide_path, output_dir / typ)
        return f"处理完成: {result}"


    for typ, slide_ids in dataset.items():
        logger.info(f'Processing {typ} slides')
        for slide in slide_ids:
            process_single_slide(slide, typ)

        convert_segment_masks_to_yolo_seg(str(output_dir / typ / "masks"), str(output_dir / typ / "labels"), 3)
