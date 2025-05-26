import gc
import json
import os
import sys
import uuid
from pathlib import Path
from typing import List

import numpy as np
import openslide
import torch
import torchvision
from PIL import Image
from loguru import logger

from ultralytics import YOLO

sys.path.insert(0, r'/data2/lbliao/Code/aslide/')
from aslide import Aslide

sys.path.insert(1, r'/data2/lbliao/Code/opensdpc/')
from opensdpc.opensdpc import OpenSdpc

Image.MAX_IMAGE_PIXELS = None


def is_background(img, threshold=20):
    img_array = np.array(img)
    diff = np.ptp(img_array, axis=2)  # ptp直接计算max-min
    return (diff > threshold).mean() < 0.15


class WSIOperator:
    @staticmethod
    def open_slide(path: Path):
        """统一WSI打开接口"""
        suffix = path.suffix.lower()
        if suffix == '.kfb':
            slide = Aslide(str(path))
        elif suffix == '.sdpc':
            slide = OpenSdpc(str(path))
        else:
            slide = openslide.OpenSlide(str(path))
        return slide

    @staticmethod
    def read_region(slide, location, level, size):
        """统一区域读取接口"""
        return slide.read_region(location, level, size)


class BaseProcessor:
    def __init__(self, config):
        self.config = config
        self.models = self._init_models()
        self._setup_paths()

    def _init_models(self):
        """模型工厂方法"""
        return [YOLO(ckpt) for ckpt in self.config.ckpt]

    def _setup_paths(self):
        """路径配置统一管理"""
        self.slide_dir = Path(self.config.slide_dir)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _infer(self):
        raise NotImplementedError

    def _multi_infer(self):
        raise NotImplementedError

    # 抽象方法由子类实现
    def process_slide(self, slide_path):
        raise NotImplementedError

    def _process_tile(self, tile_img):
        raise NotImplementedError

    def _save_results(self, results, slide_id: str):
        raise NotImplementedError

    def _generate_tiles(self, wsi) -> list:
        """生成WSI多分辨率切片坐标（参考网页1、网页6的tile生成策略）"""
        base_level = 0  # 最高分辨率层级
        tile_size = self.config.tile_size  # 默认512x512
        overlap = self.config.overlap  # 默认64像素重叠

        # 获取基础层级的全图尺寸
        base_width, base_height = wsi.level_dimensions[base_level]

        # 计算有效步长（考虑重叠）
        step = tile_size - overlap

        # 生成坐标网格
        coords = []
        for y in range(0, base_height - overlap, step):
            for x in range(0, base_width - overlap, step):
                # 添加坐标及对应层级信息
                coords.append({
                    "x": x,
                    "y": y,
                    "level": base_level,
                    "size": tile_size,
                    "overlap": overlap
                })

        return coords

    def _parallel_process(self, wsi, tile_coords) -> list:
        """并行处理切片"""
        from concurrent.futures import ThreadPoolExecutor
        import threading

        # 内存控制参数
        BATCH_SIZE = 100
        MAX_WORKERS = min(8, os.cpu_count() * 2)  # 经验值

        # 线程安全的数据结构
        results = []
        lock = threading.Lock()

        # 带内存管理的处理函数
        def process_single_tile(tile_info):
            try:
                # 读取图像区域
                tile_img = WSIOperator.read_region(
                    wsi,
                    (tile_info["x"], tile_info["y"]),
                    tile_info["level"],
                    (tile_info["size"], tile_info["size"])
                )

                # 背景过滤
                if is_background(tile_img):
                    return None

                tile_img = tile_img.convert('RGB')

                # 执行推理
                with torch.no_grad():
                    detections = self._process_tile(tile_img)

                # 坐标映射
                scale_factor = tile_info["level"] + 1
                scaled_detections = []
                for det in detections:
                    x1 = det["bbox"][0] * scale_factor + tile_info["x"]
                    y1 = det["bbox"][1] * scale_factor + tile_info["y"]
                    x2 = det["bbox"][2] * scale_factor + tile_info["x"]
                    y2 = det["bbox"][3] * scale_factor + tile_info["y"]
                    scaled_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "label": det["label"],
                        "conf": det["conf"]
                    })
                return scaled_detections
            except Exception as e:
                logger.error(f"处理切片失败: {str(e)}")
                return None

        # 分批处理机制
        for i in range(0, len(tile_coords), BATCH_SIZE):
            batch = tile_coords[i:i + BATCH_SIZE]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(process_single_tile, t) for t in batch]

                for future in futures:
                    result = future.result()
                    if result:
                        with lock:
                            results.extend(result)

            # 主动释放内存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            gc.collect()

        return results


class GeoJSONProcessor(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)
        # 医学检测类别映射（示例：前列腺癌病理检测）
        self.label_map = {
            0: "prostate",
            1: "cancer",
            2: "vessel",
            3: "ganglion",
            4: "epithelium"
        }
        self.color_schema = {
            "prostate": "#00FF00",  # 绿色
            "cancer": "#FF0000",  # 红色
            "vessel": "#FFFF00",  # 黄色
            "ganglion": "#00FFFF",  # 青色
            "epithelium": "#FF00FF"  # 品红
        }
        self.infer_params = {  # 模型推理参数
            'agnostic_nms': True,
            'iou': 0.4,
            'conf': 0.3
        }

    def process_slide(self, slide_path: Path):
        """处理整张病理切片的主流程"""
        wsi = WSIOperator.open_slide(slide_path)
        tile_coords = self._generate_tiles(wsi)
        results = self._parallel_process(wsi, tile_coords)
        self._save_results(results, slide_path.stem)

    def _process_tile(self, tile_img: np.ndarray) -> list:
        """处理单个图像块的多模型协同推理"""
        merged_results = []

        # 第一阶段：辅助模型检测
        for model in self.models[:-1]:
            results = model(tile_img, **self.infer_params)
            merged_results.extend(self._filter_results(results))

        # 第二阶段：主模型精细检测
        final_results = self.models[-1](tile_img, **self.infer_params)
        return self._merge_results(merged_results, final_results)

    def _filter_results(self, results) -> list:
        """过滤并标准化检测结果"""
        filtered = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                label = self.label_map.get(labels[i], "unknown")
                if confs[i] < self.infer_params['conf']:
                    continue
                filtered.append({
                    "bbox": boxes[i].tolist(),
                    "label": label,
                    "conf": float(confs[i])
                })
        return filtered

    @staticmethod
    def _merge_results(initial, final) -> list:
        """多模型结果融合策略"""
        # 实现NMS融合逻辑
        all_boxes = torch.cat([initial, final], dim=0)
        # 执行NMS过滤
        keep_idx = torchvision.ops.nms(
            boxes=all_boxes[:, :4],
            scores=all_boxes[:, 4],
            iou_threshold=0.1
        )
        return all_boxes[keep_idx].tolist()

    def _save_results(self, results, slide_id: str):
        """生成GeoJSON格式的标注文件"""
        features = []
        for idx, detection in enumerate(results):
            feature = {
                "type": "Feature",
                "id": f"{slide_id}_{idx}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": self._bbox_to_polygon(detection['bbox'])
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": detection['label'],
                        "color": self.color_schema[detection['label']],
                        "confidence": detection['conf']
                    }
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        output_path = self.output_dir / f"{slide_id}.geojson"
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

    @staticmethod
    def _bbox_to_polygon(bbox: list) -> list:
        """将边界框转换为GeoJSON多边形坐标"""
        x1, y1, x2, y2 = bbox
        return [[[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]]


class KVProcessor(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)
        # 医学检测类别颜色映射
        self.color_schema = {
            "prostate": "#00FF00",  # 绿色
            "cancer": "#FF0000",  # 红色
            "vessel": "#FFFF00",  # 黄色
            "ganglion": "#00FFFF"  # 青色
        }
        self.annotation_template = {  # 标注模板（参考网页5的JSON结构）
            "points": [],
            "imageId": 0,
            "guid": "",
            "name": "",
            "imageindex": "1",
            "region": {"x": 0, "y": 0, "width": 0, "height": 0}
        }

    def _save_results(self, results: List[dict], slide_id: str):
        """重写保存方法（保持与GeoJSONProcessor相同签名）"""
        annotation = []

        for idx, detection in enumerate(results):
            annotation.append({
                "points": [],
                "imageId": 0,
                "guid": f"{uuid.uuid4()}",
                "name": f"矩形{idx}",
                "imageindex": "1",
                "isAllShow": False,
                "isAlwaysShowDesc": True,
                "description": "",
                "scale": 0.0388786665223509,
                "width": "2",
                "type": "Rectangle",
                "fontUnderLine": False,
                "fontSize": 11,
                "fontFamily": "Microsoft Sans Serif",
                "fontItalic": False,
                "fontBold": False,
                "visible": True,
                "color": self.color_schema[detection['label']],
                "measurement": False,
                "radius": 0,
                "arcLength": 0,
                "angle": 0,
                "region": self._bbox_to_region(detection)
            })
        slide_id = slide_id.replace('.', '_')
        path = os.path.join(self.output_dir, f"{slide_id}_kfb/Annotations/")
        os.makedirs(path, exist_ok=True)
        output_path = os.path.join(path, f"1.json")
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        logger.info(f'generated {slide_id}.json contour json!!!')

    @staticmethod
    def _bbox_to_region(detection: dict) -> dict:
        """坐标转换（与GeoJSON共享逻辑）"""
        x1, y1, x2, y2 = detection['bbox']
        return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
