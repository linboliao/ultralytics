import copy
import json
import os
import shutil
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import openslide
import pandas as pd
import torch
import torchvision
from PIL import Image
from loguru import logger
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.engine.results import Results
import xml.etree.ElementTree as ET
import warnings

warnings.filterwarnings("ignore")

import argparse

sys.path.insert(0, r'/data2/lbliao/Code/aslide/')
from aslide import Aslide

Image.MAX_IMAGE_PIXELS = None


def is_background(img, threshold=20):
    img_array = np.array(img)
    diff = np.ptp(img_array, axis=2)  # ptp直接计算max-min
    return (diff > threshold).mean() < 0.15


class Result:
    def __init__(self, opt):
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, f'slides')
        self.slide_list = opt.slide_list
        self.gpu = opt.gpu
        self.models = []
        for ckpt in opt.ckpts.split(';'):
            self.models.append(YOLO(ckpt))

        self.patch_size = opt.patch_size
        self.infer_size = opt.infer_size
        self.csv_path = opt.csv_path

        self.output_dir = opt.output_dir if opt.output_dir else os.path.join(opt.data_root, f'results/')
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)
        self.show_level = opt.show_level

        self.label_dict = {0: 'Benign', 1: 'Malignant', 2: 'vessel', 4: 'epithelium', 3: 'ganglion'}
        self.color_dict = {'Benign': [0, 255, 0], 'Malignant': [255, 0, 0], 'burn': [0, 0, 255], 'vessel': [255, 255, 0], 'epithelium': [255, 0, 255], 'ganglion': [0, 255, 255]}
        self.slide = opt.slide if opt.slide else None

    def infer(self, img, gpu):
        raise NotImplementedError()

    def process(self, data):
        # data: img, slide
        raise NotImplementedError()

    def open_slide(self, slide):
        base, ext = os.path.splitext(slide)
        slide_path = os.path.join(self.slide_dir, slide)
        if ext == '.kfb':
            wsi = Aslide(slide_path)
        elif ext == '.tif':
            wsi = Image.open(slide_path)
            wsi.level_dimensions = [[wsi.size[0], wsi.size[1]]]
            wsi.mpp = 20
        else:
            wsi = openslide.OpenSlide(slide_path)
            wsi.mpp = int(wsi.properties.get('aperio.AppMag', '20'))
        return wsi

    @property
    def slides(self):
        # 1. 初步候选列表
        candidates = (
            self.slide_list if getattr(self, "slide_list", None) else
            [self.slide] if getattr(self, "slide", None) else
            [f for f in os.listdir(self.slide_dir)
             if os.path.isfile(os.path.join(self.slide_dir, f))]
        )

        # 2. 如果没有 CSV 或 CSV 不存在，直接返回 candidates
        if not getattr(self, "csv_path", None) or not os.path.exists(self.csv_path):
            filtered = candidates
        else:
            # 读取 CSV 获取 prediction == 1 的 slide_id
            try:
                df = pd.read_csv(self.csv_path)
                positive_ids = set(
                    df.loc[df["prediction"] == 1, "slide_id"].astype(str).str.strip()
                )
            except Exception as e:
                print(f"读取 {self.csv_path} 失败: {e}")
                filtered = candidates
            else:
                filtered = [
                    f for f in candidates
                    if os.path.splitext(os.path.basename(f))[0] in positive_ids
                ]

        # 3. 排除已经生成过 detect.geojson 的 slide
        final_slides = [
            f for f in filtered
            if not os.path.exists(
                os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(f))[0]}-detect.geojson")
            )
        ]

        return final_slides
    def run_(self):
        for slide in self.slides:
            self.process(slide)

    def parallel_run(self):
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.process, slide) for slide in self.slides]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()


def save_area(area_data, csv_path, key_column='slide_id'):
    """
    如果new_data中的key_column值已存在于CSV文件中，则更新该行；否则追加新行。

    Args:
        area_data (list of dict): 新的数据行，每个字典代表一行。
        csv_path (str): CSV文件的路径。
        key_column (str): 用于判断是否重复的列名，默认为'slide_id'。
    """
    new_df = pd.DataFrame(area_data)

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)

        mask = existing_df[key_column].isin(new_df[key_column])
        existing_df_clean = existing_df[~mask]

        updated_df = pd.concat([existing_df_clean, new_df], ignore_index=True)

        updated_df.to_csv(csv_path, index=False)
        print(f"成功更新CSV文件: {csv_path}。")

    else:
        new_df.to_csv(csv_path, index=False)
        print(f"创建新的CSV文件并写入数据: {csv_path}")


class GeoResults(Result):
    def __init__(self, opt):
        super().__init__(opt)
        self.area = 0

    def infer(self, img, gpu):
        # img : str or path or PIL.Image or np.ndarray：BGR
        results = self.models[0](img, device=gpu, agnostic_nms=True, iou=0.4)
        coords = []
        labels = []
        confs = []
        for result in results:
            boxes = result.boxes
            Malignant_area = 0
            remove_list = []
            for i, box in enumerate(reversed(boxes)):
                [x1, y1, x2, y2] = box.xyxy.tolist()[0]
                label = self.label_dict[int(box.cls.tolist()[0])]
                if label == "Malignant":
                    Malignant_area += (x2 - x1) * (y2 - y1)
                    remove_list.append(i)
                conf = box.conf.tolist()[0]

                coords.append([x1, y1, x2, y2])
                labels.append(label)
                confs.append(conf)
            if Malignant_area < self.infer_size ** 2 * 0.1:
                coords = [coords[i] for i in range(len(coords)) if i not in remove_list]
                labels = [labels[i] for i in range(len(labels)) if i not in remove_list]
                confs = [confs[i] for i in range(len(confs)) if i not in remove_list]
        return coords, labels, confs

    def multi_infer(self, img, gpu):
        # n-1个模型的推理结果
        coords, labels, confs = [], [], []
        with torch.no_grad():
            for model in self.models[:-1]:
                results = model(img, device=gpu, agnostic_nms=True, iou=0.4, half=True, verbose=False)
                for result in results:
                    boxes = result.boxes
                    for i, box in enumerate(reversed(boxes)):
                        [x1, y1, x2, y2] = box.xyxy.tolist()[0]
                        label = self.label_dict[int(box.cls.tolist()[0])]
                        conf = box.conf.tolist()[0]

                        coords.append([x1, y1, x2, y2])
                        labels.append(label)
                        confs.append(conf)

            old_coords = copy.copy(coords)
            old_labels = copy.copy(labels)
            old_confs = copy.copy(confs)
            results = self.models[-1](img, device=gpu, agnostic_nms=True, half=True, verbose=False)

            # n 个模型结果
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(reversed(boxes)):
                    [x1, y1, x2, y2] = box.xyxy.tolist()[0]
                    label = self.label_dict[int(box.cls.tolist()[0])]
                    conf = box.conf.tolist()[0]
                    if conf < 0.3:
                        continue
                    coords.append([x1, y1, x2, y2])
                    labels.append(label)
                    confs.append(conf * 0.1)

            if len(coords) > 0:
                boxes = torch.tensor(coords, dtype=torch.float32)
                scores = torch.tensor(confs, dtype=torch.float32)
                # N个模型结果进行NMS
                i = torchvision.ops.nms(boxes, scores, 0)  # NMS
                index = i.tolist()
                coords = [coords[i] for i in index if 0 <= i < len(coords)]
                labels = [labels[i] for i in index if 0 <= i < len(labels)]
                confs = [confs[i] for i in index if 0 <= i < len(confs)]
                idxs = [i for i, label in enumerate(labels) if label == 'Malignant']
                area = 0
                for idx in idxs:
                    [x1, y1, x2, y2] = coords[idx]
                    area += (x2 - x1) * (y2 - y1)
                # 获取癌症面积小于0.2 或者数量少于4的时的非癌结果
                if area < self.patch_size ** 2 * 0.2 or len(idxs) < 4:
                    coords = [coords[i] for i in range(len(coords)) if i not in idxs]
                    labels = [labels[i] for i in range(len(labels)) if i not in idxs]
                    confs = [confs[i] for i in range(len(confs)) if i not in idxs]

                # N-1 个模型 + 前面的非癌结果
                old_coords.extend(coords)
                old_labels.extend(labels)
                old_confs.extend(confs)
                boxes = torch.tensor(old_coords, dtype=torch.float32)
                scores = torch.tensor(old_confs, dtype=torch.float32)
                # 合并
                i = torchvision.ops.nms(boxes, scores, 0.3)
                index = i.tolist()
                #
                coords = [old_coords[i] for i in index if 0 <= i < len(old_coords)]
                labels = [old_labels[i] for i in index if 0 <= i < len(old_coords)]
                confs = [old_confs[i] for i in index if 0 <= i < len(old_coords)]
                # 挑选有癌的结果
                idxs = [i for i, label in enumerate(labels) if label == 'Malignant']
                area = 0
                for idx in idxs:
                    [x1, y1, x2, y2] = coords[idx]
                    area += (x2 - x1) * (y2 - y1)
                # 当癌去极小或者数量为1 时 去除癌区结果
                if area < self.patch_size ** 2 * 0.03 or len(idxs) == 1:
                    coords = [coords[i] for i in range(len(coords)) if i not in idxs]
                    labels = [labels[i] for i in range(len(labels)) if i not in idxs]
                    confs = [confs[i] for i in range(len(confs)) if i not in idxs]

        return coords, labels, confs

    def process(self, slide):
        base, ext = os.path.splitext(slide)
        slide_path = os.path.join(self.slide_dir, slide)
        if ext == '.kfb':
            wsi = Aslide(slide_path)
            width, height = wsi.level_dimensions[0]
            mpp = wsi.mpp
        elif ext == '.tif':
            wsi = Image.open(slide_path)
            width, height = wsi.size[0], wsi.size[1]
            mpp = 20
        else:
            wsi = openslide.OpenSlide(slide_path)
            width, height = wsi.level_dimensions[0]
            mpp = int(wsi.properties.get('aperio.AppMag', '20'))
        step = int(self.patch_size * (mpp / 20))
        patch_count = 0
        t_coords, t_labels, t_confs = [], [], []
        times = width // wsi.level_dimensions[self.show_level][0]

        for w_s in range(0, width - step, step):
            for h_s in range(0, height - step, step):
                if ext == '.tif':
                    input_img = wsi.crop((w_s, h_s, w_s + step, h_s + step))
                else:
                    input_img = wsi.read_region((w_s, h_s), 0, (step, step))
                if is_background(input_img):
                    continue
                if isinstance(input_img, Image.Image):
                    input_img = input_img.convert('RGB')
                else:
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

                # coords, labels, confs = self.multi_infer(input_img, self.gpu)
                coords, labels, confs = self.infer(input_img, self.gpu)
                for (x1, y1, x2, y2) in coords:
                    x1 = int(x1 + w_s)
                    y1 = int(y1 + h_s)
                    x2 = int(x2 + w_s)
                    y2 = int(y2 + h_s)
                    coord = [[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]
                    coord = [[item // times for item in sublist] for sublist in coord]
                    t_coords.append([coord])
                t_confs.extend(confs)
                t_labels.extend(labels)
                patch_count += 1

        self.post_process(t_coords, t_labels, t_confs, base, patch_count)

    def cal_rate(self, coords, labels):
        Malignant_area = 0
        for coord, label in zip(coords, labels):
            x_list = [x[0] for x in coord[0]]
            y_list = [x[1] for x in coord[0]]
            x1 = min(x_list)
            x2 = max(x_list)
            y1 = min(y_list)
            y2 = max(y_list)
            if label == 'Malignant':
                Malignant_area += (x2 - x1) * (y2 - y1)
        return Malignant_area * 0.7

    def post_process(self, coords, labels, confs, base, patch_count):
        feature_template = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": None},
            "properties": {
                "objectType": "annotation",
                "classification": {"name": None, "color": None}
            }
        }

        features = [
            {
                **feature_template,  # 复制模板
                "id": str(uuid.uuid4()),  # 生成唯一 ID
                "geometry": {"type": "Polygon", "coordinates": coord},  # 填充坐标
                "properties": {
                    "name": f'{conf:.4f}',
                    "classification": {
                        "name": label,
                        "color": self.color_dict[label]  # 填充颜色
                    }
                }
            }
            for coord, label, conf in zip(coords, labels, confs) if label == "Malignant"
        ]

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        output_path = os.path.join(self.output_dir, f"{base}-detect.geojson")
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        logger.info(f'generated {base}.geojson contour json!!!')

        malignant_area = self.cal_rate(coords, labels)
        total_area = patch_count * self.patch_size ** 2
        area_data = [{'slide_id': base, 'area': f'{malignant_area / total_area * 100:.4f}%'}]
        area_path = os.path.join(self.output_dir, f"area.csv")
        save_area(area_data, area_path)


class MultiGeoResults(GeoResults):
    def process(self, slide):

        wsi = self.open_slide(slide)
        width, height = wsi.level_dimensions[0]

        step = int(self.patch_size * (wsi.mpp / 20))
        times = wsi.level_dimensions[0][0] // wsi.level_dimensions[self.show_level][0]

        coordinates = [
            (w, h)
            for w in range(0, width - step, step)
            for h in range(0, height - step, step)
        ]

        total_patches = len(coordinates)

        t_coords, t_labels, t_confs = [], [], []

        def read_region(coord):
            input_img = wsi.read_region(coord, 0, (step, step))
            input_img = input_img.convert("RGB")

            if is_background(input_img):
                return None

            with torch.no_grad():
                coords, labels, confs = self.multi_infer(input_img, self.gpu)

            return coord, coords, labels, confs

        print(f"\nProcessing slide: {slide}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(read_region, c) for c in coordinates]

            # ✅ 进度条在主线程更新
            with tqdm(total=total_patches, desc="Infer Patches", ncols=100) as pbar:
                patch_count = 0
                for future in as_completed(futures):
                    try:
                        result = future.result()

                        if result is not None:
                            patch_count +=1
                            coord, coords, labels, confs = result

                            for (x1, y1, x2, y2) in coords:
                                x1 += coord[0]
                                y1 += coord[1]
                                x2 += coord[0]
                                y2 += coord[1]

                                poly = [[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]
                                poly = [[p // times for p in pt] for pt in poly]

                                t_coords.append([poly])

                            t_labels.extend(labels)
                            t_confs.extend(confs)


                    except Exception:
                        traceback.print_exc()

                    # 每完成一个 patch 更新一次
                    pbar.update(1)

        base = os.path.splitext(slide)[0]
        self.post_process(t_coords, t_labels, t_confs, base, patch_count)


parser = argparse.ArgumentParser(description='YOLO to X')
parser.add_argument('--slide_dir', type=str, default=None)
parser.add_argument('--gpu', type=str, default='0', help='patch directory')
parser.add_argument('--ckpts', type=str, default=None)
parser.add_argument('--patch_size', type=int, default=2048, help='patch size')
parser.add_argument('--infer_size', type=int, default=1536, help='infer size')
parser.add_argument('--csv_path', type=str, default=None, help='csv path')
parser.add_argument('--slide', type=str, default='')
parser.add_argument('--slide_list', type=list, default=[])
parser.add_argument('--output_dir', type=str, default='/NAS145/liaolinbo/Data/MXB/301/yolo2')
parser.add_argument('--show_level', type=int, default=0)
if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    MultiGeoResults(args).parallel_run()
