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
import torch
import torchvision
from PIL import Image
from loguru import logger

from ultralytics import YOLO
from ultralytics.engine.results import Results
import xml.etree.ElementTree as ET

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
        for ckpt in opt.ckpt:
            self.models.append(YOLO(ckpt))

        self.patch_size = opt.patch_size
        self.infer_size = opt.infer_size

        self.output_dir = opt.output_dir if opt.output_dir else os.path.join(opt.data_root, f'results/')
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)
        self.show_level = opt.show_level

        self.label_dict = {0: 'prostate', 1: 'cancer', 2: 'vessel', 4: 'epithelium', 3: 'ganglion'}
        self.color_dict = {'prostate': [0, 255, 0], 'cancer': [255, 0, 0], 'burn': [0, 0, 255], 'vessel': [255, 255, 0], 'epithelium': [255, 0, 255], 'ganglion': [0, 255, 255]}
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
        if self.slide_list:
            slides = self.slide_list
        elif self.slide:
            slides = [self.slide]
        else:
            slides = [f for f in os.listdir(self.slide_dir) if os.path.isfile(os.path.join(self.slide_dir, f))]
        return slides

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


class GeoResults(Result):
    def __init__(self, opt):
        super().__init__(opt)

    def infer(self, img, gpu):
        # img : str or path or PIL.Image or np.ndarray：BGR
        results = self.models[0](img, device=gpu, agnostic_nms=True, iou=0.4)
        coords = []
        labels = []
        confs = []
        for result in results:
            boxes = result.boxes
            cancer_area = 0
            remove_list = []
            for i, box in enumerate(reversed(boxes)):
                [x1, y1, x2, y2] = box.xyxy.tolist()[0]
                label = self.label_dict[int(box.cls.tolist()[0])]
                if label == "cancer":
                    cancer_area += (x2 - x1) * (y2 - y1)
                    remove_list.append(i)
                conf = box.conf.tolist()[0]

                coords.append([x1, y1, x2, y2])
                labels.append(label)
                confs.append(conf)
            if cancer_area < self.infer_size ** 2 * 0.1:
                coords = [coords[i] for i in range(len(coords)) if i not in remove_list]
                labels = [labels[i] for i in range(len(labels)) if i not in remove_list]
                confs = [confs[i] for i in range(len(confs)) if i not in remove_list]
        return coords, labels, confs

    def multi_infer(self, img, gpu):
        coords, labels, confs = [], [], []
        for model in self.models[:-1]:
            results = model(img, device=gpu, agnostic_nms=True, iou=0.4)
            cancer_area = 0
            remove_list = []
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(reversed(boxes)):
                    [x1, y1, x2, y2] = box.xyxy.tolist()[0]
                    label = self.label_dict[int(box.cls.tolist()[0])]
                    conf = box.conf.tolist()[0]

                    coords.append([x1, y1, x2, y2])
                    labels.append(label)
                    confs.append(conf)
            #         if label == "cancer":
            #             cancer_area += (x2 - x1) * (y2 - y1)
            #             remove_list.append(i)
            #
            # if cancer_area < self.infer_size ** 2 * 0.1:
            #     coords = [coords[i] for i in range(len(coords)) if i not in remove_list]
            #     labels = [labels[i] for i in range(len(labels)) if i not in remove_list]
            #     confs = [confs[i] for i in range(len(confs)) if i not in remove_list]

        # if len(coords) > 0:
        #     boxes = torch.tensor(coords, dtype=torch.float32)
        #     scores = torch.tensor(confs, dtype=torch.float32)
        #
        #     i = torchvision.ops.nms(boxes, scores, 0.5)  # NMS
        #     index = i.tolist()
        #     coords = [coords[i] for i in index if 0 <= i < len(coords)]
        #     labels = [labels[i] for i in index if 0 <= i < len(labels)]
        #     confs = [confs[i] for i in index if 0 <= i < len(confs)]
        old_coords = copy.copy(coords)
        old_labels = copy.copy(labels)
        old_confs = copy.copy(confs)
        results = self.models[-1](img, device=gpu, agnostic_nms=True)
        cancer_area = 0
        remove_list = []
        length = len(coords)
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
        #         if label == "cancer":
        #             cancer_area += (x2 - x1) * (y2 - y1)
        #             remove_list.append(i + length)
        #
        # if cancer_area < self.patch_size ** 2 * 0.2:
        #     coords = [coords[i] for i in range(len(coords)) if i not in remove_list]
        #     labels = [labels[i] for i in range(len(labels)) if i not in remove_list]
        #     confs = [confs[i] for i in range(len(confs)) if i not in remove_list]
        if len(coords) > 0:
            boxes = torch.tensor(coords, dtype=torch.float32)
            scores = torch.tensor(confs, dtype=torch.float32)

            i = torchvision.ops.nms(boxes, scores, 0.5)  # NMS
            index = i.tolist()
            coords = [coords[i] for i in index if 0 <= i < len(coords)]
            labels = [labels[i] for i in index if 0 <= i < len(labels)]
            confs = [confs[i] for i in index if 0 <= i < len(confs)]
            idxs = [i for i, label in enumerate(labels) if label == 'cancer']
            area = 0
            for idx in idxs:
                [x1,y1, x2,y2] = coords[idx]
                area += (x2 - x1) * (y2 - y1)
            if area < self.patch_size ** 2 * 0.2 or len(idxs) < 4:
                coords = [coords[i] for i in range(len(coords)) if i not in idxs]
                labels = [labels[i] for i in range(len(labels)) if i not in idxs]
                confs = [confs[i] for i in range(len(confs)) if i not in idxs]
            old_coords.extend(coords)
            old_labels.extend(labels)
            old_confs.extend(confs)
            boxes = torch.tensor(old_coords, dtype=torch.float32)
            scores = torch.tensor(old_confs, dtype=torch.float32)
            i = torchvision.ops.nms(boxes, scores, 0.3)
            index = i.tolist()
            coords = [old_coords[i] for i in index if 0 <= i < len(old_coords)]
            labels = [old_labels[i] for i in index if 0 <= i < len(old_coords)]
            confs = [old_confs[i] for i in index if 0 <= i < len(old_coords)]
            idxs = [i for i, label in enumerate(labels) if label == 'cancer']
            area = 0
            for idx in idxs:
                [x1,y1, x2,y2] = coords[idx]
                area += (x2 - x1) * (y2 - y1)
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
        self.post_process(t_coords, t_labels, t_confs, base)

    def post_process(self, coords, labels, confs, base):
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
                    "classification": {
                        "name": label,
                        "color": self.color_dict[label]  # 填充颜色
                    }
                }
            }
            for coord, label in zip(coords, labels)
        ]

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        output_path = os.path.join(self.output_dir, f"{base}.geojson")
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        logger.info(f'generated {base}.geojson contour json!!!')

        indexed_confs = list(enumerate(confs))

        # 按值降序排序（确保稳定性）
        sorted_pairs = sorted(indexed_confs, key=lambda x: x[1], reverse=True)

        # 提取前50名的索引
        top50_indices = [idx for idx, val in sorted_pairs[:min(50, len(sorted_pairs))]]
        selected_coords = [coords[i] for i in top50_indices]
        selected_coords = [coord[0][:4] for coord in selected_coords]
        output_path = os.path.join(self.output_dir, f"{base}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(selected_coords, f, indent=4)  # 缩进4空格美化格式


class MultiGeoResults(GeoResults):
    def process(self, slide):
        wsi = self.open_slide(slide)
        width, height = wsi.level_dimensions[0]
        step = int(self.patch_size * (wsi.mpp / 20))
        times = wsi.level_dimensions[0][0] // wsi.level_dimensions[self.show_level][0]
        coordinates = [(w, h) for w in range(0, width - step, step)
                       for h in range(0, height - step, step)]

        t_coords, t_labels, t_confs = [], [], []

        def read_region(coord):
            input_img = wsi.read_region(coord, 0, (step, step))
            if isinstance(input_img, Image.Image):
                input_img = input_img.convert('RGB')
            else:
                input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
            if is_background(input_img):
                return
            coords, labels, confs = self.multi_infer(input_img, self.gpu)
            for (x1, y1, x2, y2) in coords:
                x1 = int(x1 + coord[0])
                y1 = int(y1 + coord[1])
                x2 = int(x2 + coord[0])
                y2 = int(y2 + coord[1])
                coords = [[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]
                coords = [[item // times for item in sublist] for sublist in coords]
                t_coords.append([coords])
            t_confs.extend(confs)
            t_labels.extend(labels)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(read_region, coord) for coord in coordinates]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()
        base, ext = os.path.splitext(slide)
        self.post_process(t_coords, t_labels, t_confs, base)


class TiffResults(Result):
    def __init__(self, opt):
        super().__init__(opt)

    def infer(self, img, gpu):
        # img : str or path or PIL.Image or np.ndarray：BGR
        results = self.models[0](img, device=gpu, agnostic_nms=True, iou=0.4)
        if isinstance(results[0], Results):
            return results[0].plot()
        else:
            return np.zeros([self.patch_size, self.patch_size, 3], dtype=np.uint8)

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
        step = int(self.patch_size * wsi.mpp / 20)
        times = step // self.patch_size
        canvas = np.zeros([height, width, 3], dtype=np.uint8)
        for w_s in range(0, width - step, step):
            for h_s in range(0, height - step, step):
                if ext == '.tif':
                    input_img = wsi.crop((w_s, h_s, w_s + step, h_s + step))
                else:
                    input_img = wsi.read_region((w_s, h_s), 0, (step, step))
                if isinstance(input_img, Image.Image):
                    input_img = input_img.convert('RGB')
                    numpy_array = np.array(input_img)
                    input_img = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
                else:
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
                h_i, w_i = h_s // times, w_s // times
                if is_background(input_img):
                    input_img = cv2.resize(input_img, (self.patch_size, self.patch_size))
                    canvas[h_i:h_i + self.patch_size, w_i:w_i + self.patch_size] = input_img
                    logger.info(f'{slide} {w_s} --- {h_s} is background.')
                    continue

                output_img = self.infer(input_img, self.gpu)
                output_img = cv2.resize(output_img, (self.patch_size, self.patch_size))
                canvas[h_i:h_i + self.patch_size, w_i:w_i + self.patch_size] = output_img
        output_path = os.path.join(self.output_dir, f'{base}.png')
        canvas = Image.fromarray(canvas)
        canvas.thumbnail((width // 2, height // 2))
        canvas.save(output_path)
        logger.info(f'{slide} result saved to {output_path}')


class MdsResults(Result):
    def __init__(self, opt):
        super().__init__(opt)
        self.color_dict = {'prostate': '4278255615', 'cancer': '4294901760', 'burn': [0, 0, 255], 'vessel': [255, 255, 0], 'epithelium': [255, 0, 255], 'ganglion': [0, 255, 255]}

    def infer(self, img, gpu):
        # img : str or path or PIL.Image or np.ndarray：BGR
        results = self.model(img, device=gpu, agnostic_nms=True, iou=0.4)
        coords = []
        labels = []
        for result in results:
            boxes = result.boxes
            cancer_area = 0
            remove_list = []
            for i, box in enumerate(reversed(boxes)):
                [x, y, w, h] = box.xywh.tolist()[0]
                label = self.label_dict[int(box.cls.tolist()[0])]
                if label == "cancer":
                    cancer_area += w * h
                    remove_list.append(i)
                coords.append([x, y, w, h])
                labels.append(label)
            if cancer_area < self.infer_size ** 2 * 0.1:
                coords = [coords[i] for i in range(len(coords)) if i not in remove_list]
                labels = [labels[i] for i in range(len(labels)) if i not in remove_list]
        return coords, labels

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

        t_coords, t_labels = [], []
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

                coords, labels = self.infer(input_img, self.gpu)
                for (x, y, w, h) in coords:
                    x = int(x + w_s)
                    y = int(y + h_s)
                    w = int(w)
                    h = int(h)
                    t_coords.append([x, y, w, h])

                t_labels.extend(labels)

        self.post_process(t_coords, t_labels, base)

    def post_process(self, coords, labels, base):
        """
        将 (x, y, w, h) 列表保存为 XML 文件。

        参数:
            annotation_list: 包含 (x, y, w, h) 的列表。
            output_file: 输出 XML 文件的路径。
        """
        root = ET.Element("Annotations", attrib={"Unit": "", "Scale": "1"})

        for idx, ([x, y, w, h], label) in enumerate(zip(coords, labels)):
            annotation = ET.SubElement(root, "Annotation", attrib={
                "Visible": "-1",
                "Measurement": "1",
                "FontUnderline": "0",
                "Type": "Point2",
                "Width": "1",
                "Selected": "0",
                "FontFamily": "Arial",
                "Subtype": "2",
                "GUID": f"{uuid.uuid4()}",  # 可以生成唯一的 GUID 或使用其他方式
                "DetailVisible": "0",
                "FontItalic": "0",
                "FontBold": "0",
                "Color": self.color_dict[label],
                "FontSize": "12"
            })

            ET.SubElement(annotation, "Metadata", attrib={
                "Length": "-1",
                "Angle": "-1",
                "Name": f"标注 {idx + 1}",
                "Scale": "1",
                "Radius": "-1",
                "Detail": f"描述 {idx + 1}",
                "Area": f"{w * h}",
                "ArcLength": "-1",
                "Path": ""
            })

            ET.SubElement(annotation, "P", attrib={"X": str(x - w / 2), "Y": str(y - h / 2)})

            ET.SubElement(annotation, "S", attrib={"H": str(h), "W": str(w)})

        tree = ET.ElementTree(root)
        output_dir = os.path.join(self.output_dir, f'{base}.dsmeta')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'notes')
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"XML 文件已保存到 {output_path}")


class KVResults(MdsResults):
    def post_process(self, coords, labels, base):
        annotation = []
        for idx, ([x, y, w, h], label) in enumerate(zip(coords, labels)):
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
                "color": self.color_dict[label],
                "measurement": False,
                "radius": 0,
                "arcLength": 0,
                "angle": 0,
                "region": {
                    "x": x - w / 2,
                    "y": y - h / 2,
                    "width": w,
                    "height": h
                }
            })
        base = base.replace('.', '_')
        path = os.path.join(self.output_dir, f"{base}_kfb/Annotations/")
        os.makedirs(path, exist_ok=True)
        output_path = os.path.join(path, f"1.json")
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        logger.info(f'generated {base}.json contour json!!!')


class LMResults(Result):
    def __init__(self, opt):
        super().__init__(opt)

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

        os.makedirs(self.output_dir, exist_ok=True)
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
                    numpy_array = np.array(input_img)
                    input_img = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
                else:
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
                base, ext = os.path.splitext(slide)
                cv2.imwrite(os.path.join(self.output_dir, f'{base}_{w_s}_{h_s}.jpg'), input_img)
                results = self.model(input_img, device=self.gpu)
                shapes = []
                for result in results:
                    boxes = result.boxes  # Boxes object for bounding box outputs
                    for i, box in enumerate(reversed(boxes)):
                        [x1, y1, x2, y2] = box.xyxy.tolist()[0]
                        points = [[x1, y1], [x2, y2]]
                        shapes.append({
                            "label": self.label_dict[int(box.cls.tolist()[0])],
                            "points": points,
                            "group_id": None,
                            "description": "",
                            "shape_type": "rectangle",
                            "flags": {},
                            "mask": None
                        })
                ann = {
                    "version": "5.6.0",
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": f"{base}_{w_s}_{h_s}.jpg",
                    "imageData": None,
                    "imageHeight": step,
                    "imageWidth": step,
                }
                with open(os.path.join(self.output_dir, f'{base}_{w_s}_{h_s}.json'), 'w') as f:
                    json.dump(ann, f, indent=2)
                logger.info(f'process {base}_{w_s}_{h_s}.jpg')


class PicResults(Result):
    def __init__(self, opt):
        super().__init__(opt)
        self.label_dir = os.path.join(opt.data_root, f'lms-cell/')
        self.image_dir = os.path.join(opt.data_root, f'images/')
        os.makedirs(self.label_dir, exist_ok=True)

    @property
    def slides(self):
        return os.listdir(self.image_dir)

    def process(self, img):
        img_path = os.path.join(self.image_dir, img)
        input_img = cv2.imread(img_path)
        results = self.models[0](input_img, device=self.gpu)
        shapes = []
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            for i, box in enumerate(reversed(boxes)):
                [x1, y1, x2, y2] = box.xyxy.tolist()[0]
                points = [[x1, y1], [x2, y2]]
                label = self.label_dict[int(box.cls.tolist()[0])]
                if label != 'vessel':
                    continue
                shapes.append({
                    "label": label,
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None
                })
        ann = {
            "version": "5.6.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": img,
            "imageData": None,
            "imageHeight": self.patch_size,
            "imageWidth": self.patch_size,
        }
        base, _ = os.path.splitext(img)
        with open(os.path.join(self.label_dir, f'{base}.json'), 'w', encoding='utf-8') as f:
            json.dump(ann, f, indent=2)
        logger.info(f'process {base}.jpg')
