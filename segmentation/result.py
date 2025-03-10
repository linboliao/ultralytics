import json
import os
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import openslide
from PIL import Image
from loguru import logger

from ultralytics import YOLO
from ultralytics.engine.results import Results
import xml.etree.ElementTree as ET

sys.path.insert(0, r'/data2/lbliao/Code/aslide/')
from aslide import Aslide

Image.MAX_IMAGE_PIXELS = None


def is_background(img, threshold=5):
    img_array = np.array(img)
    pixel_max = np.max(img_array, axis=2)
    pixel_min = np.min(img_array, axis=2)
    difference = pixel_max - pixel_min
    return np.sum(difference > threshold) < 800000


class Result:
    def __init__(self, opt):
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, f'slides')
        self.slide_list = opt.slide_list
        self.gpu = opt.gpu
        self.model = YOLO(opt.ckpt)

        self.patch_size = opt.patch_size
        self.infer_size = opt.infer_size

        self.output_dir = opt.output_dir if opt.output_dir else os.path.join(opt.data_root, f'results/')
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)

        self.label_dict = {0: 'prostate', 1: 'cancer', 2: 'burn', 3: 'vessel', 4: 'epithelium', 5: 'ganglion'}
        self.color_dict = {'prostate': [0, 255, 0], 'cancer': [255, 0, 0], 'burn': [0, 0, 255], 'vessel': [255, 255, 0], 'epithelium': [255, 0, 255], 'ganglion': [0, 255, 255]}

    def infer(self, img, gpu):
        raise NotImplementedError()

    def process(self, data):
        # data: img, slide
        raise NotImplementedError()

    @property
    def slides(self):
        if self.slide_list:
            slides = self.slide_list
        else:
            slides = [f for f in os.listdir(self.slide_dir) if os.path.isfile(os.path.join(self.slide_dir, f))]
        return slides

    def run_(self):
        for slide in self.slides:
            self.process(slide)

    def parallel_run(self):
        with ThreadPoolExecutor(max_workers=20) as executor:
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
        results = self.model(img, device=gpu)
        coords = []
        labels = []
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
                coords.append([x1, y1, x2, y2])
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
                for (x1, y1, x2, y2) in coords:
                    x1 = int(x1 + w_s)
                    y1 = int(y1 + h_s)
                    x2 = int(x2 + w_s)
                    y2 = int(y2 + h_s)
                    t_coords.append([[[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]])

                t_labels.extend(labels)
        self.post_process(t_coords, t_labels, base)

    def post_process(self, coords, labels, base):
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


class TiffResults(Result):
    def __init__(self, opt):
        super().__init__(opt)

    def infer(self, img, gpu):
        # img : str or path or PIL.Image or np.ndarray：BGR
        results = self.model(img, device=gpu)
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
        results = self.model(img, device=gpu)
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

            ET.SubElement(annotation, "P", attrib={"X": str(x), "Y": str(y)})

            ET.SubElement(annotation, "S", attrib={"H": str(h), "W": str(w)})

        tree = ET.ElementTree(root)
        output_dir = os.path.join(self.output_dir, f'{base}.dsmeta')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'notes')
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"XML 文件已保存到 {output_path}")


class LMResults(Result):
    def __init__(self, opt):
        super().__init__(opt)

    def process(self, img):
        pass
