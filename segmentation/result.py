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

sys.path.insert(0, r'/data2/lbliao/Code/aslide/')
from aslide import Aslide


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
            for i, box in enumerate(reversed(boxes)):
                [x1, y1, x2, y2] = box.xyxy.tolist()[0]
                label = self.label_dict[int(box.cls.tolist()[0])]
                coords.append([x1, y1, x2, y2])
                labels.append(label)
        return coords, labels

    def process(self, slide):
        base, ext = os.path.splitext(slide)
        slide_path = os.path.join(self.slide_dir, slide)

        wsi = Aslide(slide_path) if ext == '.kfb' else openslide.OpenSlide(slide_path)

        width, height = wsi.level_dimensions[0]
        times = self.patch_size / self.infer_size
        step = int(self.patch_size * wsi.mpp / 20)

        t_coords, t_labels = [], []
        for w_s in range(0, width - step, step):
            for h_s in range(0, height - step, step):
                input_img = wsi.read_region((w_s, h_s), 0, (self.patch_size, self.patch_size))
                if is_background(input_img):
                    continue
                if isinstance(input_img, Image.Image):
                    input_img = input_img.convert('RGB')
                    input_img = input_img.resize((self.infer_size, self.infer_size))
                else:
                    input_img = cv2.resize(input_img, (self.infer_size, self.infer_size), interpolation=cv2.INTER_LINEAR)
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

                coords, labels = self.infer(input_img, self.gpu)
                for (x1, y1, x2, y2) in coords:
                    x1 = int(x1 * times + w_s)
                    y1 = int(y1 * times + h_s)
                    x2 = int(x2 * times + w_s)
                    y2 = int(y2 * times + h_s)
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
                "geometry": {"coordinates": coord},  # 填充坐标
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
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.process, slide) for slide in self.slides]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()


class PicResults(Result):
    def __init__(self, opt):
        super().__init__(opt)

    def process(self, img):
        pass


class LMResults(Result):
    def __init__(self, opt):
        super().__init__(opt)

    def process(self, img):
        pass
