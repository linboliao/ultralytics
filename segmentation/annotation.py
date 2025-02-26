import argparse
import json
import math
import os
import shutil
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import cv2
import h5py
import numpy as np
import openslide
from PIL import Image
from loguru import logger

from ultralytics import YOLO

sys.path.insert(0, r'/data2/lbliao/Code/aslide/')
from aslide import Aslide

MIN_AREA = 3000


def is_background(img, threshold=5):
    img_array = np.array(img)
    pixel_max = np.max(img_array, axis=2)
    pixel_min = np.min(img_array, axis=2)
    difference = pixel_max - pixel_min
    tissue_percent = np.sum(difference > threshold) / (img_array.shape[0] * img_array.shape[1])

    return tissue_percent < 0.1


class Annotation:
    def __init__(self, opt):
        self.patch_dir = opt.patch_dir if opt.patch_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/image')
        self.output_dir = opt.output_dir if opt.output_dir else os.path.join(opt.data_root, f'dataset/{opt.patch_size}/')

        self.patch_size = opt.patch_size
        self.output_size = opt.output_size
        self.skip_done = opt.skip_done
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)

    def filter_contours(self, cnt_info):
        contours, hierarchy = cnt_info
        filtered_contours = []
        for cnt, h in zip(contours, hierarchy[0]):
            area = cv2.contourArea(cnt)
            patch_area = (self.patch_size - 3) ** 2
            parent_area = cv2.contourArea(contours[h[3]]) if h[3] != -1 else float('inf')

            # 存在父contour 且 父contour不为整张图的  且 父contour面积远大于子contour 且 子contour面积很小
            if patch_area // 500 < area < patch_area and not (h[3] != -1 and parent_area < patch_area and area < parent_area // 3):
                start = 0
                while start < len(cnt):
                    start_point = cnt[start][0]
                    for i, end_point in enumerate(cnt[start + 5: start + 15]):
                        end_point = end_point[0]
                        if math.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2) < 8:
                            cnt = np.vstack((cnt[:start], cnt[start + i + 5:]))
                            break
                    start += 1
                cnt += np.array([-3, -3])
                if len(cnt) > 3:
                    filtered_contours.append(cnt)
        return filtered_contours

    def get_contours(self, patch: str):
        lower_bound = np.array([20, 20, 30])
        upper_bound = np.array([180, 180, 200])

        image_path = os.path.join(self.patch_dir, patch)
        image = cv2.imread(image_path)
        image = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        mask = cv2.inRange(image, lower_bound, upper_bound)
        dark_region = cv2.bitwise_not(mask)

        cnt_info = cv2.findContours(dark_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cnt_info

    def show_contours(self, patch, contours):
        ihc_path = os.path.join(self.patch_dir, patch)
        ihc_image = cv2.imread(ihc_path)
        cv2.drawContours(ihc_image, contours, -1, (0, 0, 255), 1)
        ihc_image = Image.fromarray(cv2.cvtColor(ihc_image, cv2.COLOR_BGR2RGB))
        ihc_contour = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/contours/ihc'
        os.makedirs(ihc_contour, exist_ok=True)
        ihc_save_path = os.path.join(ihc_contour, patch)
        ihc_image.save(ihc_save_path)

        he_path = ihc_path.replace('ihc', 'he')
        he_image = cv2.imread(he_path)
        cv2.drawContours(he_image, contours, -1, (0, 0, 255), 1)
        he_image = Image.fromarray(cv2.cvtColor(he_image, cv2.COLOR_BGR2RGB))
        he_contour = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/contours/he'
        os.makedirs(he_contour, exist_ok=True)
        he_save_path = os.path.join(he_contour, patch)
        he_image.save(he_save_path)

    def contour2txt(self, contours, patch: str, clazz: int = 0):
        base, _ = os.path.splitext(patch)
        with open(os.path.join(self.output_dir, f'{base}.txt'), 'w') as f:
            for cnt in contours:
                coords = np.squeeze(cnt.reshape(-1, 1))
                coords = coords / self.patch_size
                coords_str = ' '.join(map(str, coords))

                line = f'{clazz} {coords_str}'
                f.write(line + '\n')
        logger.info(f'{base}.txt generated')

    def run(self, patch: str):
        cnt_info = self.get_contours(patch)
        contours = self.filter_contours(cnt_info)
        if contours:
            self.show_contours(patch, contours)
            self.contour2txt(contours, patch)

    def parallel_run(self):
        images = os.listdir(self.patch_dir)
        # img_paths = [os.path.join(self.patch_dir, img) for img in images]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.run, img) for img in images]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()


class GeoAnnotation(Annotation):
    def __init__(self, opt):
        super().__init__(opt)
        self.coord_dir = opt.coord_dir if opt.coord_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/coord')
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, f'slides')
        self.geo_ann_dir = opt.geo_ann_dir if opt.geo_ann_dir else os.path.join(opt.data_root, f'annotation')
        self.slide_list = opt.slide_list

        self.label_dir = os.path.join(self.output_dir, f'labels/')
        self.image_dir = os.path.join(self.output_dir, f'images/')
        self.contour_dir = os.path.join(self.output_dir, f'contours/')
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.contour_dir, exist_ok=True)

    @property
    def slides(self):
        if self.slide_list:
            slides = self.slide_list
        else:
            slides = [f for f in os.listdir(self.slide_dir) if os.path.isfile(os.path.join(self.slide_dir, f))]
        anns = [os.path.splitext(p)[0] for p in os.listdir(self.geo_ann_dir)]
        slides = [slide for slide in slides if os.path.splitext(slide)[0] in anns]
        return slides

    def show_contours(self, patch, contours):
        image_path = os.path.join(self.contour_dir, patch)
        contours = [np.array(cnt, dtype=np.int32).reshape(-1, 1, 2) for cnt in contours]
        image = cv2.imread(image_path)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        save_path = os.path.join(self.contour_dir, patch)
        image.save(save_path)

    def get_contours(self, slide):
        wsi_path = os.path.join(self.slide_dir, slide)
        wsi = Aslide(wsi_path) if '.kfb' in slide else openslide.OpenSlide(wsi_path)
        base, ext = os.path.splitext(slide)
        file = h5py.File(os.path.join(self.coord_dir, f'{base}.h5'), mode='r')
        he_points = list(file['coords'][:])
        logger.info(f'start to process {slide}')
        for (w, h) in he_points:
            ann_path = os.path.join(self.geo_ann_dir, f'{base}.geojson')
            if self.skip_done and os.path.isfile(os.path.join(self.image_dir, f'{base}_{w}_{h}.png')):
                continue
            with open(ann_path, 'r', encoding='utf-8') as file:
                anns = json.load(file)
            features = anns.get('features')
            patch_coords = []
            label_path = os.path.join(self.label_dir, f'{base}_{w}_{h}.txt')

            def contour(data):
                lc_coords = []
                if any(w < a < w + self.patch_size and h < b < h + self.patch_size for (a, b) in data) and len(data) > 10:
                    data = [[a - w, b - h] for [a, b] in data]
                    patch_coords.append(data)
                    data = np.array(data)
                    contours = np.squeeze(data.reshape(-1, 1))
                    contours = contours / self.patch_size
                    flag = True
                    for i in range(0, len(contours), 2):
                        if 0 < float(contours[i]) < 1 and 0 < float(contours[i + 1]) < 1:
                            lc_coords.append(float(contours[i]))
                            lc_coords.append(float(contours[i + 1]))
                        elif flag:
                            lc_coords.append(min(max(0, float(contours[i])), 1))
                            lc_coords.append(min(max(0, float(contours[i + 1])), 1))
                            flag = False
                    if len(lc_coords) >= 6 and len(lc_coords) % 2 == 0:
                        contours_str = ' '.join(map(str, lc_coords))
                        name = feature.get('properties', {}).get('classification', {}).get('name', '')
                        color = feature.get('properties', {}).get('classification', {}).get('color', [])
                        if name == 'non-cancer' or name == 'Negative':
                            clazz = 0
                        elif name == 'Other':
                            return
                        elif color == [0, 255, 0]:
                            clazz = 0
                        else:
                            clazz = 1
                        line = f'{clazz} {contours_str}'
                        f.write(line + '\n')

            with open(label_path, 'w') as f:
                for feature in features:
                    coordinates = feature['geometry']['coordinates']
                    if feature['geometry']['type'] == 'Polygon':
                        for coords in coordinates:
                            if isinstance(coords, list):
                                contour(coords)
                    elif feature['geometry']['type'] == 'MultiPolygon':
                        for coords in coordinates:
                            for sub_coords in coords:
                                if isinstance(sub_coords, list):
                                    contour(sub_coords)
            patch = wsi.read_region((w, h), 0, (self.patch_size, self.patch_size))
            if isinstance(patch, np.ndarray):
                patch = Image.fromarray(patch)
            patch.save(os.path.join(self.contour_dir, f'{base}_{w}_{h}.png'))
            self.show_contours(f'{base}_{w}_{h}.png', patch_coords)
            patch = patch.resize((self.output_size, self.output_size))
            image_path = os.path.join(self.image_dir, f'{base}_{w}_{h}.png')
            patch.save(image_path, quality=95)

            logger.info(f'{base}_{w}_{h}.png Annotation generated')

    def run(self, slide: str):
        self.get_contours(slide)

    def parallel_run(self):
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.run, slide) for slide in self.slides]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()


class LMAnnotation(Annotation):
    def __init__(self, opt):
        super().__init__(opt)
        self.lm_ann_dir = '/NAS2/Data1/lbliao/Data/MXB/Seg-Relabel/labelme/250224'
        self.label_dir = os.path.join(self.output_dir, f'labels/')
        self.image_dir = os.path.join(self.output_dir, f'images/')
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def get_contours(self, patch: str):
        base, ext = os.path.splitext(patch)
        ann_path = os.path.join(self.lm_ann_dir, f'{base}.json')
        with open(ann_path, 'r', encoding='utf-8') as file:
            anns = json.load(file)
        shapes = anns.get('shapes')
        label_path = os.path.join(self.label_dir, f'{base}.txt')
        with open(label_path, 'w') as f:
            for shape in shapes:
                if shape.get('label') == 'prostate':
                    clazz = 0
                elif shape.get('label') == 'cancer':
                    clazz = 1
                elif shape.get('label') == '电切烧灼腺体':
                    clazz = 2
                elif shape.get('label') == '血管':
                    clazz = 3
                elif shape.get('label') == '鳞状上皮':
                    clazz = 4
                elif shape.get('label') == '神经节':
                    clazz = 5
                if shape.get('shape_type') == 'polygon':
                    points = shape.get('points')
                    points = [item / self.patch_size for sublist in points for item in sublist]
                elif shape.get('shape_type') == 'rectangle':
                    points = shape.get('points')
                    [x1, y1], [x2, y2] = points[0], points[1]
                    points = [x1 / self.patch_size, y1 / self.patch_size, x1 / self.patch_size, y2 / self.patch_size, x2 / self.patch_size, y2 / self.patch_size, x2 / self.patch_size, y1 / self.patch_size]
                contours_str = ' '.join(map(str, points))
                line = f'{clazz} {contours_str}'
                f.write(line + '\n')
        shutil.copy(os.path.join(self.lm_ann_dir, patch), os.path.join(self.image_dir, patch))

    def run(self, slide: str):
        self.get_contours(slide)

    def parallel_run(self):
        images = os.listdir(self.lm_ann_dir)
        images = [img for img in images if img.endswith('.jpg')]
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.run, img) for img in images]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()


class YOLOAnnotation(Annotation):
    def __init__(self, opt):
        super().__init__(opt)
        self.patch_level = opt.patch_level
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, f'test/0224')
        self.slide_list = opt.slide_list
        self.gpu_ids = opt.gpu_ids

        self.model = YOLO(opt.ckpt)
        self.label_dict = {0: 'prostate', 1: 'cancer', 2: 'burn', 3: 'vessel', 4: 'epithelium', 5: 'ganglion'}
        self.color_dict = {
            'prostate': [0, 255, 0],  # 红色
            'cancer': [255, 0, 0],  # 绿色
            'burn': [0, 0, 255],  # 蓝色
            'vessel': [255, 255, 0],  # 黄色
            'epithelium': [255, 0, 255],  # 紫色
            'ganglion': [0, 255, 255]  # 青色
        }

    def inference(self, img):
        results = self.model(img, device=self.gpu_ids)
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

    def qupath_feature(self, coords, labels, base):
        features = []
        for coord, label in zip(coords, labels):
            feature = {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coord,
                }
            }

            feature.update({"properties": {
                "objectType": "annotation",
                "classification": {"name": label, "color": self.color_dict[label]}
            }})
            features.append(feature)
        geojson = {"type": "FeatureCollection", "features": features}
        with open(os.path.join(self.output_dir, f'{base}.geojson'), 'w') as f:
            json.dump(geojson, f, indent=2)
            logger.info(f'generated {base}.geojson contour json!!!')

    def patch_process(self, slide):
        patch_level = 0

        base, ext = os.path.splitext(slide)
        slide_path = os.path.join(self.slide_dir, slide)
        wsi = Aslide(slide_path) if ext == '.kfb' else openslide.open_slide(slide_path)
        if wsi.mpp == 40:
            patch_level = 1
        elif wsi.mpp == 80:
            patch_level = 2
        [w, h] = wsi.level_dimensions[patch_level]

        step = int(self.patch_size)
        t_coords, t_labels = [], []
        for w_s in range(0, w - step, step):
            for h_s in range(0, h - step, step):
                input_img = wsi.read_region((w_s, h_s), patch_level, (self.patch_size, self.patch_size))
                if is_background(input_img):
                    continue
                if isinstance(input_img, Image.Image):
                    input_img = input_img.convert('RGB')
                    input_img = input_img.resize((1536, 1536))
                else:
                    input_img = cv2.resize(input_img, (1536, 1536), interpolation=cv2.INTER_LINEAR)
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
                coords, labels = self.inference(input_img)
                new_coords = []
                for (x1, y1, x2, y2) in coords:
                    x1 = int(x1 * 1.33333 + w_s) * wsi.level_downsamples[patch_level]
                    y1 = int(y1 * 1.33333 + h_s) * wsi.level_downsamples[patch_level]
                    x2 = int(x2 * 1.33333 + w_s) * wsi.level_downsamples[patch_level]
                    y2 = int(y2 * 1.33333 + h_s) * wsi.level_downsamples[patch_level]
                    new_coords.append([[[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]])
                t_coords += new_coords
                t_labels += labels

        self.qupath_feature(t_coords, t_labels, base)

    @property
    def slides(self):
        if self.slide_list:
            slides = self.slide_list
        else:
            slides = [f for f in os.listdir(self.slide_dir) if os.path.isfile(os.path.join(self.slide_dir, f))]
        return slides

    def run_(self):
        for slide in self.slides:
            self.patch_process(slide)

    def parallel_run(self):
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.patch_process, slide) for slide in self.slides]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='/data2/lbliao/Code/ultralytics/runs/detect/train5/weights/best.pt')
parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Seg-Relabel', help='patch directory')
parser.add_argument('--gpu_ids', type=str, default='1', help='patch directory')
parser.add_argument('--patch_dir', type=str, default='', help='patch directory')
parser.add_argument('--slide_dir', type=str, default='', help='patch directory')
parser.add_argument('--coord_dir', type=str, default='', help='coord directory')
parser.add_argument('--geo_ann_dir', type=str, default='', help='geo annotation directory')
parser.add_argument('--output_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Seg-Relabel/result/0224/', help='output directory')
parser.add_argument('--patch_size', type=int, default=2048, help='patch size')
parser.add_argument('--patch_level', type=int, default=0, help='patch size')
parser.add_argument('--output_size', type=int, default=1024, help='output size')
parser.add_argument('--skip_done', action='store_true', help='skip done')
# parser.add_argument('--slide_list', type=list,
#                     default=['122655.22024-11-01_15_54_23.kfb', '13521N2024-11-01_15_50_35.kfb', '14228N2024-11-01_15_45_17.kfb', '1527467N2024-11-01_14_18_36.kfb', '1536401N2024-11-01_14_38_56.kfb', '1537852N2024-11-01_14_33_16.kfb',
#                              '1539289N2024-11-01_16_15_14.kfb', '1540421N2024-11-01_16_12_52.kfb', '1604428.52024-11-01_14_14_14.kfb', '1734377.32024-11-01_15_18_28.kfb', '202304818I12024-11-12_13_14_11.kfb', '202305993I12024-11-12_13_19_29.kfb',
#                              '202310971A32024-11-20_11_33_01.kfb', '202310971B12024-11-20_11_34_14.kfb', '202310971B42024-11-20_11_35_39.kfb', '202310971D32024-11-12_13_22_26.kfb', '202310971D52024-11-12_13_24_02.kfb', '202311058B72024-11-12_13_26_46.kfb',
#                              '202312644J12024-11-20_11_36_57.kfb'])
parser.add_argument('--slide_list', type=list)
if __name__ == '__main__':
    args = parser.parse_args()
    # GeoAnnotation(args).parallel_run()
    # LMAnnotation(args).parallel_run()
    YOLOAnnotation(args).parallel_run()
