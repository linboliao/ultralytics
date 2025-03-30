import argparse
import json
import math
import os
import random
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

sys.path.insert(0, r'/data2/lbliao/Code/aslide/')
from aslide import Aslide

MIN_AREA = 3000


def is_background(img, threshold=5):
    img_array = np.array(img)
    pixel_max = np.max(img_array, axis=2)
    pixel_min = np.min(img_array, axis=2)
    difference = pixel_max - pixel_min
    return np.sum(difference > threshold) < 500000


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
        self.geo_ann_dir = opt.geo_ann_dir if opt.geo_ann_dir else os.path.join(opt.data_root, f'geojson')
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
        base, ext = os.path.splitext(slide)
        logger.info(f'start to process {slide}')

        if ext == '.kfb':
            wsi = Aslide(wsi_path)
            width, height = wsi.level_dimensions[0]
            mpp = wsi.mpp
        elif ext == '.tif':
            wsi = Image.open(wsi_path)
            width, height = wsi.size[0], wsi.size[1]
            mpp = 20
        else:
            wsi = openslide.OpenSlide(wsi_path)
            width, height = wsi.level_dimensions[0]
            mpp = int(wsi.properties.get('aperio.AppMag', '20'))
        ann_path = os.path.join(self.geo_ann_dir, f'{base}.geojson')
        with open(ann_path, 'r', encoding='utf-8') as file:
            anns = json.load(file)
        features = anns.get('features')
        step = int(self.patch_size * (mpp / 20))
        for w in range(0, width - step, step):
            for h in range(0, height - step, step):
                input_img = wsi.read_region((w, h), 0, (step, step))
                if is_background(input_img):
                    continue

                if self.skip_done and os.path.isfile(os.path.join(self.image_dir, f'{base}_{w}_{h}.png')):
                    continue
                patch_coords = []
                label_path = os.path.join(self.label_dir, f'{base}_{w}_{h}.txt')
                to_remove = []

                def contour(data, _w, _h):
                    lc_coords = []
                    if any(_w < a < _w + self.patch_size and _h < b < _h + self.patch_size for (a, b) in data):
                        data = [[a - _w, b - _h] for [a, b] in data]
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
                            if name == 'non-cancer' or name == 'Negative' or color == [0, 255, 0]:
                                clazz = 0
                            elif name == 'Region*':
                                clazz = 1
                            elif name == 'Necrosis':
                                clazz = 2
                            elif name == 'Other':
                                return
                            else:
                                clazz = 1
                            line = f'{clazz} {contours_str}'
                            f.write(line + '\n')
                        if random.random() < 0.3:
                            to_remove.append(feature)

                with open(label_path, 'w') as f:
                    for feature in features:
                        coordinates = feature['geometry']['coordinates']
                        if feature['geometry']['type'] == 'Polygon':
                            for coords in coordinates:
                                if isinstance(coords, list):
                                    contour(coords, w, h)
                        elif feature['geometry']['type'] == 'MultiPolygon':
                            for coords in coordinates:
                                for sub_coords in coords:
                                    if isinstance(sub_coords, list):
                                        contour(sub_coords, w, h)
                    for feature in to_remove:
                        features.remove(feature)

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
        # self.lm_ann_dir = os.path.join(self.output_dir, f'lm_annotations/')
        self.lm_ann_dir = '/NAS2/Data1/lbliao/Data/MXB/LabelMe/20250224/labelme'
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
                if shape.get('label') in ['prostate', '电切烧灼腺体']:
                    clazz = 0
                elif shape.get('label') in ['cancer']:
                    clazz = 1
                elif shape.get('label') in ['血管']:
                    clazz = 2
                elif shape.get('label') in ['神经节']:
                    clazz = 3
                elif shape.get('label') in ['鳞状上皮']:
                    clazz = 4
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


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Detection/cellvit+', help='patch directory')
parser.add_argument('--gpu_ids', type=str, default='0', help='patch directory')
parser.add_argument('--patch_dir', type=str, default='', help='patch directory')
parser.add_argument('--slide_dir', type=str, default='', help='patch directory')
parser.add_argument('--coord_dir', type=str, default='', help='coord directory')
parser.add_argument('--geo_ann_dir', type=str, default='', help='geo annotation directory')
parser.add_argument('--output_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Detection/cellvit+/dataset', help='output directory')
parser.add_argument('--patch_size', type=int, default=2048, help='patch size')
parser.add_argument('--patch_level', type=int, default=0, help='patch size')
parser.add_argument('--output_size', type=int, default=2048, help='output size')
parser.add_argument('--skip_done', action='store_true', help='skip done')
parser.add_argument('--slide_list', type=list)
if __name__ == '__main__':
    args = parser.parse_args()
    # YOLOAnnotation(args).run_()
    GeoAnnotation(args).parallel_run()
    # LMAnnotation(args).parallel_run()
