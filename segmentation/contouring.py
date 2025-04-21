import argparse
import json
import math
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
from scipy.optimize import curve_fit

sys.path.insert(0, r'/data2/lbliao/Code/aslide/')
from aslide import Aslide


def is_background(img, threshold=5):
    img_array = np.array(img)
    pixel_max = np.max(img_array, axis=2)
    pixel_min = np.min(img_array, axis=2)
    difference = pixel_max - pixel_min
    return np.sum(difference > threshold) < 800000


def affine_transform(points, a, b, c, d, e, f):
    """
    Apply an affine transformation to a set of 2D points.

    Parameters:
    - points: The input points (numpy array of shape [n, 2]).
    - a, b, c, d, e, f: The affine transformation parameters.

    Returns:
    - transformed_points: The transformed points (numpy array of shape [n, 2]).
    """
    # Perform the affine transformation
    # The points are expected to be in the form of [x, y]
    x_new = a * points[:, 0] + b * points[:, 1] + c
    y_new = d * points[:, 0] + e * points[:, 1] + f

    return np.column_stack((x_new, y_new)).flatten()


def generate_param(src_points, dst_points, filename, output_path):
    points1, points2 = src_points, dst_points.flatten()
    popt, _ = curve_fit(affine_transform, points1, points2, p0=[0, 0, 0, 0, 0, 0])
    result = {filename: list(popt)}
    with open(output_path, 'w') as f:
        json.dump(result, f)


def geojson2txt(filepath, typ='.geojson'):
    # 通常一批文件放在一个路径下
    file_list = os.listdir(filepath)
    for file in file_list:
        if not file.endswith(typ):
            continue
        json_path = os.path.join(filepath, file)
        txt_path = json_path.replace(typ, '.txt')
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        with open(txt_path, 'w') as txt:
            coordinates = json_data.get('features', [])[0].get('geometry', {}).get('coordinates', [])
            for coord in coordinates:
                x_value = coord[0]
                y_value = coord[1]
                txt.write("x:{}, y:{}\n".format(x_value, y_value))


def json2txt(filepath):
    file_list = os.listdir(filepath)
    for file in file_list:
        if not file.endswith('.json'):
            continue
        json_path = os.path.join(filepath, file)
        txt_path = json_path.replace('.json', '.txt')
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        with open(txt_path, 'w') as txt:
            for point in json_data:
                region = point.get("region", {})
                x, y = region.get("x", 0), region.get("y", 0)
                txt.write("x:{}, y:{}\n".format(x, y))


def get_points_from_txt(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            x_value = int(float(values[0].split(':')[1].strip()))
            y_value = int(float(values[1].split(':')[1].strip()))
            points.append([x_value, y_value])
    return np.array(points)


def get_contours(image):
    # 根据色彩范围，使用 opencv 框出目标
    # lower_bound = np.array([40, 30, 30])
    # upper_bound = np.array([140, 130, 120])
    lower_bound = np.array([50, 50, 50])
    upper_bound = np.array([110, 150, 180])

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    mask = cv2.inRange(image, lower_bound, upper_bound)
    dark_region = cv2.bitwise_not(mask)

    cnt_info = cv2.findContours(dark_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cnt_info


class Contouring:
    def __init__(self, opt):
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, 'slides')
        self.ihc_slide_dir = opt.ihc_slide_dir if opt.ihc_slide_dir else os.path.join(opt.data_root, 'IHC')
        self.output_dir = opt.output_dir if opt.output_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/contour')
        self.points_dir = os.path.join(opt.data_root, f'points')
        self.patch_size = opt.patch_size
        self.ihc_ext = opt.ihc_ext
        self.slide_list = opt.slide_list
        os.makedirs(self.output_dir, exist_ok=True)

    def pre_process(self):
        pass

    def process(self, slide):
        raise NotImplementedError()

    def post_process(self, data):
        raise NotImplementedError()


class GeoContouring(Contouring):
    def __init__(self, opt):
        super().__init__(opt)
        json2txt(self.points_dir)
        geojson2txt(self.points_dir)

    def get_reg_param(self, filename):
        # he_points = get_points_from_txt(os.path.join(self.points_dir, f'{filename}.txt'))
        # ihc_points = get_points_from_txt(os.path.join(self.points_dir, f'{filename}-{self.ihc_ext}.txt'))
        ihc_file = filename.replace(self.ihc_ext, '')
        points1 = get_points_from_txt(os.path.join(self.points_dir, f'{filename}.txt'))
        points2 = get_points_from_txt(os.path.join(self.points_dir, f'{ihc_file}.txt'))
        points2 = points2.flatten()
        popt, _ = curve_fit(affine_transform, points1, points2, p0=[0, 0, 0, 0, 0, 0])
        return list(popt)

    def process(self, slide):
        base, ext = os.path.splitext(slide)
        slide_path = os.path.join(self.slide_dir, slide)
        if ext == '.kfb':
            wsi = Aslide(slide_path)
            width, height = wsi.level_dimensions[0]
        elif ext == '.tif':
            wsi = Image.open(slide_path)
            width, height = wsi.size[0], wsi.size[1]
        else:
            wsi = openslide.OpenSlide(slide_path)
            width, height = wsi.level_dimensions[0]
        step = self.patch_size

        features = []
        for w_s in range(0, width - step, step):
            for h_s in range(0, height - step, step):
                if ext == '.tif':
                    input_img = wsi.crop((w_s, h_s, w_s + step, h_s + step))
                else:
                    input_img = wsi.read_region((w_s, h_s), 0, (self.patch_size, self.patch_size))
                if is_background(input_img):
                    continue
                coords, hierarchy = get_contours(input_img)

                feature = self.post_process((coords, hierarchy, base, w_s, h_s))
                features.extend(feature)
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        with open(os.path.join(self.output_dir, f'{base}.geojson'), 'w') as f:
            json.dump(geojson, f, indent=2)
            logger.info(f'generated {base}.geojson contour json!!!')

    def post_process(self, params):
        coords, hierarchy, base, w_s, h_s = params
        features = []
        (a, b, c, d, e, f) = self.get_reg_param(base)
        for cnt, h in zip(coords, hierarchy[0]):
            cnt = np.squeeze(cnt, axis=1)
            cnt += np.array([w_s - 3, h_s - 3])
            cnt = affine_transform(cnt, a, b, c, d, e, f)
            cnt = np.reshape(cnt, (len(cnt) // 2, 1, 2))
            cnt = np.int32(cnt)  # 或者使用 np.float32
            area = cv2.contourArea(cnt)
            patch_area = int(self.patch_size * 0.75) ** 2
            parent_area = cv2.contourArea(coords[h[3]]) if h[3] != -1 else float('inf')

            # 存在父contour 且 父contour不为整张图的  且 父contour面积远大于子contour 且 子contour面积很小
            if patch_area // 20000 < area < patch_area and not (h[3] != -1 and parent_area < patch_area and area < parent_area // 200):

                # 轮廓平滑 CV2 的平滑太锐利了
                start = 0
                while start < len(cnt):
                    start_point = cnt[start][0]
                    for i, end_point in enumerate(cnt[start + 5: start + 25]):
                        end_point = end_point[0]
                        if math.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2) < 25:
                            cnt = np.vstack((cnt[:start], cnt[start + i + 5:]))
                            break
                    start += 1

                cnt = cnt.reshape((-1, 2))
                cnt = cnt.tolist()
                cnt.append(cnt[0])

                if len(cnt) > 3:
                    feature = {
                        "type": "Feature",
                        "id": str(uuid.uuid4()),
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [cnt],
                        }
                    }
                    features.append(feature)
        return features

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


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Detection/0418/', help='patch directory')
parser.add_argument('--slide_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Detection/0418/IHC', help='patch directory')
parser.add_argument('--ihc_slide_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Detection/0418/slides', help='patch directory')
parser.add_argument('--output_dir', type=str, default='/NAS2//Data1/lbliao/Data/MXB/Detection/0418/geojson', help='output directory')
parser.add_argument('--patch_size', type=int, default=4096, help='patch size')
parser.add_argument('--ihc_ext', type=str, default='-CK', help='patch size')
parser.add_argument('--slide_list', type=list)  # , default=['202303007A2.kfb'])
if __name__ == '__main__':
    args = parser.parse_args()
    GeoContouring(args).parallel_run()
