import argparse

import numpy as np
import cv2
import os
import json

from loguru import logger
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString, MultiLineString
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from segmentation.wsi import WSIOperator


def get_contours(image):
    """检测图像中的轮廓，使用改进的颜色范围和处理流程"""
    # 使用更精确的颜色范围
    lower_bound = np.array([10, 10, 10])
    upper_bound = np.array([210, 190, 180])
    # 基底
    # lower_bound = np.array([220, 0, 0])
    # upper_bound = np.array([240, 100, 100])

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 添加黑色边框避免边界效应
    image = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 使用HSV颜色空间提升检测鲁棒性
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 形态学操作改善轮廓完整性
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 检测轮廓并加入面积筛选
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


class Contouring:
    def __init__(self, opt):
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, 'slides')
        self.ihc_slide_dir = opt.ihc_slide_dir if opt.ihc_slide_dir else os.path.join(opt.data_root, 'IHC')
        self.output_dir = opt.output_dir if opt.output_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/contour')
        self.points_dir = os.path.join(opt.data_root, 'points')
        self.patch_size = opt.patch_size
        self.ihc_ext = opt.ihc_ext
        self.slide_list = opt.slide_list if opt.slide_list else []
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def slides(self):
        if self.slide_list:
            return self.slide_list
        return [f for f in os.listdir(self.slide_dir) if os.path.isfile(os.path.join(self.slide_dir, f))]

    def process(self, slide):
        """处理单个切片"""
        raise NotImplementedError()

    def run(self, parallel=True):
        """执行轮廓检测"""
        if parallel:
            self.parallel_run()
        else:
            self.run_()

    def run_(self):
        for slide in self.slides:
            self.process(slide)

    def parallel_run(self):
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.process, slide): slide for slide in self.slides}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    slide = futures[future]
                    print(f"Error processing {slide}: {str(e)}")
                    traceback.print_exc()


class GeoContouring(Contouring):
    def __init__(self, opt):
        super().__init__(opt)
        # 初始化注册参数
        self.reg_params_cache = {}

    def process_slide_patch(self, w_s, h_s, wsi, step):
        """处理单个切片块"""
        width, height = wsi.level_dimensions[0]
        x, y = min(step, width - w_s), min(step, height - h_s)
        input_img = wsi.read_region((w_s, h_s), 0, (x, y), check_background=True)

        # 跳过背景区域
        if input_img is None:
            return []

        # 获取轮廓
        contours, hierarchy = get_contours(input_img)
        features = []
        if hierarchy is None:
            return []
        for cnt, hie in zip(contours, hierarchy[0]):
            # 跳过无效轮廓
            if len(cnt) < 3:
                continue
            if hie[3] != -1:
                continue
            # 添加坐标偏移
            cnt = cnt - 3 + np.array([w_s, h_s])

            # 创建特征
            feature = self.create_feature(cnt)
            if feature:
                features.extend(feature)

        return features

    @staticmethod
    def split_multipolygon(geom):
        """将 MultiPolygon 拆分为多个 Polygon 对象"""
        if isinstance(geom, MultiPolygon):
            return list(geom.geoms)  # 返回 Polygon 列表
        elif isinstance(geom, Polygon):
            return [geom]  # 单个 Polygon 包装为列表
        else:
            raise ValueError("非 Polygon 或 MultiPolygon 类型")

    def create_feature(self, cnt):
        """创建GeoJSON特征"""
        # 计算面积筛选
        area = cv2.contourArea(cnt)
        patch_area = self.patch_size ** 2

        # 过滤小面积和过大轮廓
        # if area < patch_area * 0.0001 or area > patch_area * 0.95:
        if area > patch_area * 0.95:
            return None

        # 转换为点列表
        points = cnt.squeeze().tolist()
        if len(points) < 3:
            return None

        # 闭合多边形
        if points[0] != points[-1]:
            points.append(points[0])

        # 确保有效几何
        features = []
        try:
            # 创建并修复几何
            polygon = Polygon(points)
            polygon = make_valid(polygon)

            # 检查有效性及空几何
            if polygon.is_empty or not polygon.is_valid:
                return []  # 返回空列表表示跳过

            polygons = []

            def line2poly(line):
                coords = list(line.coords)

                # 检查是否闭合（首尾点相同）
                if coords[0] != coords[-1]:
                    coords.append(coords[0])  # 添加首点使线闭合

                # 创建闭合多边形
                return Polygon(coords)

            if isinstance(polygon, (MultiPolygon, GeometryCollection)):
                for geom in polygon.geoms:
                    if isinstance(geom, Polygon) and not geom.is_empty:
                        polygons.append(geom)
                    elif isinstance(geom, MultiPolygon):
                        polygons.extend(geom.geoms)  # 返回 Polygon 列表
                    elif isinstance(geom, LineString):
                        # 获取线坐标点
                        polygons.append(line2poly(geom))
                    elif isinstance(geom, MultiLineString):
                        for geo in geom.geoms:
                            polygons.append(line2poly(geo))

            elif isinstance(polygon, Polygon) and not polygon.is_empty:
                polygons.append(polygon)

            for poly in polygons:
                # 获取外轮廓坐标（此时 poly 是 Polygon，有 exterior 属性）
                poly_area = poly.area
                if poly_area < patch_area * 0.001:
                    continue  # 跳过当前多边形
                exterior_coords = list(poly.exterior.coords)
                exterior_coords = [(point[0], point[1]) for point in exterior_coords]

                # 构建 GeoJSON Feature
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [exterior_coords]
                    },
                    "properties": {
                        "objectType": "annotation",
                        "classification": {
                            "name": "prostate",
                            "color": [0, 255, 0]
                        }
                    }
                }
                features.append(feature)
        except:
            logger.error(traceback.format_exc())
            return None

        return features

    def process(self, slide):
        """处理单个切片"""
        base_name, ext = os.path.splitext(slide)
        slide_path = os.path.join(self.slide_dir, slide)
        features = []

        wsi = WSIOperator(slide_path)
        width, height = wsi.level_dimensions[0]

        step = self.patch_size

        # 使用网格步长处理
        for w_s in range(0, width, step):
            for h_s in range(0, height, step):
                patch_features = self.process_slide_patch(w_s, h_s, wsi, step)
                features.extend(patch_features)

        # # 保存结果
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "slide_name": slide,
                "dimensions": [width, height],
                "patch_size": self.patch_size
            }
        }
        # gdf = gpd.GeoDataFrame(geojson, crs="EPSG:4326")
        output_path = os.path.join(self.output_dir, f'{base_name}.geojson')
        # processor = GeoJSONProcessor(gdf, output_path)
        # processor.execute(patch_size=4096)
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
            logger.info(f'Generated {output_path} with {len(features)} features.')


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/classification/第一批', help='patch directory')
parser.add_argument('--slide_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/classification/第一批/IHC', help='patch directory')
parser.add_argument('--ihc_slide_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/classification/第一批/slides', help='patch directory')
parser.add_argument('--output_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/classification/第一批/label', help='output directory')
parser.add_argument('--patch_size', type=int, default=4096, help='patch size')
parser.add_argument('--ihc_ext', type=str, default='-CK', help='patch size')
parser.add_argument('--slide_list', type=list)
if __name__ == '__main__':
    args = parser.parse_args()
    GeoContouring(args).parallel_run()
    # KVContouring(args).parallel_run()
