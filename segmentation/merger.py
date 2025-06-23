import argparse
import json
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import uuid

from scipy.optimize import curve_fit
from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection
from shapely.ops import unary_union, transform
from shapely import make_valid
from concurrent.futures import ProcessPoolExecutor  # 改为多进程处理
from tqdm import tqdm
from rtree import index
from pathlib import Path
from functools import partial
from loguru import logger
from segmentation.wsi import WSIOperator

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
BUFFER_SIZE = 10


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


class GeoJSONProcessor:
    """高效处理GeoJSON中多边形合并与属性继承

    Attributes:
        input_path (str): 输入GeoJSON文件路径
        output_path (str): 输出GeoJSON文件路径
        gdf (gpd.GeoDataFrame): 地理数据框
        sindex (rtree.index.Index): 空间索引
        crs (pyproj.CRS): 坐标参考系统
        bounds (tuple): 数据边界 (minx, miny, maxx, maxy)
    """

    def __init__(self, input_path, output_path, points_dir, ihc_ext, simplify_tolerance=0.01):
        """
        初始化处理器

        :param input_path: 输入GeoJSON路径
        :param output_path: 输出GeoJSON路径
        :param simplify_tolerance: 几何简化容差值
        """
        self.input_path = input_path
        self.output_path = output_path
        self.points_dir = points_dir
        self.ihc_ext = ihc_ext
        self.simplify_tolerance = simplify_tolerance
        self.load_data()

    def load_data(self):
        """加载并预处理数据"""
        # 只加载必要列以减少内存占用
        self.gdf = gpd.read_file(self.input_path, engine='pyogrio', include_fields=['classification', 'geometry'])
        self.crs = self.gdf.crs
        self.bounds = self.gdf.total_bounds

        # 拆分MultiPolygon并简化几何
        self.gdf.geometry = self.gdf.geometry.apply(
            lambda geom: make_valid(geom) if not geom.is_valid else geom
        )

        self.gdf = self.gdf.explode(index_parts=False)

        # 构建空间索引
        self.gdf.reset_index(drop=True, inplace=True)
        self.sindex = self.gdf.sindex

    def merge_intersecting(self, polygon_a):
        """合并与给定多边形相交的要素"""
        # 使用空间索引加速查询[1,4](@ref)
        candidate_idx = list(self.sindex.intersection(polygon_a.bounds))
        if not candidate_idx:
            return [], []

        candidate_gdf = self.gdf.iloc[candidate_idx]
        intersecting_gdf = candidate_gdf[candidate_gdf.intersects(polygon_a)]

        if intersecting_gdf.empty:
            return [], []

        # 几何合并与拆分
        buffered_geoms = intersecting_gdf.geometry.buffer(BUFFER_SIZE, resolution=16)
        # Step 3: 合并膨胀后的几何
        merged_buffered = unary_union(buffered_geoms)  # [4,10](@ref)

        # Step 4: 对合并结果腐蚀4像素（注意腐蚀需用负距离）
        merged_geom = merged_buffered.buffer(-BUFFER_SIZE, resolution=16)

        # 处理合并结果
        merged_features = []
        if isinstance(merged_geom, Polygon):
            merged_polys = [merged_geom]
        elif isinstance(merged_geom, MultiPolygon):
            merged_polys = list(merged_geom.geoms)
        else:
            return [], intersecting_gdf.index.tolist()

        # 属性继承逻辑
        for poly in merged_polys:
            originals = intersecting_gdf[intersecting_gdf.intersects(poly)]
            merged_features.append({
                "geometry": poly,
                "id": str(uuid.uuid4()),
                "objectType": "annotation",
                "classification": {
                    "name": "prostate",
                    "color": [0, 255, 0]
                },
                "source": "merged"
            })
        return merged_features, intersecting_gdf.index.tolist()

    def resolve_classification(self, originals_gdf):
        """解决合并后的分类冲突（取众数）"""
        # 使用value_counts提高效率[7](@ref)
        counts = originals_gdf['classification'].value_counts()
        return counts.index[0] if not counts.empty else "unknown"

    def _create_stripe_polygon(self, direction, position):
        """创建条带多边形"""
        minx, miny, maxx, maxy = self.bounds
        buffer = BUFFER_SIZE

        if direction == 'horizontal':
            y = position
            return Polygon([
                (0, y - buffer), (maxx, y - buffer),
                (maxx, y + buffer), (0, y + buffer), (0, y - buffer)
            ])
        else:  # vertical
            x = position
            return Polygon([
                (x - buffer, 0), (x - buffer, maxy),
                (x + buffer, maxy), (x + buffer, 0), (x - buffer, 0)
            ])

    def process_stripes(self, patch_size):
        """并行处理横向/纵向条带区域"""
        # minx, miny, maxx, maxy = self.bounds
        minx, miny = 0, 0
        wsi = WSIOperator('/NAS2/Data1/lbliao/Data/MXB/classification/第一批/IHC/1547583.18有癌-CK.kfb')
        maxx, maxy = wsi.level_dimensions[0]
        tasks = []

        # 创建横向条带任务
        horizontal_stripes = np.arange(patch_size, maxy - BUFFER_SIZE, patch_size)
        for y in horizontal_stripes:
            if y >= maxy - BUFFER_SIZE:
                continue
            poly = self._create_stripe_polygon('horizontal', y)
            tasks.append(poly)

        # 创建纵向条带任务
        vertical_stripes = np.arange(minx, maxx, patch_size)
        for x in vertical_stripes:
            if x >= maxx - BUFFER_SIZE:
                continue
            poly = self._create_stripe_polygon('vertical', x)
            tasks.append(poly)

        # 并行执行 - 使用多进程处理CPU密集型任务[6,7](@ref)
        merged_all = []
        removed_ids = set()

        # 使用partial绑定self引用
        merge_func = partial(self.merge_intersecting)

        with ProcessPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(merge_func, tasks),
                total=len(tasks),
                desc="处理条带"
            ))

        for features, ids in results:
            merged_all.extend(features)
            removed_ids.update(ids)

        # 构建最终数据集
        merged_gdf = gpd.GeoDataFrame(merged_all, crs=self.crs)
        merged_gdf['classification'] = {
            "name": "prostate",
            "color": [0, 255, 0]
        }
        non_intersecting_gdf = self.gdf[~self.gdf.index.isin(removed_ids)].copy()
        non_intersecting_gdf['id'] = str(uuid.uuid4())
        non_intersecting_gdf['objectType'] = 'Annotation'
        non_intersecting_gdf['classification'] = {
            "name": "prostate",
            "color": [0, 255, 0]
        }
        non_intersecting_gdf['source'] = 'original'

        return pd.concat([merged_gdf, non_intersecting_gdf], ignore_index=True)

    def remove_small_holes(self, geom, min_area):
        if geom.is_empty or geom.geom_type not in ['Polygon', 'MultiPolygon']:
            return geom
        # 处理单多边形
        if geom.geom_type == 'Polygon':
            exterior = geom.exterior
            interiors = []
            for interior in geom.interiors:
                if Polygon(interior).area >= min_area:
                    interiors.append(interior)
            return Polygon(exterior, interiors) if interiors else Polygon(exterior)
        # 处理多部件多边形
        else:
            return MultiPolygon([
                self.remove_small_holes(part, min_area) for part in geom.geoms
            ])

    def get_reg_param(self):
        path = Path(self.input_path)
        basename = path.stem
        # he_points = get_points_from_txt(os.path.join(self.points_dir, f'{basename}.txt'))
        # ihc_points = get_points_from_txt(os.path.join(self.points_dir, f'{basename}-{self.ihc_ext}.txt'))
        # TODO -CK 替换成对应后缀
        ihc_file = basename.replace(self.ihc_ext, '')
        points1 = get_points_from_txt(os.path.join(self.points_dir, f'{basename}.txt'))
        points2 = get_points_from_txt(os.path.join(self.points_dir, f'{ihc_file}.txt'))
        points2 = points2.flatten()
        popt, _ = curve_fit(affine_transform, points1, points2, p0=[0, 0, 0, 0, 0, 0])
        return list(popt)

    def transform_geometry(self, geometry, transform_params):
        """
        Apply affine transformation to a Shapely geometry object.
        Supports Point, LineString, Polygon, and their Multi- variants.
        """
        a, b, c, d, e, f = transform_params

        def _transform_point(x, y, z=None):
            """Helper function to transform a single coordinate"""
            new_x = a * x + b * y + c
            new_y = d * x + e * y + f
            return (new_x, new_y) if z is None else (new_x, new_y, z)

        return transform(partial(_transform_point), geometry)

    def merge_close_holes(self, polygon, max_distance=0.5):
        """合并距离小于阈值的孔洞"""
        if polygon.is_empty or not polygon.interiors:
            return polygon

        # 提取所有孔洞
        holes = [Polygon(hole) for hole in polygon.interiors]
        merged_holes = []
        skip_indices = set()

        # 遍历孔洞，计算两两间距
        for i in range(len(holes)):
            if i in skip_indices:
                continue
            current_hole = holes[i]
            merged_hole = current_hole

            # 寻找邻近孔洞
            for j in range(i + 10, len(holes)):
                if j in skip_indices:
                    continue
                other_hole = holes[j]
                distance = current_hole.distance(other_hole)

                # 若间距小于阈值，则合并
                if distance < max_distance:
                    # 寻找最近点对
                    nearest_points = np.array(current_hole.exterior.coords)
                    other_points = np.array(other_hole.exterior.coords)
                    min_dist = float('inf')
                    p1, p2 = None, None

                    for pt1 in nearest_points:
                        for pt2 in other_points:
                            dist = np.linalg.norm(pt1 - pt2)
                            if dist < min_dist:
                                min_dist = dist
                                p1, p2 = pt1, pt2

                    # 构建桥接边并合并孔洞
                    if p1 is not None and p2 is not None:
                        bridge = LineString([p1, p2])
                        merged_hole = unary_union([merged_hole, other_hole, bridge])
                        skip_indices.add(j)

            merged_holes.append(merged_hole)

        # 重构多边形（保留外轮廓，替换合并后的孔洞）
        new_interiors = [merged_hole.exterior.coords for merged_hole in merged_holes if not isinstance(merged_hole, GeometryCollection)]
        return Polygon(polygon.exterior, holes=new_interiors)

    def execute(self, patch_size=2048):
        """执行处理流程"""
        try:
            transform_params = self.get_reg_param()
        except FileNotFoundError:
            logger.info(f'{Path(self.input_path).stem} 无配准点，跳过')
            return
        result_gdf = self.process_stripes(patch_size)

        # 移除面积小于5000的孔洞
        result_gdf['geometry'] = result_gdf['geometry'].apply(
            lambda geom: self.remove_small_holes(geom, min_area=5000)
        )
        result_gdf['geometry'] = result_gdf['geometry'].apply(
            lambda geom: self.transform_geometry(geom, transform_params)
        )
        result_gdf['geometry'] = result_gdf['geometry'].apply(
            lambda geom: self.merge_close_holes(geom, 10)
        )
        buffered = unary_union(result_gdf.geometry)
        result_gdf = gpd.GeoDataFrame(geometry=[buffered], crs=self.crs)
        gdf_separated = result_gdf.explode(ignore_index=True)
        gdf_separated.to_file(self.output_path, driver='GeoJSON')
        return f"处理完成！已保存至 {self.output_path}"


def remove_intersecting_polygons_rtree(detect_path, seg_path, output_path):
    """
    使用R树索引优化大数据集性能

    参数:
        geojson1_path: 参考GeoJSON文件路径
        geojson2_path: 需要处理的GeoJSON文件路径
        output_path: 输出文件路径
    """
    # 读取数据
    if not os.path.exists(detect_path):
        logger.info(f'{Path(detect_path).stem} not exists')
        return
    detect = gpd.read_file(detect_path)
    seg = gpd.read_file(seg_path)

    # 确保相同坐标系
    if detect.crs != seg.crs:
        seg = seg.to_crs(detect.crs)

    # 为参考多边形创建R树索引
    idx = index.Index()
    for i, geom in enumerate(detect['geometry']):
        idx.insert(i, geom.bounds)

    # 找出需要移除的多边形
    to_remove = set()
    for i, geom in enumerate(seg['geometry']):
        # 查询可能相交的参考多边形
        for j in idx.intersection(geom.bounds):
            if geom.intersects(detect.iloc[j]['geometry']):
                to_remove.add(i)
                break

    # 保留不相交的多边形
    result = seg.drop(index=list(to_remove))

    # 保存结果
    result.to_file(output_path, driver='GeoJSON')
    logger.info(f"已移除{len(to_remove)}个相交的多边形，结果已保存到{output_path}")


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/classification/第一批', help='patch directory')
parser.add_argument('--input_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/classification/第一批/label', help='patch directory')
parser.add_argument('--point_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/classification/第一批/points', help='patch directory')
parser.add_argument('--output_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/classification/第一批/label', help='output directory')
parser.add_argument('--detect_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/classification/第一批/detect', help='output directory')
parser.add_argument('--patch_size', type=int, default=4096, help='patch size')
parser.add_argument('--ihc_ext', type=str, default='-CK', help='patch size')
if __name__ == "__main__":
    args = parser.parse_args()
    # input_dir = args.input_dir
    # geojson2txt(args.point_dir)
    # json2txt(args.point_dir)
    # for file in os.listdir(input_dir):
    #     processor = GeoJSONProcessor(
    #         input_path=os.path.join(input_dir, file.replace(args.ihc_ext, '')),
    #         output_path=os.path.join(args.output_dir, file),
    #         points_dir=args.point_dir,
    #         ihc_ext=args.ihc_ext,
    #         simplify_tolerance=0.01
    #     )
    #     processor.execute(patch_size=4096)
    for file in os.listdir(args.output_dir):
        remove_intersecting_polygons_rtree(
            detect_path=os.path.join(args.detect_dir, file.replace('-CK', '')),
            seg_path=os.path.join(args.output_dir, file),
            output_path=os.path.join(args.output_dir, file),
        )
