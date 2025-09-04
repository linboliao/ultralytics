import argparse
import json
import os
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import uuid

from scipy.optimize import curve_fit
from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection, MultiLineString
from shapely.ops import unary_union, transform
from shapely import make_valid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # 改为多进程处理
from tqdm import tqdm
from rtree import index
from pathlib import Path
from functools import partial
from loguru import logger
from segmentation.wsi import WSIOperator

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
BUFFER_SIZE = 10
import warnings

warnings.filterwarnings("ignore")  # 忽略所有警告[1,2,5](@ref)


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
        # TODO 修改 slide 后缀
        self.slide_path = Path(input_path.replace('/segment/', '/slides/').replace('.geojson', '.kfb'))
        # self.slide_path = '/NAS2/Data1/lbliao/Data/MXB/Detection/0318/slides/1834976T_可疑.svs'
        self.input_path = input_path
        self.output_path = output_path
        self.points_dir = points_dir
        self.ihc_ext = ihc_ext
        self.simplify_tolerance = simplify_tolerance
        # try:
        #     self.transform_params = self.get_reg_param()
        # except FileNotFoundError:
        #     self.transform_params = None
        #     logger.info(f'{Path(self.input_path).stem} 无配准点，跳过')
        #     return
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
        prostate, cancer = 0, 0
        for cls in intersecting_gdf['classification']:
            name = json.loads(cls.replace("'", '"')).get('name', 'cancer')
            if name == 'prostate':
                prostate += 1
            else:
                cancer += 1

        # 根据检查结果确定合并要素分类
        if prostate >= cancer:
            classification = {"name": "prostate", "color": [0, 255, 0]}
        else:
            classification = {"name": "cancer", "color": [255, 0, 0]}

        # 属性继承逻辑
        for poly in merged_polys:
            merged_features.append({
                "geometry": poly,
                "id": str(uuid.uuid4()),
                "objectType": "annotation",
                "classification": classification,
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
        if not os.path.exists(self.slide_path):
            maxx, maxy = 100000, 100000
        else:
            wsi = WSIOperator(self.slide_path)
            maxx, maxy = wsi.level_dimensions[0]
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
        if not os.path.exists(self.slide_path):
            maxx, maxy = 100000, 100000
        else:
            wsi = WSIOperator(self.slide_path)
            maxx, maxy = wsi.level_dimensions[0]
        tasks = []

        # 创建横向条带任务
        horizontal_stripes = np.arange(patch_size, maxy - BUFFER_SIZE, patch_size)
        vertical_stripes = np.arange(minx, maxx, patch_size)
        # for y in horizontal_stripes:
        #     if y >= maxy - BUFFER_SIZE:
        #         continue
        #     poly = self._create_stripe_polygon('horizontal', y)
        #     tasks.append(poly)
        #
        # # 创建纵向条带任务
        # for x in vertical_stripes:
        #     if x >= maxx - BUFFER_SIZE:
        #         continue
        #     poly = self._create_stripe_polygon('vertical', x)
        #     tasks.append(poly)
        b, p = BUFFER_SIZE, patch_size
        for x in vertical_stripes:
            for y in horizontal_stripes:
                poly = Polygon([
                    (x - b, y - b), (x - b, y + b + p), (x + b + p, y + b + p), (x + b + p, y - b), (x + b, y - b),  # 外圈
                    (x + b, y + b), (x - b + p, y + b), (x - b + p, y - b + p), (x + b, y - b + p), (x + b, y + b)  # 内圈
                ])
                tasks.append(poly)
        # 并行执行 - 使用多进程处理CPU密集型任务[6,7](@ref)
        merged_all = []
        removed_ids = set()

        # 使用partial绑定self引用
        merge_func = partial(self.merge_intersecting)

        with ThreadPoolExecutor(max_workers=15) as executor:
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

        non_intersecting_gdf = self.gdf[~self.gdf.index.isin(removed_ids)].copy()
        non_intersecting_gdf['id'] = str(uuid.uuid4())
        non_intersecting_gdf['objectType'] = 'Annotation'

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
        if isinstance(polygon, MultiPolygon):
            return MultiPolygon([
                self.merge_close_holes(part, max_distance) for part in polygon.geoms
            ])

        if polygon.is_empty or isinstance(polygon, LineString) or isinstance(polygon, MultiLineString) or not polygon.interiors:
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

    def union(self, result_gdf):
        cancer_gdf = result_gdf[result_gdf['classification'].apply(
            lambda x: 'cancer' in x
        )]
        prostate_gdf = result_gdf[result_gdf['classification'].apply(
            lambda x: 'prostate' in x
        )]
        c_buffered = unary_union(cancer_gdf.geometry)
        c_feature = {
            "geometry": c_buffered,
            "classification": {"name": "cancer", "color": [255, 0, 0]},
            "area": cancer_gdf.area.sum()  # 计算总面积
        }
        p_buffered = unary_union(prostate_gdf.geometry)

        p_feature = {
            "geometry": p_buffered,
            "classification": {"name": "prostate", "color": [0, 255, 0]},
            "area": cancer_gdf.area.sum()  # 计算总面积
        }

        merged_features = [c_feature, p_feature]
        merged_gdf = gpd.GeoDataFrame(merged_features, crs=result_gdf.crs)
        return merged_gdf

    # 定义移除孔洞的函数
    @staticmethod
    def remove_holes(geom):
        """移除几何对象的所有内部孔洞"""
        if geom.geom_type == 'Polygon':
            # 提取外边界，忽略内环（孔洞）
            return Polygon(geom.exterior)
        elif geom.geom_type == 'MultiPolygon':
            # 对每个子多边形递归处理
            return MultiPolygon([Polygon(poly.exterior) for poly in geom.geoms])
        else:
            # 非多边形类型（如点、线）直接返回
            return geom

    def execute(self, patch_size=2048):
        """执行处理流程"""
        # result_gdf = self.process_stripes(patch_size)
        result_gdf = self.gdf
        # # 移除面积小于5000的孔洞
        # result_gdf['geometry'] = result_gdf['geometry'].apply(
        #     lambda geom: self.remove_small_holes(geom, min_area=5000)
        # )

        # transform_params = self.transform_params
        # if not self.transform_params:
        #     return
        # # 仿射变换
        # result_gdf['geometry'] = result_gdf['geometry'].apply(
        #     lambda geom: self.transform_geometry(geom, transform_params)
        # )

        # 平滑
        # result_gdf['geometry'] = result_gdf['geometry'].apply(
        #     lambda geom: self.merge_close_holes(geom, 25)
        # )

        # buffered = unary_union(result_gdf.geometry)
        # result_gdf = gpd.GeoDataFrame(geometry=[buffered], crs=self.crs)
        result_gdf["area"] = result_gdf.geometry.area

        # 筛选面积 ≥ x 的要素
        result_gdf = result_gdf[result_gdf["area"] >= 2000]  # x 为面积阈值（单位需与坐标系一致）

        result_gdf = self.union(result_gdf)
        # result_gdf['geometry'] = result_gdf['geometry'].apply(
        #     lambda geom: geom.simplify(tolerance=2)  # 示例容差10米
        # )
        import geopandas as gpd

        # 计算每个几何图形的面积（单位：坐标系单位）

        # 保存结果
        # result_gdf['geometry'] = result_gdf['geometry'].apply(self.remove_holes)
        gdf_separated = result_gdf.explode(ignore_index=True)
        gdf_separated.to_file(self.output_path, driver='GeoJSON')
        print(f"处理完成！已保存至 {self.output_path}")


def remove_intersects(detect_path, seg_path, output_path, area_ratio_threshold=0.2):
    """
    移除seg中与detect相交且相交面积占detect轮廓面积超过阈值的多边形

    参数:
        detect_path: 参考GeoJSON文件路径(作为基准)
        seg_path: 需要处理的GeoJSON文件路径
        output_path: 输出文件路径
        area_ratio_threshold: 面积比例阈值(0-1)，默认0.1表示10%
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
    for i, seg_geom in enumerate(seg['geometry']):
        # 查询可能相交的参考多边形
        for j in idx.intersection(seg_geom.bounds):
            detect_geom = detect.iloc[j]['geometry']
            if seg_geom.intersects(detect_geom):
                # 计算相交面积
                intersection = seg_geom.intersection(detect_geom)
                intersection_area = intersection.area
                detect_area = detect_geom.area

                # 计算面积比例
                area_ratio = intersection_area / detect_area

                # 如果面积比例超过阈值，则标记为需要移除
                if area_ratio >= area_ratio_threshold:
                    to_remove.add(i)
                    break

    # 保留不需要移除的多边形
    result = seg.drop(index=list(to_remove))

    # 保存结果
    result.to_file(output_path, driver='GeoJSON')
    logger.info(f"已移除{len(to_remove)}个相交且面积比例超过{area_ratio_threshold * 100}%的多边形，结果已保存到{output_path}")


def segment_label(seg_path, detect_path, output_path, area_ratio_threshold=0.2):
    # 读取两个 GeoJSON 文件
    seg_gdf = gpd.read_file(seg_path)
    detect_gdf = gpd.read_file(detect_path)

    detect_gdf = detect_gdf[
        detect_gdf['classification'].apply(
            lambda s:
            json.loads(s).get('name') if s is not None
            else json.loads('{ "name": "cancer", "color": [255, 0, 0] }').get('name')
        ) == 'prostate'
        ]

    # 确保坐标系一致
    if seg_gdf.crs != detect_gdf.crs:
        detect_gdf = detect_gdf.to_crs(seg_gdf.crs)
    #
    # seg_gdf["classification"] = [
    #     {"name": "prostate", "color": [0, 255, 0]}
    #     for _ in range(len(seg_gdf))
    # ]

    # 执行空间连接（保留原始逻辑）
    intersected = gpd.sjoin(
        detect_gdf[["geometry", "classification"]],  # 确保包含几何列
        seg_gdf,
        how="inner",
        predicate="intersects",
        lsuffix="d",
        rsuffix="g"
    )

    # 新增：计算相交面积占比并过滤
    def check_intersection_area(row, seg_gdf):
        # 获取 seg_gdf 中的原始几何
        seg_geom = seg_gdf.loc[row["index_g"], 'geometry']
        # 计算相交部分几何
        intersection = row['geometry'].intersection(seg_geom)

        # 跳过非面状相交（如共享边界）
        if intersection.is_empty or intersection.area == 0:
            return False

        # 计算相交面积占比
        area_ratio = intersection.area / seg_geom.area
        return area_ratio > 0.35  # 阈值设为20%

    # 应用过滤条件
    valid_mask = intersected.apply(
        lambda row: check_intersection_area(row, seg_gdf),
        axis=1
    )
    filtered_intersected = intersected[valid_mask]

    # 更新 seg_gdf 的属性（仅通过过滤的相交项）
    for idx, row in filtered_intersected.iterrows():
        g_index = row["index_g"]
        seg_gdf.loc[g_index, "classification"] = row["classification_d"]

    # 简化几何（可选）
    # seg_gdf['geometry'] = seg_gdf['geometry'].apply(lambda geom: geom.simplify(tolerance=2))

    # 输出结果
    seg_gdf.to_file(output_path, driver="GeoJSON")


import numpy as np
from shapely.geometry import Polygon, MultiPolygon

import geopandas as gpd
from shapely.ops import unary_union


def nnunet(in_path, out_path, buffer=2):
    gdf = gpd.read_file(in_path)
    gdf['geometry'] = gdf['geometry'].apply(
        lambda geom: geom.simplify(tolerance=5)  # 示例容差10米
    )
    # 步骤1: 所有几何对象膨胀2像素
    gdf = gdf.copy()  # 避免修改原始数据
    gdf['buffered'] = gdf.geometry.buffer(buffer)  # 安全：直接覆盖新列

    # 步骤2: 合并相交的几何元素
    merged_geom = unary_union(gdf['buffered'].tolist())
    merged_polys = list(merged_geom.geoms) if merged_geom.geom_type == 'MultiPolygon' else [merged_geom]

    # 步骤3: 为合并后的几何体分配属性
    results = []
    for poly in merged_polys:
        # 定位与当前合并多边形相交的原始几何（非膨胀状态）
        intersecting = gdf[gdf.geometry.intersects(poly)]

        if not intersecting.empty:
            # ✅ 修复点：使用.loc显式赋值避免警告
            intersecting.loc[:, 'area'] = intersecting.geometry.area  # 安全添加临时列
            largest = intersecting.loc[intersecting['area'].idxmax()]
            results.append({
                'geometry': poly,
                'classification': largest['classification']
            })

    merged_gdf = gpd.GeoDataFrame(results, crs=gdf.crs)

    # 步骤4: 腐蚀操作（收缩2像素）
    merged_gdf.geometry = merged_gdf.geometry.buffer(-buffer)  # 安全：直接覆盖几何列
    merged_gdf["area"] = merged_gdf.geometry.area

    # 筛选面积 ≥ x 的要素
    merged_gdf['geometry'] = merged_gdf['geometry'].apply(
        lambda geom: geom.simplify(tolerance=2)  # 示例容差10米
    )
    merged_gdf = merged_gdf[merged_gdf["area"] >= 2000]  # x 为面积阈值（单位需与坐标系一致）
    merged_gdf.to_file(out_path, driver="GeoJSON")


from shapely import is_valid
import geopandas as gpd
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon


def nnunet2(in_path, out_path, buffer=2, min_area=2000):
    # 1. 读取数据并修复几何
    gdf = gpd.read_file(in_path)
    gdf.geometry = gdf.geometry.apply(
        lambda geom: make_valid(geom) if not is_valid(geom) else geom
    )

    # 2. 创建缓冲区并计算原始面积
    gdf['buffered'] = gdf.geometry.buffer(buffer)
    gdf['orig_area'] = gdf.geometry.area  # 保存原始面积

    # 3. 按分类溶解几何（保留最大原始面积属性）
    dissolved = gdf.dissolve(
        by='classification',
        aggfunc={'orig_area': 'max'},
        as_index=False
    )

    # 4. 计算融合后几何的实际面积
    dissolved['fused_area'] = dissolved.geometry.area

    # 5. 面积筛选和拓扑修复
    result = dissolved[dissolved['fused_area'] >= min_area].copy()
    result.geometry = result.geometry.apply(
        lambda geom: geom.buffer(0) if not geom.is_valid else geom
    )

    # 6. 过滤空几何并输出
    result = result[~result.is_empty]
    result.to_file(out_path, driver="GeoJSON")


import geopandas as gpd
from shapely.ops import unary_union
from shapely.validation import make_valid
import os


def nnunet3(in_path, out_path, buffer=2, min_area=2000):
    # 1. 读取数据并修复几何有效性
    gdf = gpd.read_file(in_path)

    # 修复无效几何（关键步骤！）
    gdf.geometry = gdf.geometry.apply(
        lambda geom: make_valid(geom) if not geom.is_valid else geom
    )

    # 2. 确保使用投影坐标系（避免缓冲区距离失真）
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=3857)  # Web墨卡托投影[7](@ref)

    # 3. 添加原始面积列（用于后续属性继承）
    gdf['orig_area'] = gdf.geometry.area

    # 4. 创建缓冲区并合并几何
    gdf['buffered'] = gdf.geometry.buffer(buffer)
    merged_geom = unary_union(gdf['buffered'].tolist())

    # 5. 创建临时合并图层
    merged_gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=gdf.crs)

    # 6. 空间连接：关联原始属性（优化查询性能）
    merged_with_attr = gpd.sjoin(
        merged_gdf,
        gdf[['geometry', 'classification', 'orig_area']],  # 仅选择必要列
        how='left',
        predicate='intersects'
    )

    # 7. 按原始面积继承最大分类属性
    if not merged_with_attr.empty:
        # 分组统计每个分类的总面积
        class_areas = merged_with_attr.groupby('classification')['orig_area'].sum()
        # 选择总面积最大的分类
        final_classification = class_areas.idxmax()

        # 8. 创建结果GeoDataFrame
        result_gdf = gpd.GeoDataFrame({
            'geometry': [merged_geom],
            'classification': [final_classification],
            'total_area': [merged_geom.area]
        }, crs=gdf.crs)

        # 9. 面积筛选和拓扑修复
        result_gdf = result_gdf[result_gdf.total_area >= min_area]
        result_gdf.geometry = result_gdf.geometry.apply(
            lambda geom: make_valid(geom.buffer(0)) if not geom.is_valid else geom
        )

        # 10. 保存结果（自动创建目录）
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_gdf.to_file(out_path, driver='GeoJSON')
    else:
        print("警告：空间连接未匹配到任何要素")


import geopandas as gpd
import json
from shapely.ops import unary_union

import geopandas as gpd
import json
from shapely.ops import unary_union


def merge_seg_detection(
        seg_path,
        detect_path,
        output_path,
        area_ratio_threshold=0.15,
        buffer_size=2,
        min_area=2000
):
    # ===== 1. 数据读取与预处理 =====
    seg_gdf = gpd.read_file(seg_path)
    detect_gdf = gpd.read_file(detect_path)

    def parse_classification(s):
        try:
            return json.loads(s).get('name')
        except:
            return 'prostate'

    detect_gdf = detect_gdf[detect_gdf['classification'].apply(parse_classification) == 'prostate']

    if seg_gdf.crs != detect_gdf.crs:
        detect_gdf = detect_gdf.to_crs(seg_gdf.crs)

    # ===== 2. 几何合并（不处理分类） =====
    def buffer_merge(gdf, buffer, min_area):
        buffered = gdf.geometry.buffer(buffer)
        merged_geom = unary_union(buffered.tolist())
        merged_polys = list(merged_geom.geoms) if merged_geom.geom_type == 'MultiPolygon' else [merged_geom]
        merged_gdf = gpd.GeoDataFrame(geometry=merged_polys, crs=gdf.crs)
        merged_gdf.geometry = merged_gdf.geometry.buffer(-buffer)
        merged_gdf['area'] = merged_gdf.geometry.area
        return merged_gdf[merged_gdf['area'] >= min_area]

    merged_seg = buffer_merge(seg_gdf, buffer_size, min_area)
    merged_seg['classification'] = None  # 初始化分类列

    # ===== 3. Detect数据赋值分类 =====
    intersected = gpd.sjoin(
        detect_gdf[["geometry", "classification"]],
        merged_seg,
        how="inner", predicate="intersects", lsuffix="d", rsuffix="g"
    )

    def calc_area_ratio(row):
        seg_geom = merged_seg.loc[row["index_g"], 'geometry']
        intersection = row['geometry'].intersection(seg_geom)
        return intersection.area / seg_geom.area if not intersection.is_empty else 0.0

    intersected["area_ratio"] = intersected.apply(calc_area_ratio, axis=1)
    valid_intersections = intersected[intersected["area_ratio"] > area_ratio_threshold]

    update_idx = valid_intersections["index_g"]
    merged_seg.loc[update_idx, "classification"] = valid_intersections["classification_d"].values

    # ===== 4. 未覆盖区域：继承原始最大相交分类 =====
    from shapely.validation import make_valid

    # 修复几何有效性函数
    def fix_invalid_geom(geom):
        if not geom.is_valid:
            # 方法1: 使用 buffer(0) 修复简单拓扑错误
            fixed_geom = geom.buffer(0)
            # 方法2: 复杂错误使用 make_valid（Shapely 2.0+）
            try:
                from shapely.validation import make_valid
                fixed_geom = make_valid(geom)
            except ImportError:
                pass
            return fixed_geom if fixed_geom.is_valid else geom
        return geom

    # 修复 merged_seg 和 seg_gdf 中的几何
    merged_seg['geometry'] = merged_seg['geometry'].apply(fix_invalid_geom)
    seg_gdf['geometry'] = seg_gdf['geometry'].apply(fix_invalid_geom)
    uncovered_mask = merged_seg['classification'].isnull()
    seg_gdf_sindex = seg_gdf.sindex  # 创建空间索引[7](@ref)

    for idx, row in merged_seg[uncovered_mask].iterrows():
        # 通过空间索引快速定位候选几何
        candidate_idx = list(seg_gdf_sindex.intersection(row.geometry.bounds))
        candidates = seg_gdf.iloc[candidate_idx]

        # 精确筛选实际相交的几何
        intersecting = candidates[candidates.intersects(row.geometry)]

        if not intersecting.empty:
            # 选择面积最大的原始几何并继承其分类
            largest = intersecting.loc[intersecting.geometry.area.idxmax()]
            merged_seg.at[idx, 'classification'] = largest['classification']
        else:
            # 无相交时的兜底方案
            global_largest = seg_gdf.loc[seg_gdf.geometry.area.idxmax(), 'classification']
            merged_seg.at[idx, 'classification'] = global_largest

    # ===== 5. 输出结果 =====
    merged_seg['geometry'] = merged_seg.geometry.simplify(buffer_size)
    merged_seg.to_file(output_path, driver="GeoJSON")


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/segment', help='patch directory')
parser.add_argument('--input_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/segment/YNZL映射/contour', help='patch directory')
parser.add_argument('--point_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/segment/points', help='patch directory')
parser.add_argument('--output_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/segment/YNZL映射/result', help='output directory')
parser.add_argument('--detect_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/segment/YNZL映射/yolo', help='output directory')
parser.add_argument('--label_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/segment/YNZL映射/result', help='output directory')
parser.add_argument('--patch_size', type=int, default=1024, help='patch size')
parser.add_argument('--ihc_ext', type=str, default='-CK', help='patch size')
if __name__ == "__main__":
    args = parser.parse_args()
    input_dir = args.input_dir
    # geojson2txt(args.point_dir)
    # json2txt(args.point_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.label_dir, exist_ok=True)
    # for file in os.listdir(input_dir):
    #     if not os.path.exists(os.path.join(input_dir, file)):
    #         logger.info(f'{file} label not exists, skip')
    #         continue
    #     processor = GeoJSONProcessor(
    #         input_path=os.path.join(input_dir, file),
    #         output_path=os.path.join(args.output_dir, file.replace(args.ihc_ext, '')),
    #         points_dir=args.point_dir,
    #         ihc_ext=args.ihc_ext,
    #         simplify_tolerance=0.01
    #     )
    #     processor.execute(patch_size=args.patch_size)
    # for file in os.listdir(args.output_dir):
    #     if '-CK' in file or '-new' in file:
    #         continue
    #     remove_intersects(
    #         detect_path=os.path.join(args.detect_dir, file.replace('有癌', '')),
    #         seg_path=os.path.join(args.output_dir, file),
    #         output_path=os.path.join(args.output_dir, file.replace('.geojson', '-new.geojson')),
    #     )
    # files = os.listdir(args.output_dir)
    # files = [file for file in files if not file.endswith('-CK.geojson') and not file.endswith('-cancer.geojson')]
    # for file in files:
    #     detect_file = os.path.join(args.detect_dir, file)
    #     if not os.path.isfile(detect_file):
    #         logger.info(f'detect file {file} not exists')
    #         continue
    #     seg_file = os.path.join(args.output_dir, file)
    #     label_file = os.path.join(args.label_dir, file)
    #     segment_label(seg_file, detect_file, label_file)
    #     logger.info(f'{file} 合并结果已经保存！！')
    files = os.listdir(args.input_dir)
    for file in files:
        start = time.time()
        print(f'start{file}')
        in_path = os.path.join(args.input_dir, file)
        detect_path = os.path.join(args.detect_dir, file)
        out_path = os.path.join(args.output_dir, file)
        nnunet(in_path, out_path, 3)
        merge_seg_detection(in_path, detect_path, out_path, 0.2, 2)
        print(time.time() - start)
