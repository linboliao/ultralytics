import argparse
import json
import multiprocessing
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import ImageDraw
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
from segmentation.wsi import WSIOperator
from loguru import logger
from tqdm import tqdm
import traceback


# ====================== 基类设计 ======================
class YOLOConverter:
    """YOLO格式转换器的抽象基类"""
    SUPPORTED_FORMATS = ['.svs', '.kfb', '.tif', '.tiff', '.sdpc', '.ndpi', '.mrxs']
    DATASET_RATIO = [0.85, 0.15, 0]

    def __init__(self, data_root, slide_dir, label_dir, output_dir, patch_size=512, patch_level=0):
        """
        初始化转换器
        :param data_root: 数据根目录
        :param slide_dir: 切片文件目录
        :param label_dir: 标注文件目录
        :param output_dir: 输出目录
        :param patch_size: 分块大小
        :param patch_level: 分块级别
        """
        self.data_root = Path(data_root)
        self.slide_dir = Path(slide_dir) if slide_dir else self.data_root / 'slides'
        self.label_dir = Path(label_dir) if label_dir else self.data_root / 'geojson'
        self.output_dir = Path(output_dir) if output_dir else self.data_root / 'dataset'
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.slide_files = []

        # 创建输出目录结构
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "val").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)

    def scan_slide_files(self):
        """扫描支持的切片文件"""
        self.slide_files = [f for f in os.listdir(self.slide_dir) if any(f.endswith(ext) for ext in self.SUPPORTED_FORMATS)]
        logger.info(f"找到 {len(self.slide_files)} 个切片文件")

    def split_dataset(self, seed=42):
        """随机划分数据集"""
        random.seed(seed)
        random.shuffle(self.slide_files)
        total = len(self.slide_files)
        num_train = int(total * self.DATASET_RATIO[0])
        num_val = int(total * self.DATASET_RATIO[1])

        return {
            'train': self.slide_files[:num_train],
            'val': self.slide_files[num_train:num_train + num_val],
            'test': self.slide_files[num_train + num_val:]
        }

    def process_slide(self, slide_name, dataset_type="train"):
        """处理单个切片（需子类实现）"""
        raise NotImplementedError("子类必须实现此方法")

    def process_dataset(self, dataset_type, slides, num_workers=None):
        """并行处理数据集"""
        num_workers = num_workers or multiprocessing.cpu_count()
        logger.info(f"启动并行处理 | 进程数: {num_workers}")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    self.process_slide,
                    slide,
                    dataset_type
                ) for slide in slides
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"处理失败: {str(e)}")

    def run(self, num_workers=None):
        """执行转换流程"""
        self.scan_slide_files()
        dataset_split = self.split_dataset()

        for dataset_type, slides in dataset_split.items():
            logger.info(f"处理{dataset_type}集 | 切片数量: {len(slides)}")
            self.process_dataset(dataset_type, slides, num_workers)


# ====================== GeoJSON转换器 ======================
class GeoJSONYOLOConverter(YOLOConverter):
    """处理GeoJSON标注的YOLO转换器"""
    # CLASS_MAPPING = {
    #     'benign': 0, 'tangential_benign': 0, 'gland': 0, 'stroma': 0,
    #     'pattern3': 1, 'pattern4': 1, 'PIN': 1, 'malignant': 1, 'tangential_malignant': 1,
    #     'unknown': 2,
    #     'artefact': 3
    # }
    # # CLASS_MAPPING = {'pattern3': 0,
    #                  'pattern4': 1,
    #                  'benign': 2,
    #                  'tangential_benign': 3,
    #                  'tangential_malignant': 4,
    #                  'unknown': 5,
    #                  'PIN': 6,
    #                  'artefact': 7}

    # GROUP_MAPPING = {
    #     'p3': 'pattern3', 'P3': 'pattern3', 'p4': 'pattern4', 'P4': 'pattern4',
    #     'b': 'benign', 'B': 'benign', 'tangential_benign': 'tangential_benign',
    #     'tangential_malignant': 'tangential_malignant', 'pattern3': 'pattern3',
    #     'pattern4': 'pattern4', 'benign': 'benign', 'unknown': 'unknown',
    #     'PIN': 'PIN', 'artefact': 'artefact', 'artifact': 'artefact',
    #     'Artefact': 'artefact', 't': 'tangential_benign', 'tangential': 'tangential_benign'
    # }

    CLASS_MAPPING = {
        'Negative': 0,
        'Positive': 1,
        'Other': 2,
    }
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 保存类别映射文件
        with open(self.output_dir / "classes.txt", 'w') as f:
            for name, clazz in self.CLASS_MAPPING.items():
                f.write(f"{clazz} {name}\n")

    def parse_geojson(self, geojson_path):
        """解析GeoJSON文件"""
        annotations = []
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)

        for feature in geojson_data['features']:
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]
                class_name = feature.get('properties', {}).get('classification', {}).get('name', 'Positive')
                # class_name = self.GROUP_MAPPING.get(class_name, '')
                if class_name and class_name != 'Other':
                    annotations.append({
                        'polygon': make_valid(Polygon(coords)),
                        'class_name': class_name
                    })
        return annotations

    def process_slide(self, slide_name, dataset_type="train"):
        """处理单个切片"""
        slide_path = self.slide_dir / slide_name
        slide_id = Path(slide_path).stem
        geojson_path = self.label_dir / f"{slide_id}.geojson"
        image_dir = self.output_dir / dataset_type / 'images'
        label_dir = self.output_dir / dataset_type / 'labels'
        contour_dir = self.output_dir / dataset_type / 'contours'
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(contour_dir, exist_ok=True)

        # 读取切片
        try:
            slide = WSIOperator(str(slide_path))
            width, height = slide.level_dimensions[self.patch_level]
            times = 2 ** self.patch_level
            logger.info(f"载入切片 {slide_name} 成功 | 级别: {self.patch_level} | 尺寸: {width}x{height}")
        except Exception as e:
            logger.error(f"打开切片文件失败: {str(e)}")
            return

        # 解析标注
        if not os.path.exists(geojson_path):
            logger.info(f'{geojson_path}标注缺失，跳过！')
            return
        annotations = self.parse_geojson(geojson_path)

        # 处理每个patch
        patch_count = 0
        skipped_count = 0

        for y in range(0, height, self.patch_size):
            for x in range(0, width, self.patch_size):
                w = min(self.patch_size, width - x)
                h = min(self.patch_size, height - y)

                # 计算实际坐标范围
                actual_x = x * times
                actual_y = y * times
                actual_w = w * times
                actual_h = h * times
                patch_box = box(actual_x, actual_y, actual_x + actual_w, actual_y + actual_h)

                patch_img = slide.read_region((x, y), self.patch_level, (self.patch_size, self.patch_size))

                if not patch_img:
                    logger.info(f'{slide_id} patch {x} {y} 为背景，跳过！')
                    skipped_count += 1
                    continue

                rgb_img = patch_img.convert("RGB")
                label_lines = []
                has_labels = False

                # # 创建轮廓图像
                # contour_img = rgb_img.copy()
                # draw = ImageDraw.Draw(contour_img)

                # 处理标注
                for anno in annotations:
                    if patch_box.intersects(anno['polygon']):
                        intersection = patch_box.intersection(anno['polygon'])

                        if not intersection.is_empty:
                            has_labels = True
                            if intersection.geom_type == 'Polygon':
                                polygons = [intersection]
                            elif intersection.geom_type == 'MultiPolygon':
                                polygons = intersection.geoms
                            else:
                                continue

                            for poly in polygons:
                                exterior = list(poly.exterior.coords)
                                normalized_points = []

                                for pt in exterior:
                                    local_x = pt[0] - actual_x
                                    local_y = pt[1] - actual_y
                                    norm_x = max(0.0, min(1.0, local_x / actual_w))
                                    norm_y = max(0.0, min(1.0, local_y / actual_h))
                                    normalized_points.extend([norm_x, norm_y])

                                clazz = self.CLASS_MAPPING.get(anno['class_name'], 0)
                                # clazz = 0
                                points_str = " ".join(f"{p:.6f}" for p in normalized_points)
                                label_lines.append(f"{clazz} {points_str}")

                                # # 绘制轮廓
                                # draw.line(exterior, fill="red", width=5)
                                # if len(exterior) > 1:
                                #     draw.line([exterior[-1], exterior[0]], fill="red", width=2)

                # 保存结果
                if has_labels or random.random() < 0.25:
                    img_name = f"{slide_id}_{x}_{y}.png"
                    rgb_img.save(image_dir / img_name)

                    txt_name = f"{slide_id}_{x}_{y}.txt"
                    with open(label_dir / txt_name, 'w') as f:
                        f.write("\n".join(label_lines))

                    # contour_img.save(contour_dir / img_name)
                    patch_count += 1
                else:
                    skipped_count += 1

        logger.info(f"处理完成! 有效patch: {patch_count} | 跳过无标签patch: {skipped_count}")


# ====================== KVJSON转换器 ======================
class KVYOLOConverter(YOLOConverter):
    """处理KV格式JSON标注的YOLO转换器"""
    LABELS = {4278255615: 0, 4294901760: 1, 4278190080: 2, 4278251008: 1}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_slide(self, slide_name, dataset_type="train"):
        """处理单个切片"""
        slide_path = self.slide_dir / slide_name
        slide_id = Path(slide_path).stem
        # json_path = self.label_dir / f"{slide_id}.json"
        json_path = self.label_dir / slide_name.replace('.', '_') / "Annotations" / "1.json"
        if not os.path.exists(json_path):
            logger.info(f'{slide_name}标注缺失，跳过！')
            return
        image_dir = self.output_dir / dataset_type / 'images'
        label_dir = self.output_dir / dataset_type / 'labels'
        contour_dir = self.output_dir / dataset_type / 'contours'
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(contour_dir, exist_ok=True)

        # 读取切片
        try:
            slide = WSIOperator(str(slide_path))
            width, height = slide.level_dimensions[self.patch_level]
            times = 2 ** self.patch_level
            logger.info(f"载入切片 {slide_name} 成功 | 级别: {self.patch_level} | 尺寸: {width}x{height}")
        except Exception as e:
            logger.error(f"打开切片文件失败: {str(e)}")
            return

        # 解析标注
        with open(json_path) as f:
            contours = json.load(f)

        # 处理每个patch
        patch_count = 0

        for y in tqdm(range(0, height, self.patch_size), desc=f"处理 {slide_id}"):
            for x in range(0, width, self.patch_size):
                w = min(self.patch_size, width - x)
                h = min(self.patch_size, height - y)
                # 保存图像块
                patch = slide.read_region((x, y), self.patch_level, (w, h), True)
                if not patch:
                    logger.info(f'{slide_id} patch {x} {y} 为背景，跳过！')
                    continue
                patch = patch.convert("RGB")

                # 计算实际坐标范围
                actual_x = x * times
                actual_y = y * times
                actual_w = w * times
                actual_h = h * times
                patch_box = box(actual_x, actual_y, actual_x + actual_w, actual_y + actual_h)

                label_lines = []
                has_labels = False

                # 生成标签
                for contour in contours:
                    points = [(p["x"], p["y"]) for p in contour.get("points", [])]
                    if not points or len(points) < 3:
                        continue

                    # 闭合多边形
                    points.append(points[0])
                    polygon = Polygon(points)
                    polygon = make_valid(polygon)

                    # 处理交集
                    intersection = polygon.intersection(patch_box)
                    if not intersection.is_empty:
                        has_labels = True
                        polygons = [intersection] if intersection.geom_type == 'Polygon' else list(intersection.geoms)

                        for poly in polygons:
                            exterior = list(poly.exterior.coords)
                            normalized_points = []

                            for pt in exterior:
                                local_x = pt[0] - actual_x
                                local_y = pt[1] - actual_y
                                norm_x = max(0.0, min(1.0, local_x / actual_w))
                                norm_y = max(0.0, min(1.0, local_y / actual_h))
                                normalized_points.extend([norm_x, norm_y])
                            color = contours.get('color')
                            if color == 4294967040:
                                logger.info(f'神经节侵犯标签，跳过！')
                                continue
                            clazz = self.LABELS.get(contours.get('color'), 0)
                            points_str = " ".join(f"{p:.6f}" for p in normalized_points)
                            label_lines.append(f"{clazz} {points_str}")

                img_name = f"{slide_id}_{x}_{y}.jpg"
                patch.save(image_dir / img_name)

                # 保存结果
                if has_labels or random.random() < 0.25:
                    txt_name = f"{slide_id}_{x}_{y}.txt"
                    with open(label_dir / txt_name, 'w') as f:
                        f.write("\n".join(label_lines))

                    patch_count += 1

        logger.info(f"生成 {patch_count} 个patch | 轮廓数量: {len(contours)}")


class MultiMagGeo2YOLO(GeoJSONYOLOConverter):
    def process_slide(self, slide_name, dataset_type="train"):
        """处理单个切片"""
        slide_path = self.slide_dir / slide_name
        slide_id = Path(slide_path).stem
        geojson_path = self.label_dir / f"{slide_id}.geojson"
        image_dir = self.output_dir / dataset_type / 'images'
        low_image_dir = self.output_dir / dataset_type / 'images_low'
        high_image_dir = self.output_dir / dataset_type / 'images_high'
        label_dir = self.output_dir / dataset_type / 'labels'
        contour_dir = self.output_dir / dataset_type / 'contours'
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(high_image_dir, exist_ok=True)
        os.makedirs(low_image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(contour_dir, exist_ok=True)

        # 读取切片
        try:
            slide = WSIOperator(str(slide_path))
            width, height = slide.level_dimensions[self.patch_level]
            times = 2 ** self.patch_level
            logger.info(f"载入切片成功 | 级别: {self.patch_level} | 尺寸: {width}x{height}")
        except Exception as e:
            logger.error(f"打开切片文件失败: {str(e)}")
            return

        # 解析标注
        if not os.path.exists(geojson_path):
            logger.info(f'{geojson_path}标注缺失，跳过！')
            return
        annotations = self.parse_geojson(geojson_path)

        # 处理每个patch
        patch_count = 0
        skipped_count = 0

        for y in range(0, height, self.patch_size):
            for x in range(0, width, self.patch_size):
                w = min(self.patch_size, width - x)
                h = min(self.patch_size, height - y)

                # 计算实际坐标范围
                actual_x = x * times
                actual_y = y * times
                actual_w = w * times
                actual_h = h * times
                patch_box = box(actual_x, actual_y, actual_x + actual_w, actual_y + actual_h)

                patch_img = slide.read_region((x, y), self.patch_level, (self.patch_size, self.patch_size), True)

                if not patch_img:
                    logger.info(f'{slide_id} patch {x} {y} 为背景，跳过！')
                    skipped_count += 1
                    continue

                rgb_img = patch_img.convert("RGB")
                label_lines = []
                has_labels = False

                # 创建轮廓图像
                contour_img = rgb_img.copy()
                draw = ImageDraw.Draw(contour_img)

                # 处理标注
                for anno in annotations:
                    if patch_box.intersects(anno['polygon']):
                        intersection = patch_box.intersection(anno['polygon'])

                        if not intersection.is_empty:
                            has_labels = True
                            if intersection.geom_type == 'Polygon':
                                polygons = [intersection]
                            elif intersection.geom_type == 'MultiPolygon':
                                polygons = intersection.geoms
                            else:
                                continue

                            for poly in polygons:
                                exterior = list(poly.exterior.coords)
                                normalized_points = []

                                for pt in exterior:
                                    local_x = pt[0] - actual_x
                                    local_y = pt[1] - actual_y
                                    norm_x = max(0.0, min(1.0, local_x / actual_w))
                                    norm_y = max(0.0, min(1.0, local_y / actual_h))
                                    normalized_points.extend([norm_x, norm_y])

                                clazz = self.CLASS_MAPPING.get(anno['class_name'], 3)
                                points_str = " ".join(f"{p:.6f}" for p in normalized_points)
                                label_lines.append(f"{clazz} {points_str}")

                                # 绘制轮廓
                                draw.line(exterior, fill="red", width=5)
                                if len(exterior) > 1:
                                    draw.line([exterior[-1], exterior[0]], fill="red", width=2)

                # 保存结果
                if has_labels:
                    img_name = f"{slide_id}_{x}_{y}.png"
                    rgb_img.save(image_dir / img_name)
                    img_low = slide.read_region((w + int(self.patch_size * 0.25), h + int(self.patch_size * 0.25)), 0, (int(self.patch_size * 0.5), int(self.patch_size * 0.5)))
                    img_high = slide.read_region((w - int(self.patch_size * 0.25), h - int(self.patch_size * 0.25)), 0, (int(self.patch_size * 1.5), int(self.patch_size * 1.5)))
                    img_low.save(low_image_dir / img_name)
                    img_high.save(high_image_dir / img_name)
                    txt_name = f"{slide_id}_{x}_{y}.txt"
                    with open(label_dir / txt_name, 'w') as f:
                        f.write("\n".join(label_lines))

                    contour_img.save(contour_dir / img_name)
                    patch_count += 1
                else:
                    skipped_count += 1

        logger.info(f"处理完成! 有效patch: {patch_count} | 跳过无标签patch: {skipped_count}")


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Detection/0425')
parser.add_argument('--slide_dir', type=str, default='')
parser.add_argument('--label_dir', type=str, default='')
parser.add_argument('--output_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Detection/0425/dataset/0611')
parser.add_argument('--patch_size', type=int, default=2048)
parser.add_argument('--patch_level', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=40)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

if __name__ == '__main__':
    converter = GeoJSONYOLOConverter(data_root=args.data_root, slide_dir=args.slide_dir, label_dir=args.label_dir, output_dir=args.output_dir, patch_size=args.patch_size, patch_level=args.patch_level)
    # converter = MultiMagGeo2YOLO(data_root=args.data_root, slide_dir=args.slide_dir, label_dir=args.label_dir, output_dir=args.output_dir, patch_size=args.patch_size, patch_level=args.patch_level)
    # converter = KVYOLOConverter(data_root=args.data_root, slide_dir=args.slide_dir, label_dir=args.label_dir, output_dir=args.output_dir, patch_size=args.patch_size, patch_level=args.patch_level)
    converter.run(num_workers=args.num_workers)
