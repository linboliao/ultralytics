import argparse
import json
import multiprocessing
import os
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from pathlib import Path
from PIL import ImageDraw
from rich import traceback
from tqdm import tqdm
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
from segmentation.wsi import WSIOperator

SUPPORTED_FORMATS = ['.svs', '.kfb', '.tif', '.tiff', '.sdpc', '.ndpi', '.mrxs']
DATASET_RATIO = [0.7, 0.3, 0]


def geo2yolo(slide_path, geojson_path, output_dir, patch_size=512):
    """
    处理SVS文件为patch，转换GeoJSON标注为YOLO多边形格式，并在图像上绘制轮廓

    参数:
    slide_path: 切片文件路径
    geojson_path: GeoJSON标注文件路径
    output_dir: 输出目录
    patch_size: patch尺寸 (默认512)
    """
    # === 1. 初始化日志系统 ===
    log_path = Path(output_dir) / "processing.log"
    logger.add(log_path)
    logger.info(f"开始处理: {Path(slide_path).name}")

    # === 2. 创建输出目录 ===
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    contours_dir = Path(output_dir) / "contours"

    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    contours_dir.mkdir(exist_ok=True)

    slide_id = Path(slide_path).stem

    # === 3. 读取SVS文件 ===
    try:
        slide = WSIOperator(slide_path)
        width, height = slide.level_dimensions[0]
        logger.info(f"载入SVS成功 | 尺寸: {width}x{height}")
    except Exception as e:
        logger.error(f"打开SVS文件失败: {str(e)}")
        return
    # === 4. 类别映射配置 ===
    CLASS_MAPPING = {
        'benign': 0, 'tangential_benign': 0, 'gland': 0, 'stroma': 0,
        'pattern3': 1, 'pattern4': 1, 'PIN': 1, 'malignant': 1, 'tangential_malignant': 1,
        'blood_vessel': 2,
        'artefact': 3,
    }
    group_dict = {'p3': 'pattern3',
                  'P3': 'pattern3',
                  'p4': 'pattern4',
                  'P4': 'pattern4',
                  'b': 'benign',
                  'B': 'benign',
                  'tangential_benign': 'tangential_benign',
                  'tangential_malignant': 'tangential_malignant',
                  'pattern3': 'pattern3',
                  'pattern4': 'pattern4',
                  'benign': 'benign',
                  'unknown': 'unknown',
                  'PIN': 'PIN',
                  'artefact': 'artefact',
                  'artifact': 'artefact',
                  'Artefact': 'artefact',
                  't': 'tangential_benign',
                  'tangential': 'tangential_benign'}

    # === 5. 解析GeoJSON标注 ===
    annotations = []
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)

        for feature in geojson_data['features']:
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]
                class_name = feature.get('properties', {}).get('group', '')
                annotations.append({
                    'polygon': Polygon(coords),
                    'class_name': group_dict.get(class_name, '')
                })
    except Exception as e:
        logger.error(f"解析GeoJSON失败: {str(e)}")
        return

    # === 6. 处理每个patch ===
    patch_count = 0
    skipped_count = 0

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch_box = box(x, y, x + patch_size, y + patch_size)
            patch_img = slide.read_region((x, y), 0, (patch_size, patch_size), True)
            if not patch_img:
                logger.info(f'{slide_id} patch {x} {y} 为背景，跳过！')
                continue
            rgb_img = patch_img.convert("RGB")

            label_lines = []
            has_labels = False
            contour_img = rgb_img.copy()  # 创建轮廓图像副本
            draw = ImageDraw.Draw(contour_img)

            # 处理当前patch的标注
            for anno in annotations:
                if patch_box.intersects(anno['polygon']):
                    intersection = patch_box.intersection(anno['polygon'])

                    if not intersection.is_empty:
                        has_labels = True

                        # === 关键修改：多边形顶点转换 ===
                        # 获取多边形顶点（全局坐标）
                        if intersection.geom_type == 'Polygon':
                            polygons = [intersection]
                        elif intersection.geom_type == 'MultiPolygon':
                            polygons = intersection.geoms
                        else:
                            continue

                        for poly in polygons:
                            # 获取多边形外边界
                            exterior = list(poly.exterior.coords)

                            # 转换为局部坐标并归一化
                            normalized_points = []
                            for pt in exterior:
                                # 转换为相对于当前patch的局部坐标
                                local_x = pt[0] - x
                                local_y = pt[1] - y

                                # 归一化到[0,1]范围
                                norm_x = local_x / patch_size
                                norm_y = local_y / patch_size

                                # 确保在[0,1]范围内
                                norm_x = max(0.0, min(1.0, norm_x))
                                norm_y = max(0.0, min(1.0, norm_y))

                                normalized_points.extend([norm_x, norm_y])

                            # 获取类别ID
                            clazz = CLASS_MAPPING.get(anno['class_name'], 3)
                            if anno['class_name'] not in CLASS_MAPPING.keys():
                                logger.info(f'缺失 ')

                            # 格式: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
                            points_str = " ".join([f"{p:.6f}" for p in normalized_points])
                            label_lines.append(f"{clazz} {points_str}")

                            # === 绘制轮廓 ===
                            # 使用局部坐标绘制（非归一化坐标）
                            draw.line(exterior, fill="red", width=5)
                            if len(exterior) > 1:
                                draw.line([exterior[-1], exterior[0]], fill="red", width=2)

            # === 7. 保存逻辑 ===
            if not has_labels:
                skipped_count += 1
                continue

            # 保存原始图像
            img_name = f"{slide_id}_{x}_{y}.png"
            img_path = images_dir / img_name
            rgb_img.save(img_path)

            # 保存YOLO多边形标签
            txt_name = f"{slide_id}_{x}_{y}.txt"
            txt_path = labels_dir / txt_name
            with open(txt_path, 'w') as f:
                f.write("\n".join(label_lines))

            # 保存轮廓图像
            contour_path = contours_dir / img_name
            contour_img.save(contour_path)

            patch_count += 1
            logger.info(f"保存patch: {img_name} | 多边形数量: {len(label_lines)}")

    # === 8. 保存类别映射 ===
    with open(Path(output_dir) / "classes.txt", 'w') as f:
        for name, id in CLASS_MAPPING.items():
            f.write(f"{id} {name}\n")
        f.write("3 other")

    logger.info(f"处理完成! 有效patch: {patch_count} | 跳过无标签patch: {skipped_count}")
    logger.info(f"日志文件: {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/zenodo', help='根目录')
    parser.add_argument('--slide_dir', type=str, default='', help='SVS文件目录')
    parser.add_argument('--label_dir', type=str, default='', help='GeoJSON标注目录')
    parser.add_argument('--output_dir', type=str, default='', help='输出目录')
    parser.add_argument('--patch_size', type=int, default=512, help='分块大小')
    parser.add_argument('--patch_level', type=int, default=0, help='分块级别')
    parser.add_argument('--num_workers', type=int, default=None, help='并行进程数（默认：CPU核心数）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()
    slide_dir = Path(args.slide_dir) if args.slide_dir else Path(args.data_root) / 'slides'
    label_dir = Path(args.label_dir) if args.label_dir else Path(args.data_root) / 'geojson'
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.data_root) / 'dataset-1'

    # 设置随机种子
    random.seed(args.seed)

    # 创建输出目录结构
    output_root = Path(output_dir)
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    test_dir = output_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 扫描所有支持的slide文件
    slide_dir = Path(slide_dir)
    slide_files = os.listdir(slide_dir)
    # slide_files = []
    # for ext in SUPPORTED_FORMATS:
    #     slide_files.extend(list(slide_dir.glob(f'*.{ext}')))

    if not slide_files:
        logger.info(f"⚠️ 未找到支持的slide文件: {slide_dir}")
        return

    logger.info(f"🔍 找到 {len(slide_files)} 个slide文件")

    # 随机划分数据集
    random.shuffle(slide_files)
    total = len(slide_files)
    num_train = int(total * DATASET_RATIO[0])
    num_val = int(total * DATASET_RATIO[1])
    num_test = total - num_train - num_val

    train_slides = slide_files[:num_train]
    val_slides = slide_files[num_train:num_train + num_val]
    test_slides = slide_files[num_train + num_val:]

    logger.info(f"📊 数据集划分: train={len(train_slides)} val={len(val_slides)} test={len(test_slides)}")
    # 配置并行处理
    num_workers = args.num_workers or multiprocessing.cpu_count()
    logger.info(f"⚡ 启动并行处理 | 进程数: {num_workers}")

    def get_icon(dataset_type):
        """返回数据集对应的图标"""
        icons = {"训练": "🔧", "验证": "📈", "测试": "🧪"}
        return icons.get(dataset_type, "▶️")

    def process_dataset(dataset_type, slides, slide_dir, label_dir, output_dir, patch_size, num_workers):
        """通用数据集处理函数"""
        try:
            logger.info(f"\n{get_icon(dataset_type)} 处理{dataset_type}集...")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        geo2yolo,
                        slide_dir / slide,
                        label_dir / slide.replace('.svs', '.geojson'),
                        output_dir,
                        patch_size
                    ) for slide in slides
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        traceback.print_exc()
        except Exception:
            pass


    # 调用示例
    process_dataset("训练", train_slides, slide_dir, label_dir, train_dir, args.patch_size, num_workers)
    process_dataset("验证", val_slides, slide_dir, label_dir, val_dir, args.patch_size, num_workers)
    process_dataset("测试", test_slides, slide_dir, label_dir, test_dir, args.patch_size, num_workers)


patch_level = 3


def process_svs_file(svs_path, json_path, output_dir, patch_size=2048):
    """
    处理单个SVS文件及其对应的JSON标注
    :param svs_path: SVS文件路径
    :param json_path: JSON标注文件路径
    :param output_dir: 输出目录
    :param patch_size: 图像块大小
    """
    # 创建输出子目录

    base_name = os.path.splitext(os.path.basename(svs_path))[0]
    img_output_dir = os.path.join(output_dir, "images")
    label_output_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    # 加载SVS文件
    slide = WSIOperator(svs_path)
    width, height = slide.level_dimensions[patch_level]
    times = 2 ** patch_level

    # 加载JSON标注（多个轮廓）
    with open(json_path) as f:
        contours = json.load(f)  # 直接加载轮廓列表[1,5](@ref)

    # 处理每个图像块
    for y in tqdm(range(0, height - patch_size, patch_size), desc=f"切割 {base_name}"):
        for x in range(0, width - patch_size, patch_size):
            w = min(patch_size, width - x)
            h = min(patch_size, height - y)
            patch_box = box(x * times, y * times, (x + w) * times, (y + h) * times)

            # 保存图像块
            patch = slide.read_region((x, y), patch_level, (w, h)).convert("RGB")
            patch_name = f"{base_name}_{x}_{y}.jpg"
            patch_path = os.path.join(img_output_dir, patch_name)
            patch.save(patch_path)

            # 生成标签文件
            label_path = os.path.join(label_output_dir, patch_name.replace(".jpg", ".txt"))
            generate_yolo_labels(contours, patch_box, label_path, w, h)

    return len(contours)


def generate_yolo_labels(contours, patch_box, label_path, patch_w, patch_h):
    """
    为多个轮廓生成YOLO格式标签
    :param contours: 轮廓列表
    :param patch_box: 当前图像块边界框
    :param label_path: 标签文件路径
    :param patch_w: 图像块宽度
    :param patch_h: 图像块高度
    """
    with open(label_path, "w") as f:
        times = 2 ** patch_level
        for contour in contours:
            # 创建多边形对象 - 直接从points字段获取坐标[1](@ref)
            points = [(p["x"], p["y"]) for p in contour["points"]]
            if len(points) <= 0:
                continue
            points.append(points[0])
            polygon = Polygon(points)

            polygon = make_valid(polygon)

            # 检查交集
            if not polygon.intersects(patch_box):
                continue

            # 获取交集
            intersection = polygon.intersection(patch_box)

            # 处理不同几何类型
            if intersection.geom_type == 'Polygon':
                polygons = [intersection]
            elif intersection.geom_type == 'MultiPolygon':
                polygons = list(intersection.geoms)
            else:
                continue

            for poly in polygons:
                # 使用默认类别ID 0（根据实际需求修改）
                class_id = 0

                # 写入类别ID
                f.write(f"{class_id} ")

                # 转换并写入坐标点
                for point in poly.exterior.coords:
                    local_x = point[0] - patch_box.bounds[0]
                    local_y = point[1] - patch_box.bounds[1]
                    x_norm = min(max(local_x / patch_w / times, 0), 1)
                    y_norm = min(max(local_y / patch_h / times, 0), 1)
                    f.write(f"{x_norm:.6f} {y_norm:.6f} ")

                f.write("\n")  # 每个轮廓单独一行

def kv2yolo():
    slide_dir = rf'/NAS2/Data1/lbliao/Data/MXB/seminal/slides/'
    json_dir = rf'/NAS2/Data1/lbliao/Data/MXB/seminal/geojson/'
    output_dir = rf'/NAS2/Data1/lbliao/Data/MXB/seminal/dataset/2048-3'
    # 扫描所有支持的slide文件
    slide_dir = Path(slide_dir)
    slide_files = os.listdir(slide_dir)

    if not slide_files:
        logger.info(f"⚠️ 未找到支持的slide文件: {slide_dir}")

    logger.info(f"🔍 找到 {len(slide_files)} 个slide文件")

    # 随机划分数据集
    random.shuffle(slide_files)
    total = len(slide_files)
    num_train = int(total * DATASET_RATIO[0])
    num_val = int(total * DATASET_RATIO[1])
    num_test = total - num_train - num_val

    train_slides = slide_files[:num_train]
    val_slides = slide_files[num_train:num_train + num_val]
    test_slides = slide_files[num_train + num_val:]

    logger.info(f"📊 数据集划分: train={len(train_slides)} val={len(val_slides)} test={len(test_slides)}")

    for slide in train_slides:
        tmp = Path(output_dir) / Path('train')
        slide_id, _ = os.path.splitext(slide)
        slide_path = os.path.join(slide_dir, slide)
        json_path = os.path.join(json_dir, f'{slide_id}.json')

        process_svs_file(slide_path, json_path, tmp)
    for slide in val_slides:
        tmp = Path(output_dir) / Path('val')
        slide_id, _ = os.path.splitext(slide)
        slide_path = os.path.join(slide_dir, slide)
        json_path = os.path.join(json_dir, f'{slide_id}.json')

        process_svs_file(slide_path, json_path, tmp)
    for slide in test_slides:
        tmp = Path(output_dir) / Path('test')
        slide_id, _ = os.path.splitext(slide)
        slide_path = os.path.join(slide_dir, slide)
        json_path = os.path.join(json_dir, f'{slide_id}.json')

        process_svs_file(slide_path, json_path, tmp)
if __name__ == '__main__':

    main()
