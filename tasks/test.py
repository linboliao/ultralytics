import os

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
import argparse

from PIL import Image, ImageDraw
from loguru import logger

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='/data2/lbliao/Code/ultralytics/runs/segment/train5/weights/best.pt')
parser.add_argument('--data', type=str, default='/data2/lbliao/Code/ultralytics/segmentation/cfg/data/segment.yaml')
args = parser.parse_args()

model = YOLO(args.ckpt)

model.val(data=args.data, split='test')


image_dir = "/NAS2/Data1/lbliao/Data/MXB/segment/dataset/2048/test/images"  # 测试集路径
label_dir = "/NAS2/Data1/lbliao/Data/MXB/segment/dataset/2048/test/labels"  # 真实掩码路径


def yolo2mask(label_path, img_size=(512, 512), num_classes=None):
    # 读取并解析多边形数据
    polygons = []
    with open(label_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.split()
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            polygons.append((cls, coords))

    # 初始化多通道掩码 [C, H, W]
    mask = np.zeros((num_classes, *img_size), dtype=np.uint8)

    # 绘制每个类别的多边形
    for cls, coords in polygons:
        img = Image.new('L', img_size[::-1], 0)  # 创建空白图像 (width, height)
        draw = ImageDraw.Draw(img)

        # 转换坐标: 归一化 -> 实际像素
        points = [
            (coords[i] * img_size[1], coords[i + 1] * img_size[0])
            for i in range(0, len(coords), 2)
        ]

        # 绘制并填充多边形
        draw.polygon(points, fill=1)
        mask[cls] = np.logical_or(mask[cls], np.array(img)).astype(np.uint8)

    return mask


def calculate_dice(pred_mask, true_mask, smooth=1e-10):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = pred_mask.sum() + true_mask.sum()
    return (2.0 * intersection + smooth) / (union + smooth)


def calculate_iou(pred_mask, true_mask, smooth=1e-10):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return (intersection + smooth) / (union + smooth)


def multi_class_metrics(pred_mask, true_mask):
    iou_scores = []
    dice_scores = []
    for c in range(pred_mask.shape[0]):
        pred_bin = (pred_mask[c] > 0.5).astype(int)  # 二值化预测 → 0/1
        true_bin = (true_mask[c] > 0.5).astype(int)  # 二值化真实 → 0/1
        if np.sum(pred_bin) == 0 and np.sum(true_bin) == 0:
            continue
        iou = calculate_iou(pred_bin, true_bin)
        dice = calculate_dice(pred_bin, true_bin)
        if 0 < iou < 1:
            iou_scores.append(iou)
        if 0 < iou < 1:
            dice_scores.append(dice)
    return {'dice': np.mean(dice_scores), 'iou': np.mean(iou_scores)}


def calculate(images_dir, labels_dir, num_classes):
    # 遍历图像目录
    iou_scores = []
    dice_scores = []
    for img_name in os.listdir(images_dir):
        # 加载图像获取尺寸
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (896, 896))
        h, w = img.shape[:2]

        # 生成真实掩码
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        true_mask = yolo2mask(label_path, (h, w), num_classes)  # 需返回整数类别掩码
        results = model.predict(img)

        # 初始化全图预测掩码（背景类别为0）
        pred_mask = np.zeros((num_classes, h, w), dtype=np.uint8)

        if results[0].masks is not None:
            masks_tensor = results[0].masks.data.cpu().numpy()  # [N, H, W]

            # 处理每个实例
            for i, mask in enumerate(masks_tensor):
                # 二值化处理
                binary_mask = (mask > 0.5).astype(np.uint8)

                # 获取实例类别ID（背景为0，其他从1开始）
                cls_id = int(results[0].boxes.cls[i].item())
                pred_mask[cls_id] = np.logical_or(pred_mask[cls_id], binary_mask).astype(np.uint8)

        result = multi_class_metrics(true_mask, pred_mask)
        iou_scores.append(result.get('iou'))
        dice_scores.append(result.get('dice'))
    iou_scores = pd.Series(iou_scores).dropna().tolist()
    dice_scores = pd.Series(dice_scores).dropna().tolist()
    return {'dice': np.mean(dice_scores), 'iou': np.mean(iou_scores)}


if __name__ == "__main__":
    metrics = calculate(image_dir, label_dir, 4)
    logger.info(f"\n数据集平均dice: {metrics.get('dice'):.4f}, 平均iou: {metrics.get('iou'):.4f}")
