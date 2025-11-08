import json
import os
import time

import torch
import numpy as np
import pandas as pd
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from ultralytics import YOLO, YOLOE, RTDETR, YOLOWorld
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import mask_iou
import warnings

warnings.filterwarnings("ignore")


class ImageDataset(Dataset):
    def __init__(self, images_dir, labels_dir, num_classes, transform=None):
        """
        初始化数据集
        Args:
            images_dir: 图像文件目录
            labels_dir: 标签文件目录
            num_classes: 类别数量
            transform:
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.num_classes = num_classes
        self.transform = transform if transform else transforms.ToTensor()

        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"找到 {len(self.image_files)} 个图像文件")

        self.images, self.masks = self._preload_data()

    def _preload_data(self):
        """预加载所有图像和mask到内存"""
        print("开始预加载数据到内存...")

        images = [None] * len(self.image_files)
        masks = [None] * len(self.image_files)

        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_index = {executor.submit(self._load_single_item, idx): idx for idx in range(len(self.image_files))}

            for future in tqdm(as_completed(future_to_index), total=len(self.image_files), desc="预加载进度"):
                idx = future_to_index[future]
                try:
                    images[idx], masks[idx] = future.result()
                except Exception as e:
                    print(f"加载索引 {idx} 的数据时出错: {e}")

        return images, masks

    def _load_single_item(self, idx):
        """加载单个数据项"""
        img_name = self.image_files[idx]
        img_base_name = os.path.splitext(img_name)[0]

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        image = self.transform(image)

        label_path = os.path.join(self.labels_dir, img_base_name + '.txt')
        mask = yolo2mask(label_path, (h, w), self.num_classes)
        mask = torch.from_numpy(mask).long()

        return image, mask

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        根据索引获取图像和对应的mask
        """
        return self.images[idx], self.masks[idx]


def yolo2mask(label_path, img_size=(512, 512), num_classes=None):
    """

    Args:
        label_path: yolo label .txt
        img_size:
        num_classes:

    Returns:
        mask (num_classes, *img_size)

    """
    polygons = []
    with open(label_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.split()
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            polygons.append((cls, coords))

    mask = np.zeros((num_classes, *img_size), dtype=np.uint8)

    for cls, coords in polygons:
        if cls >= num_classes: continue
        img = Image.new('L', img_size[::-1], 0)
        draw = ImageDraw.Draw(img)

        points = [(coords[i] * img_size[1], coords[i + 1] * img_size[0]) for i in range(0, len(coords), 2)]

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
        pred_bin = (pred_mask[c] > 0.5).astype(int)
        true_bin = (true_mask[c] > 0.5).astype(int)
        if np.sum(pred_bin) == 0 and np.sum(true_bin) == 0:
            continue
        iou = calculate_iou(pred_bin, true_bin)
        # iou = mask_iou(torch.from_numpy(pred_bin).reshape(1,-1), torch.from_numpy(true_bin).reshape(1,-1)).item()
        dice = calculate_dice(pred_bin, true_bin)
        if 0 < iou <= 1:
            iou_scores.append(iou)
            dice_scores.append(dice)
    return {'dice': np.mean(dice_scores), 'iou': np.mean(iou_scores)}


def calculate(dataloader, model, args):
    iou_scores = []
    dice_scores = []

    for b_images, b_masks in tqdm(dataloader, desc="模型推理", unit="batch"):
        results = model(b_images, verbose=False, device='4')
        for result, label in zip(results, b_masks):
            pred = np.zeros(label.shape, dtype=np.uint8)
            if result.masks is not None:
                pred_tensor = result.masks.data.cpu().numpy()
                for i, mask in enumerate(pred_tensor):
                    cls_id = int(result.boxes.cls[i].item())
                    pred[cls_id] = np.logical_or(pred[cls_id], mask).astype(np.uint8)
                label = label.numpy()
                result_metrics = multi_class_metrics(pred, label)
            elif label.max() > 0:
                result_metrics = {'dice': 0, 'iou': 0}
            else:
                continue

            iou_scores.append(result_metrics.get('iou'))
            dice_scores.append(result_metrics.get('dice'))

    iou_scores = pd.Series(iou_scores).dropna().tolist()
    dice_scores = pd.Series(dice_scores).dropna().tolist()
    return np.mean(dice_scores), np.mean(iou_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['yolo', 'rtdetr', 'yoloe', 'yoloworld'], help='Model type to test: yolo, rtdetr, or yoloe')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', type=str)
    parser.add_argument('--phase', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--project', type=str)
    args = parser.parse_args()

    if args.model == 'yolo':
        model = YOLO(args.ckpt)
    elif args.model == 'rtdetr':
        model = RTDETR(args.ckpt)
    elif args.model == 'yoloe':
        model = YOLOE(args.ckpt)
    elif args.model == 'yoloworld':
        model = YOLOWorld(args.ckpt)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    metrics = model.val(data=args.data, split=args.phase, name=args.name, batch=args.batch, project=args.project, device=args.device)
    result = metrics.results_dict

    data = check_det_dataset(args.data)
    img_dirs = data.get(args.phase)
    label_dirs = [d.replace('/images', '/labels') for d in img_dirs]
    mask_dirs = [d.replace('/images', '/masks') for d in img_dirs]

    dice_scores, iou_scores = [], []
    for img_dir, label_dir in zip(img_dirs, label_dirs):
        dataset = ImageDataset(img_dir, label_dir, 2)
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4)
        dice, iou = calculate(dataloader, model, args)
        dice_scores.append(dice)
        iou_scores.append(iou)
    logger.info(f"\n数据集平均dice: {np.mean(dice_scores):.4f}, 平均iou: {np.mean(iou_scores):.4f}")
    result.update({'dice': np.mean(dice_scores), 'iou': np.mean(iou_scores)})

    save_path = metrics.save_dir / 'metrics.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)
