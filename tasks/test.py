import gc
import os
import json
import argparse
import shutil

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from pathlib import Path
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
from ultralytics import YOLO, RTDETR, YOLOE, YOLOWorld
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ops


def collate_fn(batch):
    image_paths = []

    for image_path in batch:
        image_paths.append(image_path)

    return image_paths


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self.image_files[idx]


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


def save_masks(results, save_dir):
    """
    将每个类别的实例保存为单通道图像，像素值=类别ID+1
    如果没有检测到实例，保存全为0的mask
    """
    os.makedirs(save_dir, exist_ok=True)

    for r in results:
        h, w = r.orig_shape[:2]

        class_mask = np.zeros((h, w), dtype=np.uint8)

        if r.masks is not None and len(r.masks) > 0:
            class_ids = r.boxes.cls.cpu().numpy()
            masks = r.masks.data.cpu().numpy()

            for mask, class_id in zip(masks, class_ids):
                mask = mask.reshape(*mask.shape, 1)
                mask = ops.scale_image(mask, r.orig_shape)
                mask = np.squeeze(mask, axis=-1)
                mask_binary = (mask > 0.5).astype(bool)
                class_value = int(class_id) + 1
                class_mask[mask_binary] = class_value

        Image.fromarray(class_mask).save(f'{save_dir}/{Path(r.path).name}')


def calculate_hd95_sitk(mask_np, pred_np, voxel_spacing=(1.0, 1.0)):
    if mask_np.max() == 0 and pred_np.max() == 0:
        # 两个掩码都为空，返回距离 0
        return 0.0

    # 检查其中一个掩码是否为空
    if mask_np.max() == 0 or pred_np.max() == 0:
        # 其中一个掩码为空，返回一个很大的距离（如对角线长度）或跳过
        # 这里返回图像对角线长度作为距离
        h, w = mask_np.shape
        return np.sqrt(h ** 2 + w ** 2)
    mask_sitk = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    pred_sitk = sitk.GetImageFromArray(pred_np.astype(np.uint8))

    mask_sitk.SetSpacing(voxel_spacing)
    pred_sitk.SetSpacing(voxel_spacing)

    hausdorff_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_filter.Execute(mask_sitk, pred_sitk)

    hd95 = hausdorff_filter.GetHausdorffDistance()

    return hd95


def calc_one(mask_path, pred_path, classes=(1, 2)):
    dice_l, iou_l, hd_l = [], [], []
    m = cv2.imread(mask_path, 0)
    if not os.path.exists(pred_path):
        p = np.zeros_like(m)
    else:
        p = cv2.imread(pred_path, 0)

    for c in classes:
        mb = (m == c)
        pb = (p == c)
        if not mb.any() and not pb.any():
            continue
        inter = np.logical_and(mb, pb).sum()
        sum_ab = mb.sum() + pb.sum()
        union = np.logical_or(mb, pb).sum()
        dice = 2 * inter / (sum_ab + 1e-8)
        iou = inter / (union + 1e-8)

        hd = calculate_hd95_sitk(mb, pb)
        dice_l.append(dice)
        iou_l.append(iou)
        hd_l.append(hd)
        del m, p, mb, pb

    return (np.mean(dice_l) if dice_l else np.nan,
            np.mean(iou_l) if iou_l else np.nan,
            np.nanmean(hd_l) if hd_l else np.nan)


def evaluate(mask_dir, pred_dir, classes=(1, 2), max_workers=None):
    mask_dir, pred_dir = Path(mask_dir), Path(pred_dir)
    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    mask_files = [f for f in mask_dir.iterdir() if f.suffix.lower() in IMG_EXTS and f.is_file()]
    tasks = [(m, pred_dir / m.name) for m in mask_files]

    if max_workers is None:
        max_workers = min(8, os.cpu_count() // 2)

    results = {'dice': [], 'iou': [], 'hd95': []}
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(calc_one, m, p, classes) for m, p in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='评估', smoothing=0.1):
            r = fut.result()
            if r and not any(np.isnan(x) for x in r):
                results['dice'].append(r[0])
                results['iou'].append(r[1])
                results['hd95'].append(r[2])

            if len(results['dice']) % 50 == 0:
                gc.collect()

    gc.collect()

    return (
        np.mean(results['dice']) if results['dice'] else np.nan,
        np.mean(results['iou']) if results['iou'] else np.nan,
        np.mean(results['hd95']) if results['hd95'] else np.nan
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['yolo', 'rtdetr', 'yoloe', 'yoloworld'], required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--phase', type=str, default='val')
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--project', type=str, default='runs/val')
    parser.add_argument('--device', default='0')
    args = parser.parse_args()

    ModelCls = dict(yolo=YOLO, rtdetr=RTDETR, yoloe=YOLOE, yoloworld=YOLOWorld)[args.model]
    model = ModelCls(args.ckpt)

    metrics = model.val(data=args.data, split=args.phase, batch=args.batch, device=args.device, name=args.name, project=args.project)
    res_dict = metrics.results_dict
    torch.cuda.empty_cache()
    data_cfg = check_det_dataset(args.data)
    img_dirs = data_cfg.get(args.phase)
    label_dirs = [d.replace('/images', '/labels') for d in img_dirs]
    mask_dirs = [d.replace('/images', '/masks') for d in img_dirs]
    pred_dirs = [d.replace('/images', '/preds') for d in img_dirs]
    dice_list, iou_list, hd_list = [], [], []
    for img_dir, pred_dir, mask_dir in zip(img_dirs, pred_dirs, mask_dirs):
        dataset = ImageDataset(img_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate_fn)
        os.makedirs(pred_dir, exist_ok=True)
        for img_paths in tqdm(dataloader, desc="模型推理", unit="batch"):
            results = model(img_paths, verbose=False, device=args.device)
            save_masks(results, pred_dir)

        d, i, h = evaluate(mask_dir, pred_dir)
        dice_list.append(d)
        iou_list.append(i)
        hd_list.append(h)

    d, i, h = np.mean(dice_list), np.mean(iou_list), np.mean(hd_list)
    logger.info(f'Dice: {d:.4f} | IoU: {i:.4f} | HD95: {h:.2f}')

    # 5. 合并结果并保存
    res_dict.update(dice=d, iou=i, hd95=h)
    save_path = Path(metrics.save_dir) / 'metrics.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res_dict, save_path.open('w', encoding='utf-8'), indent=4, ensure_ascii=False)
