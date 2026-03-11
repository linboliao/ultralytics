import argparse
import copy
import json
import os
import time
import uuid
from datetime import datetime

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from infer.dataset_h5 import Whole_Slide_Bag_FP
from tasks.wsi import WSIOperator
from ultralytics import YOLO


class Yolo2X:
    def __init__(self, ckpts):
        self.models = []
        for ckpt in ckpts:
            self.models.append(YOLO(ckpt))

    def infer(self, **kwargs):
        raise NotImplementedError()

    def postprocess(self, **kwargs):
        raise NotImplementedError()


class Yolo2XDetect(Yolo2X):
    def __init__(self, ckpts):
        super().__init__(ckpts)
        self.labels = {0: 'Benign', 1: 'Malignant', 2: 'Other'}
        self.colors = {'Benign': [0, 255, 0], 'Malignant': [255, 0, 0], 'Other': [128, 128, 128]}
        self.patch_size = 2048

    # def process(self, img, gpu):
    #     coords, labels, confs = [], [], []
    #     for model in self.models[:-1]:
    #         results = model(img, device=gpu, agnostic_nms=True, iou=0.4, verbose=False)
    #         for result in results:
    #             boxes = result.boxes
    #             for i, box in enumerate(reversed(boxes)):
    #                 [x1, y1, x2, y2] = box.xyxy.tolist()[0]
    #                 label = self.labels.get(int(box.cls.tolist()[0]), 'Other')
    #                 conf = box.conf.tolist()[0]
    #
    #                 coords.append([x1, y1, x2, y2])
    #                 labels.append(label)
    #                 confs.append(conf)
    #
    #     old_coords = copy.copy(coords)
    #     old_labels = copy.copy(labels)
    #     old_confs = copy.copy(confs)
    #
    #     results = self.models[-1](img, device=gpu, agnostic_nms=True, verbose=False)
    #
    #     # n 个模型结果
    #     for result in results:
    #         boxes = result.boxes
    #         for i, box in enumerate(reversed(boxes)):
    #             [x1, y1, x2, y2] = box.xyxy.tolist()[0]
    #             label = self.labels[int(box.cls.tolist()[0])]
    #             conf = box.conf.tolist()[0]
    #             if conf < 0.2:
    #                 continue
    #             coords.append([x1, y1, x2, y2])
    #             labels.append(label)
    #             confs.append(conf * 0.1)
    #
    #     if len(coords) > 0:
    #         boxes = torch.tensor(coords, dtype=torch.float32)
    #         scores = torch.tensor(confs, dtype=torch.float32)
    #         # N个模型结果进行NMS
    #         i = torchvision.ops.nms(boxes, scores, 0)  # NMS
    #         index = i.tolist()
    #         coords = [coords[i] for i in index if 0 <= i < len(coords)]
    #         labels = [labels[i] for i in index if 0 <= i < len(labels)]
    #         confs = [confs[i] for i in index if 0 <= i < len(confs)]
    #         idxs = [i for i, label in enumerate(labels) if label == 'Malignant']
    #         area = 0
    #         for idx in idxs:
    #             [x1, y1, x2, y2] = coords[idx]
    #             area += (x2 - x1) * (y2 - y1)
    #         # 获取癌症面积小于0.2 或者数量少于4的时的非癌结果
    #         if area < self.patch_size ** 2 * 0.05 or len(idxs) < 4:
    #             coords = [coords[i] for i in range(len(coords)) if i not in idxs]
    #             labels = [labels[i] for i in range(len(labels)) if i not in idxs]
    #             confs = [confs[i] for i in range(len(confs)) if i not in idxs]
    #
    #         # N-1 个模型 + 前面的非癌结果
    #         if coords:
    #
    #             old_coords.extend(coords)
    #             old_labels.extend(labels)
    #             old_confs.extend(confs)
    #             boxes = torch.tensor(old_coords, dtype=torch.float32)
    #             scores = torch.tensor(old_confs, dtype=torch.float32)
    #             # 合并
    #             i = torchvision.ops.nms(boxes, scores, 0.3)
    #             index = i.tolist()
    #             #
    #             coords = [old_coords[i] for i in index if 0 <= i < len(old_coords)]
    #             labels = [old_labels[i] for i in index if 0 <= i < len(old_coords)]
    #             confs = [old_confs[i] for i in index if 0 <= i < len(old_coords)]
    #             # 挑选有癌的结果
    #             idxs = [i for i, label in enumerate(labels) if label == 'Malignant']
    #             area = 0
    #             for idx in idxs:
    #                 [x1, y1, x2, y2] = coords[idx]
    #                 area += (x2 - x1) * (y2 - y1)
    #             # 当癌去极小或者数量为1 时 去除癌区结果
    #             if area < self.patch_size ** 2 * 0.03 or len(idxs) == 1:
    #                 coords = [coords[i] for i in range(len(coords)) if i not in idxs]
    #                 labels = [labels[i] for i in range(len(labels)) if i not in idxs]
    #                 confs = [confs[i] for i in range(len(confs)) if i not in idxs]
    #
    #     return coords, labels, confs

    def process(self, img, gpu):
        # 收集前 N-1 个模型的检测结果（正常置信度权重）
        all_boxes = []  # 坐标 [x1, y1, x2, y2]
        all_labels = []
        all_scores = []

        for model in self.models[:-1]:
            results = model(img, device=gpu, agnostic_nms=False, iou=0.4, verbose=False)
            for result in results:
                boxes = result.boxes
                for box in reversed(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    label = self.labels.get(cls_id, 'Other')
                    confidence = float(box.conf[0].item())

                    all_boxes.append([x1, y1, x2, y2])
                    all_labels.append(label)
                    all_scores.append(confidence)

        # 备份前 N-1 个模型的原始结果（后面融合时会用到）
        backup_boxes = all_boxes.copy()
        backup_labels = all_labels.copy()
        backup_scores = all_scores.copy()

        # 最后一个模型（通常是 malignant 专用），置信度权重大幅降低
        malignant_results = self.models[-1](img, device=gpu, agnostic_nms=False, verbose=False)
        for result in malignant_results:
            boxes = result.boxes
            for box in reversed(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                label = self.labels[cls_id]
                confidence = float(box.conf[0].item())

                if confidence < 0.2:
                    continue

                all_boxes.append([x1, y1, x2, y2])
                all_labels.append(label)
                all_scores.append(confidence * 0.7)  # 权重降到 10%

        if not all_boxes:
            return [], [], []

        # ─────────────────────────────
        # 第一次 NMS：融合所有模型的预测（包括低权重的 malignant 结果）
        # ─────────────────────────────
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)

        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, 0.0).tolist()

        filtered_boxes = [all_boxes[i] for i in keep_indices]
        filtered_labels = [all_labels[i] for i in keep_indices]
        filtered_scores = [all_scores[i] for i in keep_indices]

        # 找出被判定为 Malignant 的检测框索引
        malignant_indices = [i for i, label in enumerate(filtered_labels) if label == 'Malignant']

        total_malignant_area = 0
        for idx in malignant_indices:
            x1, y1, x2, y2 = filtered_boxes[idx]
            total_malignant_area += (x2 - x1) * (y2 - y1)

        # 恶性区域总面积太小 或 数量太少 → 视为非癌，移除所有 malignant 预测
        if total_malignant_area < self.patch_size ** 2 * 0.05 or len(malignant_indices) < 4:
            filtered_boxes = [b for i, b in enumerate(filtered_boxes) if i not in malignant_indices]
            filtered_labels = [l for i, l in enumerate(filtered_labels) if i not in malignant_indices]
            filtered_scores = [s for i, s in enumerate(filtered_scores) if i not in malignant_indices]

        # ─────────────────────────────
        # 如果还有检测结果，把第一次过滤后的结果 + 前 N-1 模型原始结果 再做一次 NMS
        # ─────────────────────────────
        if filtered_boxes:
            combined_boxes = backup_boxes + filtered_boxes
            combined_labels = backup_labels + filtered_labels
            combined_scores = backup_scores + filtered_scores

            boxes_tensor = torch.tensor(combined_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(combined_scores, dtype=torch.float32)

            keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, 0.3).tolist()

            final_boxes = [combined_boxes[i] for i in keep_indices]
            final_labels = [combined_labels[i] for i in keep_indices]
            final_scores = [combined_scores[i] for i in keep_indices]

            # 再次检查恶性区域（采用更严格的过滤条件）
            malignant_indices = [i for i, label in enumerate(final_labels) if label == 'Malignant']

            total_malignant_area = 0
            for idx in malignant_indices:
                x1, y1, x2, y2 = final_boxes[idx]
                total_malignant_area += (x2 - x1) * (y2 - y1)

            # 面积极小 或 只检测到单个 malignant → 去除 malignant 判断
            if total_malignant_area < self.patch_size ** 2 * 0.03 or len(malignant_indices) == 1:
                final_boxes = [b for i, b in enumerate(final_boxes) if i not in malignant_indices]
                final_labels = [l for i, l in enumerate(final_labels) if i not in malignant_indices]
                final_scores = [s for i, s in enumerate(final_scores) if i not in malignant_indices]

            return final_boxes, final_labels, final_scores

        # 如果中间过程已把 malignant 全部过滤掉，则返回前 N-1 模型的备份结果
        return backup_boxes, backup_labels, backup_scores

    def infer(self, loader):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        t_coords, t_confs, t_labels = [], [], []
        malignant_area = 0.0
        pbar = tqdm(loader, desc="Inferencing Epoch", ncols=100)
        for count, (batch, coords) in enumerate(pbar):
            coords = coords[0]
            # with torch.no_grad():
            #     batch = batch.to(device, non_blocking=True)
            contours, labels, confs = self.process(batch, 0)
            for i, (x1, y1, x2, y2) in enumerate(contours):
                x1 = int(x1 + coords[0])
                y1 = int(y1 + coords[1])
                x2 = int(x2 + coords[0])
                y2 = int(y2 + coords[1])
                cnt = [[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]
                t_coords.append([cnt])
                if labels[i] == 'Malignant':
                    malignant_area += (x2 - x1) * (y2 - y1)

            t_confs.extend(confs)
            t_labels.extend(labels)
        self.tissue_area = self.patch_size * self.patch_size * len(loader)
        self.malignant_area = malignant_area
        return t_coords, t_confs, t_labels

    def postprocess(self, result, output_path):
        coords, confs, labels = result
        feature_template = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": None},
            "properties": {
                "objectType": "annotation",
                "classification": {"name": None, "color": None}
            }
        }

        features = [
            {
                **feature_template,  # 复制模板
                "id": str(uuid.uuid4()),  # 生成唯一 ID
                "geometry": {"type": "Polygon", "coordinates": coord},  # 填充坐标
                "properties": {
                    "name": f'{conf:.4f}',
                    "classification": {
                        "name": label,
                        "color": self.colors[label]  # 填充颜色
                    }
                }
            }
            for coord, label, conf in zip(coords, labels, confs) if label == 'Malignant'
        ]
        if not len(features):
            return
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)


def find_all_wsi_paths(wsi_root, extentions):
    """
    find the full wsi path under data_root, return a dict {slide_id: full_path}
    """
    # to support more than one ext, e.g., support .svs and .mrxs
    exts = extentions.split(';')
    result = {}
    for file in os.listdir(wsi_root):
        base, ext = os.path.splitext(file)
        if ext.lower() in exts:
            result[base] = os.path.join(wsi_root, file)
    return result


def save_area(area_data, csv_path, key_column='slide_id'):
    """
    如果new_data中的key_column值已存在于CSV文件中，则更新该行；否则追加新行。

    Args:
        area_data (list of dict): 新的数据行，每个字典代表一行。
        csv_path (str): CSV文件的路径。
        key_column (str): 用于判断是否重复的列名，默认为'slide_id'。
    """
    new_df = pd.DataFrame(area_data)

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)

        mask = existing_df[key_column].isin(new_df[key_column])
        existing_df_clean = existing_df[~mask]

        updated_df = pd.concat([existing_df_clean, new_df], ignore_index=True)

        updated_df.to_csv(csv_path, index=False)
        print(f"成功更新CSV文件: {csv_path}。")

    else:
        new_df.to_csv(csv_path, index=False)
        print(f"创建新的CSV文件并写入数据: {csv_path}")


def my_collate(batch):
    imgs = [item[0] for item in batch]  # 仍然是 PIL
    coords = [item[1] for item in batch]
    return imgs, coords


parser = argparse.ArgumentParser(description='YOLO to X')
parser.add_argument('--data_coors_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--ckpts', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--model', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--output_dir', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    process_start_time = time.time()
    print('initializing dataset')

    exist_idxs = []
    all_wsi_paths = find_all_wsi_paths(args.data_slide_dir, args.slide_ext)
    total = len(all_wsi_paths)
    print('Total number of WSIs:', total)
    os.makedirs(args.output_dir, exist_ok=True)
    dest_files = os.listdir(args.output_dir)
    for slide_id in all_wsi_paths.keys():
        h5_file_path = str(os.path.join(args.data_coors_dir, 'patches', slide_id + '.h5'))
        if not os.path.exists(h5_file_path):
            print(h5_file_path, 'does not exist ...')
            continue
        elif slide_id + f'-{args.task}.geojson' in dest_files:
            print('geojosn file exist, skip {}'.format(slide_id))
            continue
        else:
            exist_idxs.append(slide_id)

    ckpts = args.ckpts.split(';')
    converter = Yolo2XDetect(ckpts)

    print('WSIs need to be processed: {} of {}'.format(len(exist_idxs), total))
    area_data = []
    for index, slide_id in enumerate(exist_idxs):
        h5_file_path = str(os.path.join(args.data_coors_dir, 'patches', slide_id + '.h5'))
        slide_file_path = all_wsi_paths[slide_id]

        print('Time:', datetime.now().strftime('"%Y-%m-%d, %H:%M:%S"'))
        print('\nprogress: {}/{}, slide_id: {}'.format(index, len(exist_idxs), slide_id))

        output_path = os.path.join(args.output_dir, slide_id + f'-{args.task}.geojson')

        one_slide_start = time.time()
        try:
            wsi = WSIOperator(slide_file_path)
        except:
            print('Failed to read WSI:', slide_file_path)
            continue

        # custom_transformer = transforms.Compose([
        #     transforms.ToTensor(),
        # ])

        # dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, pretrained=True, custom_transforms=custom_transformer, fast_read=True)
        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, pretrained=True, fast_read=True)
        if slide_file_path.endswith('.svs'):
            kwargs = {'num_workers': 8, 'pin_memory': True}
            print('Data Loader args:', kwargs)
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs, prefetch_factor=16, collate_fn=my_collate)
        else:
            kwargs = {'num_workers': 1, 'pin_memory': True}
            print('Data Loader args:', kwargs)
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs, collate_fn=my_collate)
        t_coords, t_confs, t_labels = converter.infer(loader)
        converter.postprocess((t_coords, t_confs, t_labels), output_path)
        area_data.append({'slide_id': slide_id, 'area': f'{converter.malignant_area / converter.tissue_area * 100:.4f}%'})

        print('time per slide: {:.1f}'.format(time.time() - one_slide_start))
    if area_data:
        save_area(area_data, os.path.join(args.output_dir, 'area.csv'))
    print('Time used for this dataset:{:.1f}'.format(time.time() - process_start_time))
    print('Inference ends', end='')
