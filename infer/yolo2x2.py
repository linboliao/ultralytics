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

    def process(self, img, gpu):
        coords, labels, confs = [], [], []
        for model in self.models[:-1]:
            results = model(img, device=gpu, agnostic_nms=True, iou=0.4, verbose=False)
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(reversed(boxes)):
                    [x1, y1, x2, y2] = box.xyxy.tolist()[0]
                    label = self.labels.get(int(box.cls.tolist()[0]), 'Other')
                    conf = box.conf.tolist()[0]

                    coords.append([x1, y1, x2, y2])
                    labels.append(label)
                    confs.append(conf)

        old_coords = copy.copy(coords)
        old_labels = copy.copy(labels)
        old_confs = copy.copy(confs)
        results = self.models[-1](img, device=gpu, agnostic_nms=True, verbose=False)

        # n 个模型结果
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(reversed(boxes)):
                [x1, y1, x2, y2] = box.xyxy.tolist()[0]
                label = self.labels[int(box.cls.tolist()[0])]
                conf = box.conf.tolist()[0]
                if conf < 0.3:
                    continue
                coords.append([x1, y1, x2, y2])
                labels.append(label)
                confs.append(conf * 0.1)

        if len(coords) > 0:
            boxes = torch.tensor(coords, dtype=torch.float32)
            scores = torch.tensor(confs, dtype=torch.float32)
            # N个模型结果进行NMS
            i = torchvision.ops.nms(boxes, scores, 0)  # NMS
            index = i.tolist()
            coords = [coords[i] for i in index if 0 <= i < len(coords)]
            labels = [labels[i] for i in index if 0 <= i < len(labels)]
            confs = [confs[i] for i in index if 0 <= i < len(confs)]
            idxs = [i for i, label in enumerate(labels) if label == 'Malignant']
            area = 0
            for idx in idxs:
                [x1, y1, x2, y2] = coords[idx]
                area += (x2 - x1) * (y2 - y1)
            # 获取癌症面积小于0.2 或者数量少于4的时的非癌结果
            if area < self.patch_size ** 2 * 0.2 or len(idxs) < 4:
                coords = [coords[i] for i in range(len(coords)) if i not in idxs]
                labels = [labels[i] for i in range(len(labels)) if i not in idxs]
                confs = [confs[i] for i in range(len(confs)) if i not in idxs]

            # N-1 个模型 + 前面的非癌结果
            if coords:

                old_coords.extend(coords)
                old_labels.extend(labels)
                old_confs.extend(confs)
                boxes = torch.tensor(old_coords, dtype=torch.float32)
                scores = torch.tensor(old_confs, dtype=torch.float32)
                # 合并
                i = torchvision.ops.nms(boxes, scores, 0.3)
                index = i.tolist()
                #
                coords = [old_coords[i] for i in index if 0 <= i < len(old_coords)]
                labels = [old_labels[i] for i in index if 0 <= i < len(old_coords)]
                confs = [old_confs[i] for i in index if 0 <= i < len(old_coords)]
                # 挑选有癌的结果
                idxs = [i for i, label in enumerate(labels) if label == 'Malignant']
                area = 0
                for idx in idxs:
                    [x1, y1, x2, y2] = coords[idx]
                    area += (x2 - x1) * (y2 - y1)
                # 当癌去极小或者数量为1 时 去除癌区结果
                if area < self.patch_size ** 2 * 0.03 or len(idxs) == 1:
                    coords = [coords[i] for i in range(len(coords)) if i not in idxs]
                    labels = [labels[i] for i in range(len(labels)) if i not in idxs]
                    confs = [confs[i] for i in range(len(confs)) if i not in idxs]

        return coords, labels, confs

    def infer(self, loader):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        t_coords, t_confs, t_labels = [], [], []

        pbar = tqdm(loader, desc="Inferencing Epoch", ncols=100)
        for count, (batch, coords) in enumerate(pbar):
            coords = coords[0]
            # with torch.no_grad():
            #     batch = batch.to(device, non_blocking=True)
            contours, labels, confs = self.process(batch, 0)
            for (x1, y1, x2, y2) in contours:
                x1 = int(x1 + coords[0])
                y1 = int(y1 + coords[1])
                x2 = int(x2 + coords[0])
                y2 = int(y2 + coords[1])
                cnt = [[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]
                t_coords.append([cnt])

            t_confs.extend(confs)
            t_labels.extend(labels)
        self.tissue_area = self.patch_size * self.patch_size * len(loader)
        return t_coords, t_confs, t_labels

    def postprocess(self, result, output_path):
        coords, confs,labels= result
        malignant_area = 0.0
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

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        self.malignant_area = malignant_area


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
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs, prefetch_factor=16,collate_fn=my_collate)
        else:
            kwargs = {'num_workers': 1, 'pin_memory': True}
            print('Data Loader args:', kwargs)
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs,collate_fn=my_collate)
        t_coords, t_confs, t_labels = converter.infer(loader)
        converter.postprocess((t_coords, t_confs, t_labels), output_path)
        area_data.append({'slide_id': slide_id, 'area': f'{converter.malignant_area / converter.tissue_area * 100:.4f}%'})

        print('time per slide: {:.1f}'.format(time.time() - one_slide_start))
    if area_data:
        save_area(area_data, os.path.join(args.output_dir, 'area.csv'))
    print('Time used for this dataset:{:.1f}'.format(time.time() - process_start_time))
    print('Inference ends', end='')
