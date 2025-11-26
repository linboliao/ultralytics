import argparse
import glob
import json
import os
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon
from shapely.ops import unary_union
from torch.utils.data import DataLoader
from torchvision import transforms, ops
from tqdm import tqdm

from infer.dataset_h5 import Whole_Slide_Bag_FP
from tasks.wsi import WSIOperator
from ultralytics import YOLO, YOLOE, RTDETR, YOLOWorld
from ultralytics.engine.results import Boxes

THRESHOLD = 0.2


class YOLO2X:
    def __init__(self, model, ckpts):
        self.models = []
        if model == 'yolo':
            model = YOLO
        elif model == 'rtdetr':
            model = RTDETR
        elif model == 'yoloe':
            model = YOLOE
        elif model == 'yoloworld':
            model = YOLOWorld
        else:
            raise ValueError(f"Unsupported model type: {model}")

        for ckpt in ckpts:
            self.models.append(model(ckpt))

    def infer(self, **kwargs):
        raise NotImplementedError()

    def post_process(self, **kwargs):
        raise NotImplementedError()


class YOLO2GeoJsonDetect(YOLO2X):
    def __init__(self, model, ckpts):
        super().__init__(model, ckpts)
        self.labels = {0: 'Benign', 1: 'Malignant', 2: 'Other', 3: 'Other', 4: 'Other'}
        self.colors = {'Benign': [0, 255, 0], 'Malignant': [255, 0, 0], 'Other': [128, 128, 128]}

    def infer(self, loader):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        results_list = []
        coords_list = []
        pbar = tqdm(loader, desc="Inference", ncols=100)

        def cal_area(xyxy_tensor):
            """计算边界框面积之和"""
            if len(xyxy_tensor) == 0:
                return 0.0
            x1 = xyxy_tensor[:, 0]
            y1 = xyxy_tensor[:, 1]
            x2 = xyxy_tensor[:, 2]
            y2 = xyxy_tensor[:, 3]
            widths = x2 - x1
            heights = y2 - y1
            areas = widths * heights
            return torch.sum(areas).item()

        for count, (batch, coords) in enumerate(pbar):
            with torch.no_grad():
                batch = batch.to(device, non_blocking=True)
                # 通过所有模型进行推理
                results = []
                for idx, model in enumerate(self.models):
                    results.append(model(batch, device='0', agnostic_nms=True, iou=0.4, verbose=False, conf=0.2))

                result = []
                for idx in range(batch.size(0)):

                    xyxy_all, conf_all, cls_all = [], [], []
                    boxes = [item[idx].boxes for item in results]

                    for j, box in enumerate(boxes):
                        if len(box) > 0:
                            xyxy_all.extend(box.xyxy)
                            cls_all.extend(box.cls)
                            adjusted_conf = []
                            for i in range(len(box)):
                                if box.cls[i] != 1:  # 类别不是1
                                    adjusted_conf.append(box.conf[i] * 0.8)
                                else:  # 类别是1，保持原置信度
                                    adjusted_conf.append(box.conf[i])

                            conf_all.extend(torch.stack(adjusted_conf))

                    if xyxy_all:
                        xyxy_tensor = torch.stack(xyxy_all, dim=0)
                        conf_tensor = torch.stack(conf_all, dim=0)
                        cls_tensor = torch.stack(cls_all, dim=0)

                        i_first = ops.nms(xyxy_tensor, conf_tensor, 0.5)
                        xyxy_A = xyxy_tensor[i_first]
                        conf_A = conf_tensor[i_first]
                        cls_A = cls_tensor[i_first]

                        cls1_mask_A = (cls_A == 1)
                        count_cls1_A = cls1_mask_A.sum().item()
                        if count_cls1_A > 0:
                            xyxy_cls1_A = xyxy_A[cls1_mask_A]
                            area_cls1_A = cal_area(xyxy_cls1_A)
                        else:
                            area_cls1_A = 0.0

                        img_area = batch.size(2) * batch.size(3)  # 图像面积

                        # 第一阶段过滤：如果条件满足，去除cls=1的边界框
                        if count_cls1_A < 3 and area_cls1_A < img_area * 0.2:
                            keep_mask_A = (cls_A != 1)  # 只保留非cls=1的边界框
                            xyxy_A = xyxy_A[keep_mask_A]
                            conf_A = conf_A[keep_mask_A]
                            cls_A = cls_A[keep_mask_A]

                        # 第二阶段：收集前三个模型的结果（结果B）
                        xyxy_B, conf_B, cls_B = [], [], []
                        for j in range(3):  # 只取前三个模型
                            box = boxes[j]
                            if len(box) > 0:
                                xyxy_B.extend(box.xyxy)
                                cls_B.extend(box.cls)
                                adjusted_conf = []
                                for i in range(len(box)):
                                    if box.cls[i] != 1:  # 类别不是1
                                        adjusted_conf.append(box.conf[i] * 0.8)
                                    else:  # 类别是1，保持原置信度
                                        adjusted_conf.append(box.conf[i])

                                conf_B.extend(torch.stack(adjusted_conf))

                        # 将第一阶段过滤后的结果A与前三个模型的结果B合并
                        if xyxy_B:
                            xyxy_tensor_B = torch.stack(xyxy_B, dim=0)
                            conf_tensor_B = torch.stack(conf_B, dim=0)
                            cls_tensor_B = torch.stack(cls_B, dim=0)
                        else:
                            xyxy_tensor_B = torch.tensor([], device=device)
                            conf_tensor_B = torch.tensor([], device=device)
                            cls_tensor_B = torch.tensor([], device=device)

                        # 如果结果A不为空，则与结果B合并
                        if len(xyxy_A) > 0:
                            xyxy_combined = torch.cat([xyxy_A, xyxy_tensor_B], dim=0)
                            conf_combined = torch.cat([conf_A, conf_tensor_B], dim=0)
                            cls_combined = torch.cat([cls_A, cls_tensor_B], dim=0)
                        else:
                            xyxy_combined = xyxy_tensor_B
                            conf_combined = conf_tensor_B
                            cls_combined = cls_tensor_B

                        # 第二次NMS合并（合并结果C）
                        if len(xyxy_combined) > 0:
                            i_second = ops.nms(xyxy_combined, conf_combined, 0.3)
                            xyxy_C = xyxy_combined[i_second]
                            conf_C = conf_combined[i_second]
                            cls_C = cls_combined[i_second]
                        else:
                            xyxy_C = xyxy_combined
                            conf_C = conf_combined
                            cls_C = cls_combined

                        cls1_mask_C = (cls_C == 1)
                        count_cls1_C = cls1_mask_C.sum().item()
                        if count_cls1_C > 0:
                            xyxy_cls1_C = xyxy_C[cls1_mask_C]
                            area_cls1_C = cal_area(xyxy_cls1_C)
                        else:
                            area_cls1_C = 0.0

                        if count_cls1_C == 1 or area_cls1_C < img_area * 0.03:
                            keep_mask_C = (cls_C != 1)  # 只保留非cls=1的边界框
                            xyxy_C = xyxy_C[keep_mask_C]
                            conf_C = conf_C[keep_mask_C]
                            cls_C = cls_C[keep_mask_C]

                        if len(xyxy_C) > 0:
                            conf_C = conf_C.unsqueeze(1)  # 增加维度
                            cls_C = cls_C.unsqueeze(1)
                            box_tensor = torch.cat([xyxy_C, conf_C, cls_C], dim=1)
                            result.append(Boxes(box_tensor, (batch.shape[2], batch.shape[3])))
                        else:
                            result.append(None)
                    else:
                        result.append(None)

                results_list.append(result)
                coords_list.append(coords)

        self.tissue_area = batch.shape[2] * batch.shape[3] * len(loader)
        return results_list, coords_list

    def post_process(self, results_list, coords_list, output_path):
        features = []
        malignant_polygons= []
        for results, coords in zip(results_list, coords_list):
            for boxes, coord in zip(results, coords):
                coord = coord.to('cpu').tolist()
                if boxes is not None:
                    xyxy_list = boxes.xyxy.cpu().tolist()
                    conf_list = boxes.conf.cpu().tolist()
                    cls_list = boxes.cls.cpu().tolist()
                    for i in range(len(xyxy_list)):
                        confidence = conf_list[i]
                        if confidence < THRESHOLD and cls_list[i] == 1:
                            continue

                        x1, y1, x2, y2 = xyxy_list[i]

                        x1 = x1 + coord[0]
                        y1 = y1 + coord[1]
                        x2 = x2 + coord[0]
                        y2 = y2 + coord[1]

                        polygon_coordinates = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]
                        label = self.labels[cls_list[i]]
                        if cls_list[i] == 1:
                            rect = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                            malignant_polygons.append(rect)

                        feature = {
                            "type": "Feature",
                            "id": str(uuid.uuid4()),
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": polygon_coordinates
                            },
                            "properties": {
                                "name": f'{confidence:.2f}',
                                "classification": {
                                    "name": label,
                                    "color": self.colors[label]
                                }
                            }
                        }
                        features.append(feature)
        geojson_dict = {
            "type": "FeatureCollection",
            "features": features
        }

        if malignant_polygons:
            merged_malignant = unary_union(malignant_polygons)
            if merged_malignant.geom_type == 'MultiPolygon':
                malignant_area = sum(poly.area for poly in merged_malignant.geoms)
            else:
                malignant_area = merged_malignant.area
        else:
            malignant_area = 0.0
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_dict, f, indent=2, ensure_ascii=False)

    self.malignant_area = malignant_area


class YOLO2GeoJsonSegment(YOLO2X):
    def __init__(self, model, ckpts):
        super().__init__(model, ckpts)
        self.labels = {0: 'Benign', 1: 'Malignant', 2: 'Other'}
        self.colors = {'Benign': [0, 255, 0], 'Malignant': [255, 0, 0], 'Other': [128, 128, 128]}

    def obtain_features(self, results, coords):
        features = []
        malignant_area = 0.0
        for result, coord in zip(results, coords):
            if not result.masks:
                continue
            coord = coord.to('cpu').tolist()
            masks_tensor = result.masks.data.cpu().numpy()
            boxes = result.boxes
            class_ids = boxes.cls.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            if hasattr(result, 'names') and result.names is not None:
                names_map = result.names
            else:
                names_map = self.labels

            for idx, (mask, class_id, confidence) in enumerate(zip(masks_tensor, class_ids, confidences)):
                if confidence < THRESHOLD and class_id == 1:
                    continue

                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                box_coords = boxes.xyxy[idx].cpu().numpy()
                x1, y1, x2, y2 = box_coords
                bbox_area = (x2 - x1) * (y2 - y1)

                polygons = []
                for contour in contours:
                    contour_area = cv2.contourArea(contour)

                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.02 * perimeter
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    if len(approx) == 4 and abs(contour_area - bbox_area) < (bbox_area * 0.1):
                        continue
                    if class_id == 1:
                        malignant_area += contour_area
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

                    if len(approx_polygon) >= 3:
                        points = approx_polygon.reshape(-1, 2).tolist()
                        if points[0] != points[-1]:
                            points.append(points[0])
                        polygons.append([[point[0] + coord[0], point[1] + coord[1]] for point in points])
                if not polygons:
                    continue

                label = names_map.get(int(class_id), "unknown")
                for polygon in polygons:
                    feature = {
                        "type": "Feature",
                        "id": str(uuid.uuid4()),
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [polygon]
                        },
                        "properties": {
                            "name": f'{confidence:.2f}',
                            "classification": {
                                "name": label,
                                "color": self.colors[label]
                            }
                        }
                    }

                    features.append(feature)
        self.malignant_area = malignant_area
        return features

    def infer(self, loader):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        results_list = []
        coords_list = []
        pbar = tqdm(loader, desc="Training Epoch", ncols=100)
        for count, (batch, coords) in enumerate(pbar):
            with torch.no_grad():
                batch = batch.to(device, non_blocking=True)
                results = self.models[0](batch, device='0', agnostic_nms=True, iou=0.4, verbose=False)
                features = self.obtain_features(results, coords)
                results_list.append(features)
                coords_list.append(None)
        self.tissue_area = batch.shape[2] * batch.shape[3] * len(loader)
        return results_list, coords_list

    def post_process(self, results_list, coords_list, output_path):
        features = []
        for results in results_list:
            features.extend(results)
        geojson_dict = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_dict, f, indent=2, ensure_ascii=False)


def find_all_wsi_paths(wsi_root, extentions):
    """
    find the full wsi path under data_root, return a dict {slide_id: full_path}
    """
    # to support more than one ext, e.g., support .svs and .mrxs
    result = {}
    link_path = os.path.join(wsi_root, 'symlink_record.csv')
    if os.path.exists(link_path):
        df = pd.read_csv(link_path)
        all_paths = df['target'].to_list()
    else:
        all_paths = glob.glob(os.path.join(wsi_root, '**'), recursive=True)

    for ext in extentions.split(';'):
        print('Process format:', ext)
        ext = ext[1:]
        paths = [i for i in all_paths if i.split('.')[-1].lower() == ext.lower()]
        for h in paths:
            slide_name = os.path.split(h)[1]
            slide_id = '.'.join(slide_name.split('.')[0:-1])
            if slide_id != '202535080-26': continue
            result[slide_id] = h
    print("found {} wsi".format(len(result)))
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
        # elif slide_id + f'-{args.task}.geojson' in dest_files:
        #     print('geojosn file exist, skip {}'.format(slide_id))
        #     continue
        else:
            exist_idxs.append(slide_id)

    ckpts = args.ckpts.split(';')
    if args.task == 'detect':
        converter = YOLO2GeoJsonDetect(args.model, ckpts)
    elif args.task == 'segment':
        converter = YOLO2GeoJsonSegment(args.model, ckpts)
    else:
        raise ValueError('Unknown task: {}'.format(args.task))

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

        custom_transformer = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, pretrained=True, custom_transforms=custom_transformer, fast_read=True)
        if slide_file_path.endswith('.svs'):
            kwargs = {'num_workers': 8, 'pin_memory': True}
            print('Data Loader args:', kwargs)
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs, prefetch_factor=16)
        else:
            kwargs = {'num_workers': 1, 'pin_memory': True}
            print('Data Loader args:', kwargs)
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs)
        results, coords = converter.infer(loader)
        converter.post_process(results, coords, output_path)
        area_data.append({'slide_id': slide_id, 'area': f'{converter.malignant_area / converter.tissue_area * 100:.4f}%'})
        # print(f'malignant tissue area: {converter.malignant_area}; all tissue area: {converter.tissue_area}  Proportion of malignant tissue: {converter.malignant_area / converter.tissue_area * 100:.4f} %')

        print('time per slide: {:.1f}'.format(time.time() - one_slide_start))
    if area_data:
        save_area(area_data, os.path.join(args.output_dir, 'area.csv'))
    print('Time used for this dataset:{:.1f}'.format(time.time() - process_start_time))
    print('Inference ends', end='')
