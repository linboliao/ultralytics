# import os
#
# from ultralytics import YOLO
#
# # Load a model
# model = YOLO("dvclive/artifacts/best.pt")
#
# img_dir = "/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/yolo/val/images/"
# save_dir = "/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/yolo/test/"
# images = os.listdir(img_dir)
# for img in images:
#     results = model(os.path.join(img_dir, img), device="6")
#     # Process results list
#     for result in results:
#         boxes = result.boxes  # Boxes object for bounding box outputs
#         masks = result.masks  # Masks object for segmentation masks outputs
#         keypoints = result.keypoints  # Keypoints object for pose outputs
#         probs = result.probs  # Probs object for classification outputs
#         obb = result.obb  # Oriented boxes object for OBB outputs
#         # result.show()  # display to screen
#         result.save(filename=os.path.join(save_dir, img))  # save to disk

import json
import os

import argparse
import shutil

from PIL import Image

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str)
parser.add_argument('--test_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--gpu_ids', type=str)
args = parser.parse_args()

model = YOLO(args.ckpt)
label_dict = {0: 'prostate', 1: 'cancer'}

os.makedirs(args.output_dir, exist_ok=True)
images = os.listdir(args.test_dir)
for img in images:
    base, _ = os.path.splitext(img)
    img_path = os.path.join(args.test_dir, img)
    img_f = Image.open(img_path)
    results = model(img_path, device=args.gpu_ids)
    # Process results list
    shapes = []
    flag = False
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen
        result.save(filename=os.path.join(args.output_dir, img))  # save to disk
        # boxes = result.boxes  # Boxes object for bounding box outputs
        # indexes = []
        # for i, d1 in enumerate(reversed(boxes)):
        #     for j, d2 in enumerate(reversed(boxes)):
        #         if j <= i:
        #             continue
        #         x, y, w, h = d1.xywh[0]
        #         x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        #         x, y, w, h = d2.xywh[0]
        #         x3, y3, x4, y4 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        #         if x1 * 0.95 <= x3 and y1 * 0.95 <= y3 and x2 * 1.05 >= x4 and y2 * 1.05 >= y4 and d1.cls == d2.cls:
        #             indexes.append(j)
        #         elif x1 >= x3 * 0.95 and y1 >= y3 * 0.95 and x2 <= x4 * 1.05 and y2 <= y4 * 1.05 and d1.cls == d2.cls:
        #             indexes.append(i)
    #
    #     for i, box in enumerate(reversed(boxes)):
    #         # if i in indexes:
    #         #     continue
    #         [x1, y1, x2, y2] = box.xyxy.tolist()[0]
    #         points = [[x1, y1], [x2, y2]]
    #         shapes.append({
    #             "label": label_dict[int(box.cls.tolist()[0])],
    #             "points": points,
    #             "group_id": None,
    #             "description": "",
    #             "shape_type": "rectangle",
    #             "flags": {},
    #             "mask": None
    #         })
    #         if label_dict[int(box.cls.tolist()[0])] == 'cancer':
    #             flag = True
    # ann = {
    #     "version": "5.6.0",
    #     "flags": {},
    #     "shapes": shapes,
    #     "imagePath": img,
    #     "imageData": None,
    #     "imageHeight": img_f.height,
    #     "imageWidth": img_f.width,
    # }
    # if not flag :
    #     continue
    # with open(os.path.join(args.output_dir, f'{base}.json'), 'w') as f:
    #     json.dump(ann, f, indent=2)
    # shutil.copy(img_path, os.path.join(args.output_dir, img))
