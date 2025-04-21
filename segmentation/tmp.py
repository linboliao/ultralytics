import json
import os
import random
import shutil
import uuid

import torch
import torchvision

contour_path = '/NAS2/Data1/lbliao/Data/MXB/123/2013350N_有癌_2024-11-01_11_41_54-2.geojson'
cells_path = '/NAS2/Data1/lbliao/Data/MXB/123/2013350N_有癌_2024-11-01_11_41_54.geojson'
with open(contour_path, 'r', encoding='utf-8') as file:
    contour_anns = json.load(file)
cont_features = contour_anns.get('features')
with open(cells_path, 'r', encoding='utf-8') as file:
    cell_anns = json.load(file)
cell_features = cell_anns.get('features')
cell_data = []
boxes = []
scores = []
for feature in cell_features:
    if feature['geometry']['type'] == 'Point':
        cell_coords = feature['geometry']['coordinates']
        flag = False
        for cnt_feature in cont_features:
            if cnt_feature['geometry']['type'] == 'Polygon':
                cnt_coords = cnt_feature['geometry']['coordinates'][0]
                if cnt_coords[0][0] < cell_coords[0] < cnt_coords[1][0] and cnt_coords[1][1] < cell_coords[1] < cnt_coords[2][1]:
                    flag = True
        if not flag:
            h, w = random.randint(33, 47), random.randint(33, 47)
            cell_data.append(
                [[cell_coords[0] - h, cell_coords[1] - w], [cell_coords[0] - h, cell_coords[1] + w], [cell_coords[0] + h, cell_coords[1] + w], [cell_coords[0] + h, cell_coords[1] - w], [cell_coords[0] - h, cell_coords[1] - w]])
            boxes.append([cell_coords[0] - h, cell_coords[1] - w, cell_coords[0] + h, cell_coords[1] + w])
            scores.append(0.6)
boxes = torch.tensor(boxes, dtype=torch.float32)
scores = torch.tensor(scores, dtype=torch.float32)
i = torchvision.ops.nms(boxes, scores, 0.15)  # NMS
index = i.tolist()
cell_data = [cell_data[i] for i in index if 0 <= i < len(cell_data)]
for cell in cell_data:
    feature = {
        "type": "Feature",
        "id": str(uuid.uuid4()),
        "geometry": {
            "type": "Polygon",
            "coordinates": [cell],
        }
    }
    cont_features.append(feature)
geojson = {
    "type": "FeatureCollection",
    "features": cont_features
}

with open(os.path.join('/NAS2/Data1/lbliao/Data/MXB/123/', f'result.geojson'), 'w') as f:
    json.dump(geojson, f, indent=2)
    print(f'generated result.geojson contour json!!!')

# slides = os.listdir('/NAS2/Data1/lbliao/Data/MXB/Detection/0401/slides')
# slides = [slide.replace('.svs', '') for slide in slides]
# for slide in slides:
#     cells_path = f'/NAS2/Data1/lbliao/Data/MXB/Detection/0401/cellvit/{slide}/cell_detection/cells.geojson'
#     if os.path.exists(cells_path):
#         new_path = f'/NAS2/Data1/lbliao/Data/MXB/Detection/0401/results/{slide}.geojson'
#         shutil.copy(cells_path, new_path)
