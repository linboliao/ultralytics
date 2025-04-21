import json
import os
import uuid

slides = os.listdir(f'/NAS2/Data1/lbliao/Data/MXB/Detection/0401/slides')
slides = [slide.replace('.svs', '') for slide in slides]
for slide in slides:
    cells_path = f'/NAS2/Data1/lbliao/Data/MXB/Detection/0401/cellvit/{slide}/cell_detection/cells.geojson'
    if not os.path.exists(cells_path):
        continue
    cont_features = []
    with open(cells_path, 'r', encoding='utf-8') as file:
        features = json.load(file)
    for feature in features:
        cell_coords = feature['geometry']['coordinates']
        for coords in cell_coords:
            cont_features.append({
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coords,
                }
            })
    geojson = {
        "type": "FeatureCollection",
        "features": cont_features
    }

    with open(f'/NAS2/Data1/lbliao/Data/MXB/Detection/0401/results/{slide}.geojson', 'w') as f:
        json.dump(geojson, f, indent=2)
        print(f'generated {slide}.geojson contour json!!!')