import argparse
import os

import geopandas as gpd
from shapely.geometry import Polygon


def merge_geojson(a_path, b_path, output_path):
    gdf_a = gpd.read_file(a_path)
    gdf_b = gpd.read_file(b_path)

    if gdf_a.crs != gdf_b.crs:
        gdf_b = gdf_b.to_crs(gdf_a.crs)

    result_features = []

    for idx_a, row_a in gdf_a.iterrows():
        geom_a = row_a.geometry
        if not isinstance(geom_a, Polygon):
            continue

        intersects_b = gdf_b[gdf_b.geometry.intersects(geom_a)]

        if not intersects_b.empty:
            first_b = intersects_b.iloc[0]
            new_geom = first_b.geometry  # 保留b的轮廓
            new_attrs = row_a.drop('geometry').to_dict()  # 继承a的属性
            result_features.append({**new_attrs, 'geometry': new_geom})
        else:
            result_features.append(row_a.to_dict())

    result_gdf = gpd.GeoDataFrame(result_features, geometry='geometry', crs=gdf_a.crs)
    result_gdf.to_file(output_path, driver='GeoJSON')
    print(f"合并完成，结果保存至：{output_path}")


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help='patch directory')
parser.add_argument('--output_dir', type=str, help='output directory')
args = parser.parse_args()
if __name__ == "__main__":
    for file in os.listdir(args.input_dir):

        if file.endswith('-detect.geojson'):
            a_path = os.path.join(args.input_dir, file)
            b_path = os.path.join(args.output_dir, file.replace('-detect.geojson', '-segment.geojson'))
            if os.path.exists(b_path):
                output_path = os.path.join(args.output_dir, file.replace('-detect.geojson', '.geojson'))
                merge_geojson(a_path, b_path, output_path)
