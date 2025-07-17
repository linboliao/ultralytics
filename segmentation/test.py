import warnings

warnings.filterwarnings("ignore")

import argparse

from segmentation.result import *
from segmentation.new_result import *

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=list, default=[
    '/data2/lbliao/Code/ultralytics/runs/detect/Pconv/weights/best.pt',
    '/data2/lbliao/Code/ultralytics/runs/detect/2048-1536-2/weights/best.pt',
    '/data2/lbliao/Code/ultralytics/runs/detect/yolov11/weights/best.pt',
    '/data2/lbliao/Code/ultralytics/runs/detect/cbam/weights/best.pt',
    '/data2/lbliao/Code/ultralytics/runs/detect/pki/weights/best.pt'
])
parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/seminal', help='patch directory')
parser.add_argument('--gpu', type=str, default='0', help='patch directory')
parser.add_argument('--slide_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/segment/0716/slides', help='patch directory')
parser.add_argument('--output_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/segment/0716/yolo_detect', help='output directory')
parser.add_argument('--patch_size', type=int, default=2048, help='patch size')
parser.add_argument('--infer_size', type=int, default=1536, help='patch size')
parser.add_argument('--slide_list', type=list, default=[])
parser.add_argument('--slide', type=str, default='', help='patch directory')
parser.add_argument('--show_level', type=int, default=0)

if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # GeoResults(args).parallel_run()
    MultiGeoResults(args).parallel_run()

    # GeoJSONProcessor(args).parallel_process()
    # TiffResults(args).parallel_run()
    # PicResults(args).parallel_run()
    # MdsResults(args).parallel_run()
    # LMResults(args).parallel_run()
    # KVResults(args).parallel_run()
