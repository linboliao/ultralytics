import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


import argparse



from segmentation.result import *

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=list, default=['/NAS2/Data1/lbliao/Code/ultralytics/runs/detect/0512/weights/best.pt'])
parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Detection/0224', help='patch directory')
parser.add_argument('--gpu', type=str, default='4', help='patch directory')
parser.add_argument('--slide_dir', type=str, default='', help='patch directory')
parser.add_argument('--output_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Detection/0224/result', help='output directory')
parser.add_argument('--patch_size', type=int, default=2048, help='patch size')
parser.add_argument('--infer_size', type=int, default=1536, help='patch size')
parser.add_argument('--slide_list', type=list)#, default=['202454720.31.kfb', '202453379.60.kfb', '202454720.34.kfb'])
parser.add_argument('--slide', type=str, default='1834976T.svs', help='patch directory')
parser.add_argument('--show_level', type=int, default=0)

if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # GeoResults(args).parallel_run()
    MultiGeoResults(args).parallel_run()
    # TiffResults(args).parallel_run()
    # MdsResults(args).parallel_run()
    # LMResults(args).parallel_run()
    # KVResults(args).parallel_run()
