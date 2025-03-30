import argparse

from segmentation.result import *

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='')
parser.add_argument('--data_root', type=str, default='', help='patch directory')
parser.add_argument('--gpu', type=str, default='7', help='patch directory')
parser.add_argument('--slide_dir', type=str, default='', help='patch directory')
parser.add_argument('--output_dir', type=str, default='', help='output directory')
parser.add_argument('--patch_size', type=int, default=2048, help='patch size')
parser.add_argument('--infer_size', type=int, default=1536, help='patch size')
parser.add_argument('--slide_list', type=list)#, default=['1920233C2024-11-01_13_26_37.kfb'])
if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # GeoResults(args).parallel_run()
    # TiffResults(args).parallel_run()
    # MdsResults(args).parallel_run()
    # LMResults(args).parallel_run()
    KVResults(args).parallel_run()
