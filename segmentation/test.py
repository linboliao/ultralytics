import argparse

from segmentation.result import *

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='/data2/lbliao/Code/ultralytics/runs/detect/Pconv2/weights/best.pt')
parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/0318/', help='patch directory')
parser.add_argument('--gpu', type=str, default='7', help='patch directory')
parser.add_argument('--slide_dir', type=str, default='', help='patch directory')
parser.add_argument('--output_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/0318/result/Pconv2', help='output directory')
parser.add_argument('--patch_size', type=int, default=2048, help='patch size')
parser.add_argument('--infer_size', type=int, default=1536, help='patch size')
parser.add_argument('--slide_list', type=list)#, default=['202468220.15.kfb','202467227.7.kfb', '202467227.8.kfb','202467810.39.kfb', '202467810.45.46.kfb', '202467810.51.kfb'])
if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    GeoResults(args).parallel_run()
    # TiffResults(args).parallel_run()
    # MdsResults(args).parallel_run()
    # LMResults(args).parallel_run()
