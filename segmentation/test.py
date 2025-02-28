import argparse

from segmentation.result import GeoResults

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='/NAS2/Data1/lbliao/Code/ultralytics/runs/detect/train3/weights/last.pt')
parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Repeat/2024-12-10/', help='patch directory')
parser.add_argument('--gpu', type=str, default='7', help='patch directory')
parser.add_argument('--slide_dir', type=str, default='', help='patch directory')
parser.add_argument('--output_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Repeat/result/', help='output directory')
parser.add_argument('--patch_size', type=int, default=2048, help='patch size')
parser.add_argument('--infer_size', type=int, default=1536, help='patch size')
parser.add_argument('--slide_list', type=list, default=['202303007A2.kfb'])
if __name__ == '__main__':
    args = parser.parse_args()
    GeoResults(args).parallel_run()