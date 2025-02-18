import argparse
import os
import sys
import traceback

import cv2
import numpy as np
import openslide
import pyvips
import torch
from PIL import Image
from loguru import logger

from ultralytics import YOLO

sys.path.insert(0, r'/data2/lbliao/Code/aslide')
from aslide import Aslide

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str)
parser.add_argument('--output_dir', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Repeat/result/')
parser.add_argument('--gpu_ids', type=str)
parser.add_argument('--data_root', type=str, default='/NAS2/Data1/lbliao/Data/MXB/Repeat/2024-12-10/')
parser.add_argument('--patch_size', type=int, default=2048)
parser.add_argument('--patch_level', type=int, default=0)
parser.add_argument('--overlap', type=float, default=0)
parser.add_argument('--slide_dir', type=str)

args = parser.parse_args()
args.slide_list = ['202303007A2.kfb']

model = YOLO(args.ckpt)


def is_background(img, threshold=5):
    img_array = np.array(img)
    pixel_max = np.max(img_array, axis=2)
    pixel_min = np.min(img_array, axis=2)
    difference = pixel_max - pixel_min
    tissue_percent = np.sum(difference > threshold) / (img_array.shape[0] * img_array.shape[1])

    return tissue_percent < 0.05


class Patch2WSI:
    def __init__(self, opt):
        self.data_root = opt.data_root
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, 'slides')
        self.output_dir = opt.output_dir if opt.output_dir else os.path.join(opt.data_root, f'cut')

        self.overlap = opt.overlap
        self.patch_size = opt.patch_size
        self.patch_level = opt.patch_level
        self.slide_list = opt.slide_list
        os.makedirs(self.output_dir, exist_ok=True)

    def convert(self, slide):
        base, ext = os.path.splitext(slide)
        slide_path = os.path.join(self.slide_dir, slide)
        wsi = Aslide(slide_path) if ext == '.kfb' else openslide.open_slide(slide_path)
        [w, h] = wsi.level_dimensions[self.patch_level]
        canvas = np.full([h, w, 3], (255, 255, 255), dtype=np.uint8)
        step = int(self.patch_size * (1 - self.overlap))
        for w_s in range(0, w, step):
            for h_s in range(0, h, step):
                img_real = wsi.read_region((w_s, h_s), self.patch_level, (min(self.patch_size, w - w_s), min(self.patch_size, h - h_s)))
                if isinstance(img_real, Image.Image):
                    img_real = img_real.convert('RGB')
                    input_img = img_real.resize((1024, 1024))
                else:
                    input_img = cv2.resize(img_real, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

                if is_background(img_real):
                    canvas[h_s:h_s + self.patch_size, w_s:w_s + self.patch_size] = np.array(img_real)
                else:
                    results = model(input_img, device=args.gpu_ids)
                    for result in results:
                        result.save(filename="result.jpg", )
                        output_img = cv2.imread("result.jpg")
                        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                        output_img = cv2.resize(output_img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
                    patch = canvas[h_s:h_s + self.patch_size, w_s:w_s + self.patch_size]
                    index_overlap = np.where(patch != 255)
                    output_img[index_overlap] = output_img[index_overlap] * 0.5 + output_img[index_overlap] * 0.5
                    canvas[h_s:h_s + self.patch_size, w_s:w_s + self.patch_size] = output_img.astype(np.uint8)[0:min(self.patch_size, h - h_s), 0:min(self.patch_size, w - w_s)]
        try:
            output_path = os.path.join(self.output_dir, f'{base}.tif')
            img = pyvips.Image.new_from_array(canvas)
            img = img.colourspace("srgb")
            img = img.copy(interpretation="srgb")
            img.tiffsave(output_path, compression="jpeg", tile=True, tile_width=1024, tile_height=1024,
                         pyramid=True, bigtiff=True, Q=30, rgbjpeg=True)
        except Exception as e:
            traceback.print_exc()
        finally:
            output_path = os.path.join(self.output_dir, f'{base}.png')
            canvas = Image.fromarray(canvas)
            # canvas.thumbnail((w // 10, h // 10))
            canvas.save(output_path)

    def save_fake_patch(self, slide):
        target_patch_size = 10000
        base, ext = os.path.splitext(slide)
        slide_path = os.path.join(self.slide_dir, slide)
        wsi = Aslide(slide_path) if ext == '.kfb' else openslide.open_slide(slide_path)
        [w, h] = wsi.level_dimensions[self.patch_level]
        logger.info(f'{slide} width {w} height {h}')
        for h_s in range(0, h, target_patch_size):
            for w_s in range(0, w, target_patch_size):
                output_path = os.path.join(self.output_dir, f'{base}_{w_s}_{h_s}.png')
                canvas = np.full([target_patch_size, target_patch_size, 3], (255, 255, 255), dtype=np.uint8)
                step = int(self.patch_size * (1 - self.overlap))
                h_i = 0
                while h_i < target_patch_size:
                    if h - h_s - h_i <= 0:
                        h_i += step
                        continue
                    w_i = 0
                    while w_i < target_patch_size:
                        if w - w_s - w_i <= 0:
                            w_i += step
                            continue
                        img_real = wsi.read_region((w_i + w_s, h_i + h_s), self.patch_level, (min(self.patch_size, w - w_s - w_i), min(self.patch_size, h - h_s - h_i)))
                        if isinstance(img_real, Image.Image):
                            img_real = img_real.convert('RGB')

                        data = {
                            'inst': torch.tensor([0]),
                            'image': torch.tensor([0]),
                            'feat': torch.tensor([0]),
                            'path': 'None',
                            'label': transform(img_real).unsqueeze(0)
                        }
                        h_step = min(min(self.patch_size, target_patch_size - h_i), h - h_s - h_i)
                        w_step = min(min(self.patch_size, target_patch_size - w_i), w - w_s - w_i)
                        if is_background(img_real):
                            canvas[h_i:h_i + h_step, w_i:w_i + w_step] = np.array(img_real)[0:h_step, 0:w_step]
                        else:
                            generated = model.inference(data['label'], data['inst'], data['image'])
                            img_fake = util.tensor2im(generated.data[0])
                            fake_patch = canvas[h_i:h_i + h_step, w_i:w_i + w_step]
                            index_overlap = np.where(fake_patch != 255)
                            img_fake[index_overlap] = img_fake[index_overlap] * 0.5 + fake_patch[index_overlap] * 0.5
                            canvas[h_i:h_i + h_step, w_i:w_i + w_step] = np.array(img_fake)[0:h_step, 0:w_step]
                        w_i += step
                    h_i += step
                canvas = Image.fromarray(canvas)
                canvas.save(output_path)

    @property
    def slides(self):
        if self.slide_list:
            return self.slide_list
        else:
            return [f for f in os.listdir(self.slide_dir) if os.path.isfile(os.path.join(self.slide_dir, f))]

    def run(self):
        for slide in self.slides:
            self.convert(slide)
            # self.save_fake_patch(slide)
            logger.info(f'{slide} processed')


if __name__ == '__main__':
    Patch2WSI(args).run()
