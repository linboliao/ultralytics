import argparse
import os

from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import TQDM, YAML, IterableSimpleNamespace, DEFAULT_CFG_PATH
from ultralytics.utils.plotting import plot_images


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--task', type=str, default='segment', help='segment, detect')
parser.add_argument('--imgsz', type=int, default=512)
parser.add_argument('--phase', type=str, default='train')

args = parser.parse_args()

if __name__ == "__main__":
    DEFAULT_CFG_DICT = YAML.load(DEFAULT_CFG_PATH)
    DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
    DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.task = args.task
    data = check_det_dataset(args.data)
    mode = 'val'
    batch = 1
    for img_path in data[args.phase]:
        dataset = build_yolo_dataset(cfg, img_path, batch, data, mode=mode, rect=mode == "val")
        dataloader = build_dataloader(dataset, batch=batch, workers=4)
        output_dir = img_path.replace("/images", f"/{args.task}_vis")
        os.makedirs(output_dir, exist_ok=True)
        bar = TQDM(dataloader, total=len(dataloader))
        try:
            for batch_i, batch in enumerate(bar):
                plot_images(
                    labels=batch,
                    paths=batch["im_file"],
                    fname=batch["im_file"][0].replace("/images/", f"/{args.task}_vis/"),
                    names=None,
                )
        except Exception as e:
            print(e)