import argparse

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str)
parser.add_argument('--data', type=str)
parser.add_argument('--epoches', type=int)
parser.add_argument('--patience', type=int)
parser.add_argument('--image_size', type=int)
parser.add_argument('--gpu_ids', type=str)
parser.add_argument('--batch', type=int)
parser.add_argument('--lr0', type=float)
args = parser.parse_args()

model = YOLOE(args.ckpt)

results = model.train(data=args.data, epochs=args.epoches, imgsz=args.image_size, device=args.gpu_ids, batch=args.batch, patience=args.patience, lr0=args.lr0, trainer=YOLOEPESegTrainer)
