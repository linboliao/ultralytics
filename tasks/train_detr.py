import argparse

from ultralytics import RTDETR

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str)
parser.add_argument('--data', type=str)
parser.add_argument('--epoches', type=int)
parser.add_argument('--image_size', type=int)
parser.add_argument('--gpu_ids', type=str)
parser.add_argument('--batch', type=int)
parser.add_argument('--lr0', type=float)
args = parser.parse_args()

model = RTDETR(args.ckpt)

results = model.train(data=args.data, epochs=args.epoches, imgsz=args.image_size, device=args.gpu_ids, batch=args.batch, patience=200, lr0=args.lr0)
