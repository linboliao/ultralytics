import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str)
parser.add_argument('--data', type=str)
parser.add_argument('--epoches', type=int)
parser.add_argument('--image_size', type=int)
parser.add_argument('--gpu_ids', type=str)
parser.add_argument('--batch', type=int)
parser.add_argument('--lr0', type=float)
args = parser.parse_args()

# Load a model
model = YOLO(args.ckpt)

# Train the model
results = model.train(data=args.data, epochs=args.epoches, imgsz=args.image_size, device=args.gpu_ids, batch=args.batch, patience=500, lr0=args.lr0)
