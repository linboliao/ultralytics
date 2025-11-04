import argparse

from ultralytics import YOLO, YOLOE, RTDETR, YOLOWorld
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=['yolo', 'rtdetr', 'yoloe', 'yoloworld'], help='Model type to train: yolo, rtdetr, or yoloe')
parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint file')
parser.add_argument('--data', type=str, required=True, help='Path to data configuration file')
parser.add_argument('--epoches', type=int, help='Number of training epochs')
parser.add_argument('--image_size', type=int, help='Image size for training')
parser.add_argument('--gpu_ids', type=str, help='GPU device ID(s) to use (e.g., 0 or 0,1,2)')
parser.add_argument('--batch', type=int, help='Training batch size')
parser.add_argument('--lr0', type=float, help='Initial learning rate')
parser.add_argument('--name', type=str, help='Training experiment name')
parser.add_argument('--resume', action='store_true', help='Resume the training')
parser.add_argument('--patience', type=int, help='Early stopping patience')
parser.add_argument('--no_amp', action='store_false', help='close amp')
args = parser.parse_args()
trainer = None
if args.model == 'yolo':
    model = YOLO(args.ckpt)
elif args.model == 'rtdetr':
    model = RTDETR(args.ckpt)
elif args.model == 'yoloe':
    trainer = YOLOEPESegTrainer
    model = YOLOE(args.ckpt)
elif args.model == 'yoloworld':
    model = YOLOWorld(args.ckpt)
else:
    raise ValueError(f"Unsupported model type: {args.model}")

results = model.train(data=args.data, epochs=args.epoches, imgsz=args.image_size, device=args.gpu_ids, batch=args.batch, patience=args.patience, lr0=args.lr0, trainer=trainer, name=args.name, resume=args.resume, amp=args.no_amp)
