from ultralytics import YOLO

model = YOLO('runs/mx/12n-msc-v3/weights/best.pt')

results = model.predict(
    source='/NAS145/liaolinbo/Data/GlandSeg/maixin-2/dataset/train/images/train_slides_01_116.png',
    visualize=True,
    project='test_runs/mx2',
    name='12n-msc-vis',
    exist_ok=True
)