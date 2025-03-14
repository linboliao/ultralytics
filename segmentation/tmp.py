import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/data2/lbliao/Code/ultralytics/runs/detect/train5/weights/best.pt')  # select your model.pt path
    model.predict(source='/NAS2/Data1/lbliao/Data/MXB/0307/dataset/2048/val/images/1638897.11_7680_9216.png',
                  imgsz=2048,
                  project='runs/detect/feature',
                  name='test',
                  save=True,
                  device='6',
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  visualize=True,  # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )