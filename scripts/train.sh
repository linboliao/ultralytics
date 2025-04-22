export PYTHONPATH=/data2/lbliao/Code/ultralytics:$PYTHONPATH
python ../segmentation/train.py --ckpt /home/jing/linboliao/code/ultralytics/segmentation/cfg/model/yolo11-MM.yaml --data /home/jing/linboliao/code/ultralytics/segmentation/cfg/data/Pconv.yaml --epoches 1000 --image_size 1024 --gpu_ids '0,1,2,3,4,5' --batch 12 --lr0 0.001
