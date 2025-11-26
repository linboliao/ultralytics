cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/ultralytics/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

model=yolo # yolo yoloe rtdetr yoloworld
ckpt=runs/segment/nc2/12s-msc_v1_1000_cos/weights/best.pt
data=tasks/cfg/datasets/segment.yaml
phase='test'
name=12s-msc_v1_cos
project=test_runs/segment/nc2
CUDA_VISIBLE_DEVICES=0 python tasks/test.py --model $model --ckpt $ckpt --data $data --phase $phase --name $name --project $project

#ckpt=runs/segment/nc2/yolo12s/weights/best.pt
#data=tasks/cfg/datasets/segment.yaml
#phase='test'
#name=yolo12s
#project=test_runs/segment/nc2
##CUDA_VISIBLE_DEVICES=1 python tasks/test.py --model $model --ckpt $ckpt --data $data --phase $phase --name $name --project $project
#
#ckpt=runs/segment/nc2/yolo12s-mscv3/weights/best.pt
#data=tasks/cfg/datasets/segment.yaml
#phase='test'
#name=yolo12s-mscv
#project=test_runs/segment/nc2
##CUDA_VISIBLE_DEVICES=2 python tasks/test.py --model $model --ckpt $ckpt --data $data --phase $phase --name $name --project $project
#
#
##echo --model $model --ckpt $ckpt --data $data --phase $phase --name $name