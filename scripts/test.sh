cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/ultralytics/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

model=yolo # yolo yoloe rtdetr yoloworld
ckpt=runs/segment/yolo12s-msc/weights/best.pt
data=tasks/cfg/datasets/segment.yaml
phase='test'
name=yolo12s-msc_$phase
project=test_runs/segment
CUDA_VISIBLE_DEVICES=0 python tasks/test.py --model $model --ckpt $ckpt --data $data --phase $phase --name $name --project $project
#echo --model $model --ckpt $ckpt --data $data --phase $phase --name $name