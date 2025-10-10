cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/ultralytics/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

model=yolo # yolo yoloe rtdetr yoloworld
ckpt=runs/segment/yolo11m/weights/best.pt
data=tasks/cfg/datasets/segment.yaml
phase=val
name=yolo11m_$phase
CUDA_VISIBLE_DEVICES=3 python tasks/test.py --model $model --ckpt $ckpt --data $data --phase $phase --name $name
#echo --model $model --ckpt $ckpt --data $data --phase $phase --name $name