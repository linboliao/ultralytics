cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/ultralytics/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

model=yolo
ckpt=runs/mx/12n-msc-v3/weights/best.pt
data=tasks/cfg/datasets/segment.yaml
phase='test'
name=12n-msc-v3
project=test_runs/mx
#CUDA_VISIBLE_DEVICES=7 python tasks/test.py --model $model --ckpt $ckpt --data $data --phase $phase --name $name --project $project
phase='val'
CUDA_VISIBLE_DEVICES=7 python tasks/test.py --model $model --ckpt $ckpt --data $data --phase $phase --name $name --project $project
phase='train'
#CUDA_VISIBLE_DEVICES=7 python tasks/test.py --model $model --ckpt $ckpt --data $data --phase $phase --name $name --project $project
#echo --model $model --ckpt $ckpt --data $data --phase $phase --name $name --project $project
