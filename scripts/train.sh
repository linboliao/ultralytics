cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/ultralytics/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

model=yoloe # yolo yoloe rtdetr yoloworld
ckpt=ckpt/yoloe-11m-seg-pf.pt
data=tasks/cfg/datasets/segment.yaml
epoches=300
patience=100
image_size=1024
gpu_ids='1'
batch=1
lr=0.01
name=yolo11m
python tasks/train.py --model $model --ckpt $ckpt --data $data --epoches $epoches --image_size $image_size --gpu_ids $gpu_ids --batch $batch --lr0 $lr --name $name
#echo --ckpt $ckpt --data $data --epoches $epoches --image_size $image_size --gpu_ids $gpu_ids --batch $batch --lr0 $lr --name $name