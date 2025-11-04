#cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/ultralytics/lib:$LD_LIBRARY_PATH
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

model=yolo # yolo yoloe rtdetr yoloworld
ckpt=tasks/cfg/models/yolo12-seg-msc.yaml
data=tasks/cfg/datasets/segment.yaml
epoches=500
patience=100
image_size=1024
gpu_ids='0'
batch=8
lr=0.01
name=yolo12s-msc
CUDA_VISIBLE_DEVICES=0 python tasks/train.py --model $model --ckpt $ckpt --data $data --epoches $epoches --image_size $image_size --gpu_ids $gpu_ids --batch $batch --lr0 $lr --name $name --resume --no_amp
#echo --model $model --ckpt $ckpt --data $data --epoches $epoches --image_size $image_size --gpu_ids $gpu_ids --batch $batch --lr0 $lr --name $name --resume --no_amp