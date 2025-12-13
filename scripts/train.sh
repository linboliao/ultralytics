cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/ultralytics/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

model=yolo # yolo yoloe rtdetr yoloworld
ckpt=tasks/cfg/models/msc_v1.yaml
data=tasks/cfg/datasets/crag.yaml
epoches=1000
patience=100
image_size=1024
gpu_ids='0,1,2,3,4,5'
batch=6
name=12m-msc-v3
project=runs/crag
#python tasks/train.py --model $model --ckpt $ckpt --data $data --epoches $epoches --patience $patience --image_size $image_size --gpu_ids $gpu_ids --batch $batch --name $name --resume --project $project # --no_amp
echo --model $model --ckpt $ckpt --data $data --epoches $epoches --image_size $image_size --gpu_ids $gpu_ids --batch $batch --name $name --resume --project $project # --no_amp