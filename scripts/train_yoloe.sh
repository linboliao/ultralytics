cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/ultralytics/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

ckpt=ckpt/yoloe-11m-seg-pf.pt
data=tasks/cfg/datasets/test.yaml
epoches=200
patience=50
image_size=1024
gpu_ids='5,6,7'
batch=3
lr=0.01

python tasks/train_yoloe.py --ckpt $ckpt --data $data --epoches $epoches --patience $patience --image_size $image_size --gpu_ids $gpu_ids --batch $batch --lr0 $lr
#echo --ckpt $ckpt --data $data --epoches $epoches --image_size $image_size --gpu_ids $gpu_ids --batch $batch --lr0 $lr