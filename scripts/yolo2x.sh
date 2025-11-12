#cd ../
export PYTHONPATH=.:$PYTHONPATH
#export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/ultralytics/lib:$LD_LIBRARY_PATH
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
#model=yolo
#task=detect
#data_coors_dir=/NAS3/lbliao/Data/MXB/classification/100例测试/patches_0_2048
#data_slide_dir=/NAS3/lbliao/Data/MXB/classification/100例测试/癌
#ckpts='/NAS3/lbliao/Code/ultralytics/runs/detect/yolo11s_0512/weights/best.pt;/NAS3/lbliao/Code/ultralytics/runs/detect/yolo11s_0702/weights/best.pt;/NAS3/lbliao/Code/ultralytics/runs/detect/cbam/weights/best.pt;/NAS3/lbliao/Code/ultralytics/runs/detect/pki/weights/best.pt'
#slide_ext='.kfb;.svs'
#batch_size=16
#output_dir=/NAS3/lbliao/Data/MXB/classification/100例测试/infer2
#
#CUDA_VISIBLE_DEVICES=0 python infer/yolo2x.py --data_coors_dir $data_coors_dir --data_slide_dir $data_slide_dir --ckpts $ckpts --slide_ext $slide_ext --batch_size $batch_size --model $model --output_dir $output_dir --task $task
#echo --data_coors_dir $data_coors_dir --data_slide_dir $data_slide_dir --ckpts $ckpts --slide_ext $slide_ext --batch_size $batch_size --model $model --output_dir $output_dir --task $task


model=yolo
task=segment
data_coors_dir=/NAS3/lbliao/Data/MXB/classification/100例测试/patches_0_1024
data_slide_dir=/NAS3/lbliao/Data/MXB/classification/100例测试/癌
ckpts='/NAS3/lbliao/Code/ultralytics/runs/segment/nc3/yolo12s/weights/best.pt'
slide_ext='.kfb;.svs'
batch_size=32
output_dir=/NAS3/lbliao/Data/MXB/classification/100例测试/infer2

CUDA_VISIBLE_DEVICES=0 python infer/yolo2x.py --data_coors_dir $data_coors_dir --data_slide_dir $data_slide_dir --ckpts $ckpts --slide_ext $slide_ext --batch_size $batch_size --model $model --output_dir $output_dir --task $task
#echo --data_coors_dir $data_coors_dir --data_slide_dir $data_slide_dir --ckpts $ckpts --slide_ext $slide_ext --batch_size $batch_size --model $model --output_dir $output_dir --task $task