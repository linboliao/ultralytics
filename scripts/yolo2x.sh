cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/ultralytics/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
model=yolo
task=detect
data_coors_dir=/NAS2/Data1/lbliao/Data/MXB/classification/第二批/patches_0_1024
data_slide_dir=/NAS2/Data1/lbliao/Data/MXB/classification/第二批/slides
ckpts='runs/detect/yolo11s_0512/weights/best.pt;runs/detect/yolo11s_0702/weights/best.pt;runs/detect/pki/weights/best.pt;runs/detect/cbam/weights/best.pt'
slide_ext='.kfb;.svs'
batch_size=16
output_dir=/NAS2/Data1/lbliao/Data/MXB/classification/第二批/yolo_inference

CUDA_VISIBLE_DEVICES=6 python infer/yolo2x.py --data_coors_dir $data_coors_dir --data_slide_dir $data_slide_dir --ckpts $ckpts --slide_ext $slide_ext --batch_size $batch_size --model $model --output_dir $output_dir --task $task
#echo --data_coors_dir $data_coors_dir --data_slide_dir $data_slide_dir --ckpts $ckpts --slide_ext $slide_ext --batch_size $batch_size --model $model --output_dir $output_dir --task $task


#model=yolo
#task=segment
#data_coors_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治4/patches_0_1024
#data_slide_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治4/test
#ckpts='runs/segment/yolo12n/weights/best.pt'
#slide_ext='.kfb;.svs'
#batch_size=32
#output_dir=/NAS2/Data1/lbliao/Data/MXB/gleason/根治4/yolo_infer
#
#CUDA_VISIBLE_DEVICES=7 python infer/yolo2x.py --data_coors_dir $data_coors_dir --data_slide_dir $data_slide_dir --ckpts $ckpts --slide_ext $slide_ext --batch_size $batch_size --model $model --output_dir $output_dir --task $task
#echo --data_coors_dir $data_coors_dir --data_slide_dir $data_slide_dir --ckpts $ckpts --slide_ext $slide_ext --batch_size $batch_size --model $model --output_dir $output_dir --task $task