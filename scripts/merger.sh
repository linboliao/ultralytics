#cd ../
export PYTHONPATH=.:$PYTHONPATH
#export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/ultralytics/lib:$LD_LIBRARY_PATH
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
model=yolo
task=detect
input_dir=/NAS3/lbliao/Data/MXB/classification/100例测试/infer2
output_dir=/NAS3/lbliao/Data/MXB/classification/100例测试/infer2

CUDA_VISIBLE_DEVICES=0 python infer/merger.py --input_dir $input_dir --output_dir $output_dir
#echo --input_dir $input_dir --output_dir $output_dir
