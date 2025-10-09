cd ../

export PYTHONPATH=.:$PYTHONPATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

data=tasks/cfg/datasets/test.yaml
task=segment
imgsz=1024
phase=val
python tasks/utils/visualize.py --data $data  --task $task --imgsz $imgsz --phase $phase
#echo --data $data  --task $task --imgsz $imgsz --phase $phase