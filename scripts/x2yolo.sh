cd ../

export PYTHONPATH=.:$PYTHONPATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

slide_dir=/NAS2/Data1/lbliao/Data/MXB/classification/第一批/slides
slide_ext='.kfb;.svs'
geojson_dir=/NAS2/Data1/lbliao/Data/MXB/classification/第一批/geojson
output_dir=/NAS2/Data1/lbliao/Data/MXB/classification/第一批/dataset
patch_size=512
patch_level=0
python tasks/x2yolo.py --slide_dir $slide_dir --slide_ext $slide_ext --geojson_dir $geojson_dir --output_dir $output_dir --patch_size $patch_size --patch_level $patch_level --num_workers 3
#echo --slide_dir $slide_dir --slide_ext $slide_ext --geojson_dir $geojson_dir --output_dir $output_dir --patch_size $patch_size --patch_level $patch_level --num_workers 5