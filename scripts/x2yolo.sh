cd ../

export PYTHONPATH=.:$PYTHONPATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

patch_size=1024
patch_level=0

slide_ext='.kfb;.svs'
slide_dir=/NAS145/liaolinbo/Data/GlandSeg/gleason/slides
geojson_dir=/NAS145/liaolinbo/Data/GlandSeg/gleason/geojson
output_dir=/NAS145/liaolinbo/Data/GlandSeg/gleason/dataset

python tasks/utils/x2yolo.py --slide_dir $slide_dir --slide_ext $slide_ext --geojson_dir $geojson_dir --output_dir $output_dir --patch_size $patch_size --patch_level $patch_level --num_workers 3
#echo --slide_dir $slide_dir --slide_ext $slide_ext --geojson_dir $geojson_dir --output_dir $output_dir --patch_size $patch_size --patch_level $patch_level --num_workers 5