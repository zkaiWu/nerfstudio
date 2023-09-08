#!/bin/bash


CIDX=$1
BS=$2
WS=$3


CUDA_VISIBLE_DEVICES=$CIDX python /data5/wuzhongkai/proj/nerfstudio/data_processing/llff/llff_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_256perframesr \
    --obj_name flower \
    --noise_level 0 \
    --batch_size $BS \
    --world_size $WS \
    # --obj_name flower \