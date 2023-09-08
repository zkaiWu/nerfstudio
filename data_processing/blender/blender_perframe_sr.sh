#!/bin/bash

CIDX=$1

# CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
#     --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
#     --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
#     --obj_name chair \
#     --prompt "A white and green chair with golden pattern" \
#     --noise_level 0 \
#     --guidance_scale 4.0 \
#     --batch_size 4 \
#     --world_size 2 \


# CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
#     --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
#     --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
#     --obj_name chair \
#     --prompt "A white and green chair with golden pattern" \
#     --noise_level 0 \
#     --guidance_scale 4.0 \
#     --batch_size 4 \
#     --world_size 2 \
#     --split 'test'


# CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
#     --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
#     --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
#     --obj_name chair \
#     --prompt "A white and green chair with golden pattern" \
#     --noise_level 0 \
#     --guidance_scale 4.0 \
#     --batch_size 4 \
#     --world_size 2 \
#     --split 'val'

CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name drums \
    --prompt "a red drum kit" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name drums \
    --prompt "a red drum kit" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \
    --split 'test'


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name drums \
    --prompt "a red drum kit" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \
    --split 'val'


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name ficus \
    --prompt "a plant in a black vase" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name ficus \
    --prompt "a plant in a black vase" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \
    --split 'test'


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name ficus \
    --prompt "a plant in a black vase" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \
    --split 'val'

CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name hotdog \
    --prompt "two hotdogs with mustard and ketchup on a plate" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name hotdog \
    --prompt "two hotdogs with mustard and ketchup on a plate" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \
    --split 'test'


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name hotdog \
    --prompt "two hotdogs with mustard and ketchup on a plate" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \
    --split 'val'

CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name materials \
    --prompt "a bunch of different colored smoothed ball ball ball ball ball made of different materials" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name materials \
    --prompt "a bunch of different colored smoothed ball ball ball ball ball made of different materials" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \
    --split 'test'


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name materials \
    --prompt "a bunch of different colored smoothed ball ball ball ball ball made of different materials" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \
    --split 'val'

CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name mic \
    --prompt "A metal matte microphone sits on a tripod with long wires trailing behind it" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name mic \
    --prompt "A metal matte microphone sits on a tripod with long wires trailing behind it" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \
    --split 'test'


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name mic \
    --prompt "A metal matte microphone sits on a tripod with long wires trailing behind it" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \
    --split 'val'

CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name ship \
    --prompt "a large red boat floating on top of a body of water" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name ship \
    --prompt "a large red boat floating on top of a body of water" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \
    --split 'test'


CUDA_VISIBLE_DEVICES=$CIDX python data_processing/blender/blender_perframe_sr.py \
    --input_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic \
    --output_dir /data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe \
    --obj_name ship \
    --prompt "a large red boat floating on top of a body of water" \
    --noise_level 0 \
    --guidance_scale 4.0 \
    --batch_size 4 \
    --world_size 2 \
    --split 'val'