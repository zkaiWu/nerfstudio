#!/bin/bash

CIDX=$1

# CUDA_VISIBLE_DEVICES=$CIDX \
# ns-train eg3d --experiment-name blender_64x64_eg3d_paperconfig/lego \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/lego" \
#     --max-num-iterations 50000 \
#     --vis wandb \
#     blender-data \


CUDA_VISIBLE_DEVICES=$CIDX \
ns-train eg3d --experiment-name blender_256x256_eg3d_paperconfig/lego \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256/lego" \
    --max-num-iterations 50000 \
    --vis wandb \
    blender-data \

