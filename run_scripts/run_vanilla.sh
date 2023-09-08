#!/bin/bash

CIDX=$1

# # drums 
CUDA_VISIBLE_DEVICES=$CIDX \
ns-train vanilla-nerf --experiment-name blender_64x64/drums \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/drums" \
    --max-num-iterations 50000 \
    --vis wandb \
    blender-data \