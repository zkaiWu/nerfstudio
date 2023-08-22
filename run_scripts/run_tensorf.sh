#!/bin/bash

CIDX=$1

# CUDA_VISIBLE_DEVICES=$CIDX \
# ns-train eg3d --experiment-name blender_64x64_eg3d_paperconfig/lego \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/lego" \
#     --max-num-iterations 50000 \
#     --vis wandb \
#     blender-data \


CUDA_VISIBLE_DEVICES=$CIDX \
ns-train tensorf --experiment-name blender_256x256_tensorf_triplane_debug/lego \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256/lego" \
    --pipeline.model.tensorf-encoding triplane \
    --max-num-iterations 50000 \
    --vis wandb \
    blender-data \


CUDA_VISIBLE_DEVICES=$CIDX \
# ns-train tensorf-proposal --experiment-name blender_256x256_tensorf_proposal_numdendim48_numcoldim48_debug/lego \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256/lego" \
#     --pipeline.model.num_den_components 48 \
#     --pipeline.model.num_color_components 48 \
#     --max-num-iterations 30000 \
#     --vis wandb \
#     blender-data \