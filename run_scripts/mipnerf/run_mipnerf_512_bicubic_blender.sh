#!/bin/bash


# CUDA_VISIBLE_DEVICES=0 \
# ns-train tensorf --experiment-name blender_256x256_tensorf_perframesr/lego \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe/lego" \
#     --pipeline.model.tensorf-encoding triplane \
#     --max-num-iterations 30000 \
#     --vis wandb \
#     blender-data \


CUDA_VISIBLE_DEVICES=0 \
ns-train mipnerf --experiment-name blender_256bicubic_mipnerf/chair \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256_bicubic/chair" \
    --max-num-iterations 40001 \
    --vis wandb \
    blender-data &\


CUDA_VISIBLE_DEVICES=1 \
ns-train mipnerf --experiment-name blender_256bicubic_mipnerf/drums \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256_bicubic/drums" \
    --max-num-iterations 40001 \
    --vis wandb \
    blender-data &\

CUDA_VISIBLE_DEVICES=2 \
ns-train mipnerf --experiment-name blender_256bicubic_mipnerf/ficus \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256_bicubic/ficus" \
    --max-num-iterations 40001 \
    --vis wandb \
    blender-data &\

CUDA_VISIBLE_DEVICES=3 \
ns-train mipnerf --experiment-name blender_256bicubic_mipnerf/hotdog \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256_bicubic/hotdog" \
    --max-num-iterations 40001 \
    --vis wandb \
    blender-data &\

CUDA_VISIBLE_DEVICES=0 \
ns-train mipnerf --experiment-name blender_256bicubic_mipnerf/lego \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256_bicubic/lego" \
    --max-num-iterations 40001 \
    --vis wandb \
    blender-data &\

CUDA_VISIBLE_DEVICES=1 \
ns-train mipnerf --experiment-name blender_256bicubic_mipnerf/materials \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256_bicubic/materials" \
    --max-num-iterations 40001 \
    --vis wandb \
    blender-data &\

CUDA_VISIBLE_DEVICES=2 \
ns-train mipnerf --experiment-name blender_256bicubic_mipnerf/mic \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256_bicubic/mic" \
    --max-num-iterations 40001 \
    --vis wandb \
    blender-data &\

CUDA_VISIBLE_DEVICES=3 \
ns-train mipnerf --experiment-name blender_256bicubic_mipnerf/ship \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256_bicubic/ship" \
    --max-num-iterations 40001 \
    --vis wandb \
    blender-data &\