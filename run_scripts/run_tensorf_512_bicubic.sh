#!/bin/bash


# CUDA_VISIBLE_DEVICES=0 \
# ns-train tensorf --experiment-name blender_256x256_tensorf_perframesr/lego \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe/lego" \
#     --pipeline.model.tensorf-encoding triplane \
#     --max-num-iterations 30000 \
#     --vis wandb \
#     blender-data \


CUDA_VISIBLE_DEVICES=0 \
ns-train eg3d --experiment-name llff_512_bicubic_tensorf/fern \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/fern" \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --pipeline.model.use-ndc True \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\


CUDA_VISIBLE_DEVICES=1 \
ns-train eg3d --experiment-name llff_512_bicubic_tensorf/flower \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/flower" \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --pipeline.model.use-ndc True \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\

CUDA_VISIBLE_DEVICES=2 \
ns-train eg3d --experiment-name llff_512_bicubic_tensorf/fortress \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/fortress" \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --pipeline.model.use-ndc True \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\

CUDA_VISIBLE_DEVICES=3 \
ns-train eg3d --experiment-name llff_512_bicubic_tensorf/horns \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/horns" \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --pipeline.model.use-ndc True \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\

CUDA_VISIBLE_DEVICES=0 \
ns-train eg3d --experiment-name llff_512_bicubic_tensorf/leaves \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/leaves" \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --pipeline.model.use-ndc True \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\

CUDA_VISIBLE_DEVICES=1 \
ns-train eg3d --experiment-name llff_512_bicubic_tensorf/orchids \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/orchids" \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --pipeline.model.use-ndc True \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\

CUDA_VISIBLE_DEVICES=2 \
ns-train eg3d --experiment-name llff_512_bicubic_tensorf/room \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/room" \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --pipeline.model.use-ndc True \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\

CUDA_VISIBLE_DEVICES=3 \
ns-train eg3d --experiment-name llff_512_bicubic_tensorf/trex \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/trex" \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --pipeline.model.use-ndc True \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\