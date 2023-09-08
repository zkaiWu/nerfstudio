#!/bin/bash



CUDA_VISIBLE_DEVICES=0 \
ns-train eg3d --experiment-name llff_512_bicubic_tv1/flower \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/flower" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num-samples 128 \
    --pipeline.model.num-importance-samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 1.0 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --pipeline.model.use-ndc True \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\


CUDA_VISIBLE_DEVICES=1 \
ns-train eg3d --experiment-name llff_512_bicubic_tv1/fern \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/fern" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num-samples 128 \
    --pipeline.model.num-importance-samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 1.0 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --pipeline.model.use-ndc True \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\


CUDA_VISIBLE_DEVICES=2 \
ns-train eg3d --experiment-name llff_512_bicubic_tv1/trex \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/trex" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num-samples 128 \
    --pipeline.model.num-importance-samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 1.0 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --pipeline.model.use-ndc True \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\


CUDA_VISIBLE_DEVICES=3 \
ns-train eg3d --experiment-name llff_512_bicubic_tv1/horns \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/horns" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num-samples 128 \
    --pipeline.model.num-importance-samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 1.0 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --pipeline.model.use-ndc True \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\


CUDA_VISIBLE_DEVICES=0 \
ns-train eg3d --experiment-name llff_512_bicubic_tv1/leaves \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/leaves" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num-samples 128 \
    --pipeline.model.num-importance-samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 1.0 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --pipeline.model.use-ndc True \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\


CUDA_VISIBLE_DEVICES=1 \
ns-train eg3d --experiment-name llff_512_bicubic_tv1/fortress \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/fortress" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num-samples 128 \
    --pipeline.model.num-importance-samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 1.0 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --pipeline.model.use-ndc True \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\


CUDA_VISIBLE_DEVICES=2 \
ns-train eg3d --experiment-name llff_512_bicubic_tv1/orchids \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/orchids" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num-samples 128 \
    --pipeline.model.num-importance-samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 1.0 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --pipeline.model.use-ndc True \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data &\


CUDA_VISIBLE_DEVICES=3 \
ns-train eg3d --experiment-name llff_512_bicubic_tv1/room \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64to512_bicubic/room" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num-samples 128 \
    --pipeline.model.num-importance-samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 1.0 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --pipeline.model.use-ndc True \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1.0 \
    --max-num-iterations 30001 \
    --vis wandb \
    llff-data \