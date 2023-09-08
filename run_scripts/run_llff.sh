#!/bin/bash


# CUDA_VISIBLE_DEVICES=$1 ns-train nerfacto --data /data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/flower/transforms_256x256.json \
#     --experiment-name llff/flower --vis wandb \
#     nerfstudio-data --scale_factor 0.2


# CUDA_VISIBLE_DEVICES=$1 ns-train nerfacto --data /data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/flower/transforms.json \
#     --experiment-name llff/flower_autoscale --vis wandb \
#     nerfstudio-data \
#     --auto-scale-poses True \
#     --downscale-factor 8 \

#  CUDA_VISIBLE_DEVICES=$1 ns-train eg3d --data /data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/flower/transforms.json \
#     --experiment-name llff/flower_eg3d --vis wandb \
#     --pipeline.model.near-plane 0.0 \
#     --pipeline.model.far-plane 1000.0 \
#     --pipeline.model.is_contracted True \
#     --pipeline.model.grid-base-resolution 256\
#     --pipeline.model.grid-feature-dim 32 \
#     --pipeline.model.num_samples 128 \
#     --pipeline.model.num_importance_samples 64 \
#     --pipeline.model.use-viewdirs True \
#     --pipeline.model.loss-coefficients.plane-tv 0.01 \
#     --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
#     --pipeline.model.loss-coefficients.distortion_fine 0.001 \
#     nerfstudio-data \
#     --scale_factor 0.2 \
#     --downscale-factor 8 \


####### eg3d llff ndc ###############################
# CUDA_VISIBLE_DEVICES=$1 ns-train eg3d --data /data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/flower/transforms.json \
#    --experiment-name llff/flower_eg3d_ndc --vis wandb \
#    --pipeline.model.near-plane 0.0 \
#    --pipeline.model.far-plane 1.0 \
#    --pipeline.model.grid-base-resolution 256\
#    --pipeline.model.grid-feature-dim 32 \
#    --pipeline.model.num_samples 64 \
#    --pipeline.model.num_importance_samples 64 \
#    --pipeline.model.use-viewdirs True \
#    --pipeline.model.loss-coefficients.plane-tv 0.01 \
#    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
#    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
#    --pipeline.datamanager.use-ndc True \
#    nerfstudio-data \
#    --downscale-factor 8 \




# CUDA_VISIBLE_DEVICES=$1 ns-train vanilla-nerf --data /data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/flower/ \
#    --experiment-name llff/flower_vanilla_ndc --vis wandb \
#    --pipeline.model.near-plane 0.0 \
#    --pipeline.model.far-plane 1.0 \
#    --pipeline.model.use-ndc True \
#    llff-data \


# CUDA_VISIBLE_DEVICES=$1 ns-train tensorf --data /data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/flower/ \
#    --experiment-name llff/flower_tensorf_ndc --vis wandb \
#    --pipeline.model.use-ndc True \
#    llff-data \


# CUDA_VISIBLE_DEVICES=2 \
# ns-train kplanes-importance \
#     --experiment-name llff/flower_kplanes_ndc \
#     --data=/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/flower \
#     --pipeline.model.near-plane 0.0 \
#     --pipeline.mode.far-plane 1.0 \
#     --vis wandb \
#     llff-data

# CUDA_VISIBLE_DEVICES=$1 \
# ns-train kplanes-importance --experiment-name llff/flower_kplanes_ndc \
#     --pipeline.model.grid-base-resolution 256 256 256\
#     --pipeline.model.grid-feature-dim 48 \
#     --pipeline.model.multiscale-res 1 \
#     --pipeline.model.num_samples 128 \
#     --pipeline.model.num_importance_samples 64 \
#     --pipeline.model.reduce 'sum' \
#     --pipeline.model.use-viewdirs True \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/flower" \
#     --pipeline.model.near-plane 0.0 \
#     --pipeline.model.far-plane 1.0 \
#     --pipeline.model.use-ndc True \
#     --max-num-iterations 30000 \
#     --vis wandb \
#     llff-data

# CUDA_VISIBLE_DEVICES=0 \
# ns-train eg3d --experiment-name llff_256_perframesr_tv1/flower \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_256perframesr/flower" \
#     --pipeline.model.grid-base-resolution 256\
#     --pipeline.model.grid-feature-dim 32 \
#     --pipeline.model.num-samples 128 \
#     --pipeline.model.num-importance-samples 64 \
#     --pipeline.model.use-viewdirs True \
#     --pipeline.model.loss-coefficients.plane-tv 1.0 \
#     --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
#     --pipeline.model.loss-coefficients.distortion_fine 0.001 \
#     --pipeline.model.use-tcnn False \
#     --pipeline.model.reduce 'mean' \
#     --pipeline.model.use-ndc True \
#     --pipeline.model.near-plane 0.0 \
#     --pipeline.model.far-plane 1.0 \
#     --max-num-iterations 30001 \
#     --vis wandb \
#     llff-data \


CUDA_VISIBLE_DEVICES=0 \
ns-train eg3d --experiment-name llff_64_tv1/flower \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_256perframesr/flower" \
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
ns-train eg3d --experiment-name llff_64_tv1/fern \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64/fern" \
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
ns-train eg3d --experiment-name llff_64_tv1/trex \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64/trex" \
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
ns-train eg3d --experiment-name llff_64_tv1/horns \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64/horns" \
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
ns-train eg3d --experiment-name llff_64_tv1/leaves \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64/leaves" \
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
ns-train eg3d --experiment-name llff_64_tv1/fortress \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64/fortress" \
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
ns-train eg3d --experiment-name llff_64_tv1/orchids \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64/orchids" \
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
ns-train eg3d --experiment-name llff_64_tv1/room \
    --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64/room" \
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


# CUDA_VISIBLE_DEVICES=3 \
# ns-train eg3d --experiment-name llff_64_tv1/flower \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64/flower" \
#     --pipeline.model.grid-base-resolution 256\
#     --pipeline.model.grid-feature-dim 32 \
#     --pipeline.model.num-samples 128 \
#     --pipeline.model.num-importance-samples 64 \
#     --pipeline.model.use-viewdirs True \
#     --pipeline.model.loss-coefficients.plane-tv 1.0 \
#     --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
#     --pipeline.model.loss-coefficients.distortion_fine 0.001 \
#     --pipeline.model.use-tcnn False \
#     --pipeline.model.reduce 'mean' \
#     --pipeline.model.use-ndc True \
#     --pipeline.model.near-plane 0.0 \
#     --pipeline.model.far-plane 1.0 \
#     --max-num-iterations 30001 \
#     --vis wandb \
#     llff-data \

# CUDA_VISIBLE_DEVICES=$1 \
# ns-train eg3d --experiment-name llff_64_tv1/flower \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_64/flower" \
#     --pipeline.model.grid-base-resolution 256\
#     --pipeline.model.grid-feature-dim 32 \
#     --pipeline.model.num-samples 128 \
#     --pipeline.model.num-importance-samples 64 \
#     --pipeline.model.use-viewdirs True \
#     --pipeline.model.loss-coefficients.plane-tv 1.0 \
#     --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
#     --pipeline.model.loss-coefficients.distortion_fine 0.001 \
#     --pipeline.model.use-tcnn False \
#     --pipeline.model.reduce 'mean' \
#     --pipeline.model.use-ndc True \
#     --pipeline.model.near-plane 0.0 \
#     --pipeline.model.far-plane 1.0 \
#     --max-num-iterations 30001 \
#     --vis wandb \
#     llff-data \