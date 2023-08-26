#!/bin/bash

CIDX=$1

# CUDA_VISIBLE_DEVICES=$CIDX \
# ns-train eg3d --experiment-name blender_64x64_eg3d_paperconfig/lego \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/lego" \
#     --max-num-iterations 50000 \
#     --vis wandb \
#     blender-data \


# CUDA_VISIBLE_DEVICES=$CIDX \
# ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_bigDecoder_trires256_softplus_wotcnn_meanReduce_zeropadding/lego \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/lego" \
#     --pipeline.model.grid-base-resolution 256\
#     --pipeline.model.grid-feature-dim 48 \
#     --pipeline.model.num_samples 128 \
#     --pipeline.model.num_importance_samples 64 \
#     --pipeline.model.use-viewdirs True \
#     --pipeline.model.loss-coefficients.plane-tv 0.01 \
#     --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
#     --pipeline.model.loss-coefficients.distortion_fine 0.001 \
#     --pipeline.model.use-tcnn False \
#     --pipeline.model.reduce 'mean' \
#     --max-num-iterations 30001 \
#     --vis wandb \
#     blender-data \




CUDA_VISIBLE_DEVICES=0 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_32dim_softplus_wotcnn_meanReduce_zeropadding/chair \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/chair" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num_samples 128 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 0.01 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --max-num-iterations 30001 \
    --vis wandb \
    blender-data &\


CUDA_VISIBLE_DEVICES=0 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_32dim_softplus_wotcnn_meanReduce_zeropadding/drums \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/drums" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num_samples 128 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 0.01 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --max-num-iterations 30001 \
    --vis wandb \
    blender-data &\


CUDA_VISIBLE_DEVICES=0 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_32dim_softplus_wotcnn_meanReduce_zeropadding/ficus \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/ficus" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num_samples 128 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 0.01 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --max-num-iterations 30001 \
    --vis wandb \
    blender-data &\


CUDA_VISIBLE_DEVICES=0 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_32dim_softplus_wotcnn_meanReduce_zeropadding/lego \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/lego" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num_samples 128 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 0.01 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --max-num-iterations 30001 \
    --vis wandb \
    blender-data &\


CUDA_VISIBLE_DEVICES=1 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_32dim_softplus_wotcnn_meanReduce_zeropadding/hotdog \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/hotdog" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num_samples 128 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 0.01 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --max-num-iterations 30001 \
    --vis wandb \
    blender-data &\




CUDA_VISIBLE_DEVICES=1 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_32dim_softplus_wotcnn_meanReduce_zeropadding/materials \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/materials" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num_samples 128 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 0.01 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --max-num-iterations 30001 \
    --vis wandb \
    blender-data &\



CUDA_VISIBLE_DEVICES=1 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_32dim_softplus_wotcnn_meanReduce_zeropadding/mic \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/mic" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num_samples 128 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 0.01 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --max-num-iterations 30001 \
    --vis wandb \
    blender-data &\


CUDA_VISIBLE_DEVICES=1 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_32dim_softplus_wotcnn_meanReduce_zeropadding/ship \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/ship" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num_samples 128 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 0.01 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --max-num-iterations 30001 \
    --vis wandb \
    blender-data &\