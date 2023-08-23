#!/bin/bash

CIDX=$1

# CUDA_VISIBLE_DEVICES=$CIDX \
# ns-train eg3d --experiment-name blender_64x64_eg3d_paperconfig/lego \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/lego" \
#     --max-num-iterations 50000 \
#     --vis wandb \
#     blender-data \


CUDA_VISIBLE_DEVICES=$CIDX \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn/lego \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/lego" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 48 \
    --pipeline.model.num_samples 128 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 0.01 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --max-num-iterations 30001 \
    --vis wandb \
    blender-data \


# CUDA_VISIBLE_DEVICES=$CIDX \
# ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_eg3dfield_truncexp_debug2/lego \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/lego" \
#     --pipeline.model.triplane_resolution 256\
#     --pipeline.model.triplane_feature_dim 48 \
#     --pipeline.model.num_samples 128 \
#     --pipeline.model.num_importance_samples 64 \
#     --pipeline.model.use-viewdirs True \
#     --pipeline.model.loss-coefficients.plane-tv 0.01 \
#     --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
#     --pipeline.model.loss-coefficients.distortion_fine 0.001 \
#     --max-num-iterations 30001 \
#     --vis wandb \
#     blender-data \