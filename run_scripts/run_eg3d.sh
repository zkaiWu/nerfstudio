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




# CUDA_VISIBLE_DEVICES=$1 \
# ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_32dim_softplus_wotcnn_meanReduce_zeropadding_256perframesr/lego \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe/lego" \
#     --pipeline.model.grid-base-resolution 256\
#     --pipeline.model.grid-feature-dim 32 \
#     --pipeline.model.num_samples 128 \
#     --pipeline.model.num_importance_samples 64 \
#     --pipeline.model.use-viewdirs True \
#     --pipeline.model.loss-coefficients.plane-tv 0.01 \
#     --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
#     --pipeline.model.loss-coefficients.distortion_fine 0.001 \
#     --pipeline.model.use-tcnn False \
#     --pipeline.model.reduce 'mean' \
#     --pipeline.model.use-ndc False \
#     --max-num-iterations 30001 \
#     --vis wandb \
#     blender-data \

CUDA_VISIBLE_DEVICES=0 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding_256perframesr/chair \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe/chair" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 48 \
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
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding_256perframesr/drums \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe/drums" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 48 \
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


CUDA_VISIBLE_DEVICES=2 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding_256perframesr/ficus \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe/ficus" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 48 \
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


CUDA_VISIBLE_DEVICES=3 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding_256perframesr/hotdog \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe/hotdog" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 48 \
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
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding_256perframesr/lego \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe/lego" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 48 \
    --pipeline.model.num_samples 128 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 0.01 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    --pipeline.model.use-tcnn False \
    --pipeline.model.reduce 'mean' \
    --pipeline.model.use-ndc False \
    --max-num-iterations 30001 \
    --vis wandb \
    blender-data &\


CUDA_VISIBLE_DEVICES=1 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding_256perframesr/materials \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe/materials" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 48 \
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




CUDA_VISIBLE_DEVICES=2 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding_256perframesr/mic \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe/mic" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 48 \
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



CUDA_VISIBLE_DEVICES=3 \
ns-train eg3d --experiment-name blender_64x64_eg3d_tvl1e-2_Disl1e-3_wviewdirs_trires256_softplus_wotcnn_meanReduce_zeropadding_256perframesr/ship \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256perframe/ship" \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 48 \
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
    blender-data \