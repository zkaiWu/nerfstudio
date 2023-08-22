#!/bin/bash

# CUDA_VISIBLE_DEVICES=$CIDX \
# ns-train eg3d --experiment-name blender_64x64_eg3d_paperconfig/lego \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/lego" \
#     --max-num-iterations 50000 \
#     --vis wandb \
#     blender-data \


# CUDA_VISIBLE_DEVICES=$CIDX \
# ns-train kplanes --experiment-name blender_256x256_kplanes_trires256_featuredim32/lego \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256/lego" \
#     --pipeline.model.grid-base-resolution 256 256 256\
#     --pipeline.model.grid-feature-dim 32 \
#     --pipeline.model.multiscale-res 1 \
#     --max-num-iterations 50000 \
#     --vis wandb \
#     blender-data \


# Set k-planes to eg3d setting and use viewdirs
# CUDA_VISIBLE_DEVICES=$CIDX \
# ns-train kplanes-importance --experiment-name blender_256x256_kplanes_trires512_rgbcoarselossAndTvloss_sumReduce/lego \
#     --pipeline.model.grid-base-resolution 512 512 512 \
#     --pipeline.model.grid-feature-dim 48 \
#     --pipeline.model.multiscale-res 1 \
#     --pipeline.model.num_samples 200 \
#     --pipeline.model.num_importance_samples 200 \
#     --pipeline.model.reduce 'sum' \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256/lego" \
#     --max-num-iterations 30000 \
#     --vis wandb \


# CUDA_VISIBLE_DEVICES=$CIDX \
# ns-train kplanes-importance --experiment-name blender_256x256_kplanes_trires512_rgbcoarselossAndTvloss_sumReduce/lego \
#     --pipeline.model.grid-base-resolution 512 512 512 \
#     --pipeline.model.grid-feature-dim 48 \
#     --pipeline.model.multiscale-res 1 \
#     --pipeline.model.num_samples 512 \
#     --pipeline.model.num_importance_samples 256 \
#     --pipeline.model.reduce 'sum' \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256/lego" \
#     --max-num-iterations 30000 \
#     --vis wandb \

# CUDA_VISIBLE_DEVICES=$CIDX \
# ns-train kplanes-importance --experiment-name blender_256x256_kplanes_trires512_rgbcoarselossAndTvlossAndDistortionL_sumReduce_woviewdirs/lego \
#     --pipeline.model.grid-base-resolution 512 512 512 \
#     --pipeline.model.grid-feature-dim 48 \
#     --pipeline.model.multiscale-res 1 \
#     --pipeline.model.num_samples 512 \
#     --pipeline.model.num_importance_samples 256 \
#     --pipeline.model.reduce 'sum' \
#     --pipeline.model.use-viewdirs False \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_256x256/lego" \
#     --max-num-iterations 30000 \
#     --vis wandb \

# CUDA_VISIBLE_DEVICES=0 \
# ns-train kplanes-importance --experiment-name blender_64x64_kplanes_trires256_lesssample_rgbcoarselossAndTvlossDisL_sumReduce_TV0.01_DisL1e-3/ficus \
#     --pipeline.model.grid-base-resolution 256 256 256\
#     --pipeline.model.grid-feature-dim 48 \
#     --pipeline.model.multiscale-res 1 \
#     --pipeline.model.num_samples 128 \
#     --pipeline.model.num_importance_samples 64 \
#     --pipeline.model.reduce 'sum' \
#     --pipeline.model.use-viewdirs True \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/ficus" \
#     --max-num-iterations 30000 \
#     --vis wandb \
# &

# CUDA_VISIBLE_DEVICES=1 \
# ns-train kplanes-importance --experiment-name blender_64x64_kplanes_trires256_lesssample_rgbcoarselossAndTvlossDisL_sumReduce_TV0.01_DisL1e-3/ship \
#     --pipeline.model.grid-base-resolution 256 256 256\
#     --pipeline.model.grid-feature-dim 48 \
#     --pipeline.model.multiscale-res 1 \
#     --pipeline.model.num_samples 128 \
#     --pipeline.model.num_importance_samples 64 \
#     --pipeline.model.reduce 'sum' \
#     --pipeline.model.use-viewdirs True \
#     --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/ship" \
#     --max-num-iterations 30000 \
#     --vis wandb \


CUDA_VISIBLE_DEVICES=2 \
ns-train kplanes-importance --experiment-name blender_64x64_kplanes_trires256_lesssample_rgbcoarselossAndTvlossDisL_sumReduce_TV0.01_DisL1e-3_debug/drums \
    --pipeline.model.grid-base-resolution 256 256 256\
    --pipeline.model.grid-feature-dim 48 \
    --pipeline.model.multiscale-res 1 \
    --pipeline.model.num_samples 128 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.reduce 'sum' \
    --pipeline.model.use-viewdirs True \
    --data="/data5/wuzhongkai/data/dreamfusion_data/blender/nerf_synthetic_64x64/drums" \
    --max-num-iterations 30000 \
    --vis wandb \