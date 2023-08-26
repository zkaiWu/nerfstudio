# CUDA_VISIBLE_DEVICES=$1 ns-train nerfacto --data /data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/flower/transforms_256x256.json \
#     --experiment-name llff/flower --vis wandb \
#     nerfstudio-data --scale_factor 0.2


# CUDA_VISIBLE_DEVICES=$1 ns-train nerfacto --data /data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/flower/transforms.json \
#     --experiment-name llff/flower_autoscale --vis wandb \
#     nerfstudio-data \
#     --auto-scale-poses True \
#     --downscale-factor 8 \

 CUDA_VISIBLE_DEVICES=$1 ns-train eg3d --data /data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/flower/transforms.json \
    --experiment-name llff/flower_eg3d --vis wandb \
    --pipeline.model.near-plane 0.0 \
    --pipeline.model.far-plane 1000.0 \
    --pipeline.model.is_contracted True \
    --pipeline.model.grid-base-resolution 256\
    --pipeline.model.grid-feature-dim 32 \
    --pipeline.model.num_samples 128 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.use-viewdirs True \
    --pipeline.model.loss-coefficients.plane-tv 0.01 \
    --pipeline.model.loss-coefficients.distortion_coarse 0.001 \
    --pipeline.model.loss-coefficients.distortion_fine 0.001 \
    nerfstudio-data \
    --scale_factor 0.2 \
    --downscale-factor 8 \