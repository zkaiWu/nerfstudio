#!/bin/bash


CIDX=$1
EXP_DIR=$2

CUDA_VISIBLE_DEVICES=$CIDX ns-render interpolate \
    --load-config $EXP_DIR/config.yml \
    --output_path $EXP_DIR/video.mp4 \
    --image-format png \
    --interpolation-steps 3 \
    --frame-rate 24 \
    --downscale-factor 0.5 \