#!/bin/bash


CIDX=$1
EXP_DIR=$2
RN=$3

CUDA_VISIBLE_DEVICES=$CIDX ns-render interpolate \
    --load-config $EXP_DIR/config.yml \
    --output_path $EXP_DIR/video_$RN.mp4 \
    --image-format png \
    --interpolation-steps 3 \
    --frame-rate 24 \
    --downscale-factor 0.5 \
    --rendered-output-names $RN