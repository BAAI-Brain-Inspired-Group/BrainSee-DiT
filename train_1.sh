#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh

conda activate dit

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    train.py \
    --image-size   256 \
    --epochs   200 \
    --global-batch-size   36 \
    --ckpt-every   4000 \
    --num-hc   2 \
    --p-zero   1.0 \
    --time   0 \
    --num-pattern   8 \
    --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \