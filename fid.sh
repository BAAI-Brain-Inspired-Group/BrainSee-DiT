#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh

conda activate dit


python fid.py --cfg-scale 1.35  --cuda-num  3  --ckpt /share/project/checkpoint/001-mask-0.94-60000-8-myDiT-XL-2/2700000.pt  --image-size 256 --num-pattern   8 --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1
# python fid.py --cfg-scale 1.35  --cuda-num  3  --ckpt /share/project/checkpoint/000-mask-1.0-0-8-myDiT-XL-2/2700000.pt  --image-size 256 --num-pattern   8 --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1
# python fid.py --cfg-scale 1.0  --cuda-num  3  --ckpt /share/project/checkpoint/000-mask-0.94-60000-24-myDiT-XL-2/1100000.pt  --image-size 256 --num-pattern   8 --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1
# python fid.py --cfg-scale 1.0  --cuda-num  2  --ckpt /share/project/checkpoint/002-mask-1.0-0-24-myDiT-XL-2/0950000.pt  --image-size 256 --num-pattern   8 --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1