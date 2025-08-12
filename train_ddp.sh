#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh

conda activate dit



python -m torch.distributed.launch --nproc_per_node=8 \
        --nnodes=4 --node_rank=${RANK:-0} \
        --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=31415 --use_env \
    train.py \
    --image-size   256 \
    --epochs   1400 \
    --global-batch-size   8 \
    --ckpt-every   10000 \
    --num-hc   2 \
    --p-zero   0.94 \
    --time   0 \
    --num-pattern   8 \
    --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
    --catGP True \
    --ckpt  /home/zqchen/code/mask_dit/results/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/6700000.pt \
    --numstep 6700000 \

# python -m torch.distributed.launch --nproc_per_node=1 \
#         --nnodes=1 --node_rank=${RANK:-0} \
#         --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=31415 --use_env \
#     train.py \
#     --image-size   256 \
#     --epochs   1400 \
#     --global-batch-size   8 \
#     --ckpt-every   10000 \
#     --num-hc   2 \
#     --p-zero   0.94 \
#     --time   60000 \
#     --num-pattern   8 \
#     --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
#     --catGP  True \
#     --ckpt  /home/zqchen/code/mask_dit/results/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/6800000.pt \
#     --numstep 6800000 \
#     --lr 4e-5 \

# python -m torch.distributed.launch --nproc_per_node=8 \
#         --nnodes=4 --node_rank=${RANK:-0} \
#         --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=31415 --use_env \
#     train.py \
#     --image-size   256 \
#     --epochs   1400 \
#     --global-batch-size   8 \
#     --ckpt-every   10000 \
#     --num-hc   2 \
#     --p-zero   0.6 \
#     --time   60000 \
#     --num-pattern   8 \
#     --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
#     --catGP  True \
#     # --ckpt  /home/zqchen/code/mask_dit/results/002-mask-0.6-60000-8-myDiT-XL-2/checkpoints/3390000.pt \
#     # --numstep 3390000 \

# python -m torch.distributed.launch --nproc_per_node=8 \
#         --nnodes=4 --node_rank=${RANK:-0} \
#         --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=31415 --use_env \
#     train.py \
#     --image-size   512 \
#     --epochs   1400 \
#     --global-batch-size   8 \
#     --ckpt-every   10000 \
#     --num-hc   2 \
#     --p-zero   1.0 \
#     --time   0 \
#     --num-pattern   8 \
#     --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
#     --catGP True \
#     # --ckpt  /home/zqchen/code/mask_dit/results/004-mask-0.94-60000-8-512x512-myDiT-XL-2/checkpoints/0790000.pt \
#     # --numstep 790000

# python -m torch.distributed.launch --nproc_per_node=8 \
#         --nnodes=4 --node_rank=${RANK:-0} \
#         --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=31415 --use_env \
#     train.py \
#     --image-size   512 \
#     --epochs   1400 \
#     --global-batch-size   8 \
#     --ckpt-every   10000 \
#     --num-hc   2 \
#     --p-zero   0.94 \
#     --time   60000 \
#     --num-pattern   8 \
#     --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
#     --catGP  True \
#     # --ckpt  /home/zqchen/code/mask_dit/results/005-mask-1.0-0-8-512x512-myDiT-XL-2/checkpoints/0790000.pt \
#     # --numstep 790000 \
