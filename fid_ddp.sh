#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh

conda activate dit

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")

# # 设置ckpt列表
# for i in '0400000' '0050000' '0100000' '0200000' '0300000'

# 设置cfg列表
for i in 1. 1.25 1.5
# for i in 1.
do
  echo " $i "

# # 启动分布式训练
# torchrun \
#     --nproc_per_node=${NGPUS} \
#     --nnodes=${WORLD_SIZE} \
#     --node_rank=${RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     fid_ddp.py \
#     --cfg-scale $i \
#     --ckpt /home/zqchen/code/mask_dit/results/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/6930000.pt \
#     --image-size 256 \
#     --num-pattern 8 \
#     --hc-path equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
#     --scale 0 4 \
#     --save_path /share/project/dataset/cfgnew_png/dit_fid_imagenet_czq_04

# # 启动分布式训练
# torchrun \
#     --nproc_per_node=${NGPUS} \
#     --nnodes=${WORLD_SIZE} \
#     --node_rank=${RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     fid_ddp.py \
#     --cfg-scale $i \
#     --ckpt /home/zqchen/code/mask_dit/results/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/6930000.pt \
#     --image-size 256 \
#     --num-pattern 8 \
#     --hc-path equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
#     --scale 4 \
#     --save_path /share/project/dataset/cfgnew_png/dit_fid_imagenet_czq_4 \
#     --idx 1 \

# 启动分布式训练
torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    fid_ddp.py \
    --cfg-scale $i \
    --ckpt /home/zqchen/code/mask_dit/results/021-mask-0.94-60000-8-256x256-myDiT-XL-2/checkpoints/6930000.pt \
    --image-size 256 \
    --num-pattern 8 \
    --hc-path equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
    --scale 0 4 \
    --save_path /share/project/dataset/cfgnew_png/dit_fid_imagenet_czq_04 \
    --idx 0 \


# # 启动分布式训练
# torchrun \
#     --nproc_per_node=${NGPUS} \
#     --nnodes=${WORLD_SIZE} \
#     --node_rank=${RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     fid_ddp.py \
#     --cfg-scale 1. \
#     --ckpt /home/zqchen/code/mask_dit/results/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/$i.pt \
#     --image-size 256 \
#     --num-pattern 8 \
#     --hc-path equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
#     --scale 0 1 2 3 4 \
#     --save_path /share/project/dataset/cfgnew_png/dit_fid_imagenet_czq \
#     --idx 0 \

# # 启动分布式训练
# torchrun \
#     --nproc_per_node=${NGPUS} \
#     --nnodes=${WORLD_SIZE} \
#     --node_rank=${RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     fid_ddp.py \
#     --cfg-scale 1. \
#     --ckpt /home/zqchen/code/mask_dit/results/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/$i.pt \
#     --image-size 256 \
#     --num-pattern 8 \
#     --hc-path equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
#     --scale 0 1 2 3 4 \
#     --save_path /share/project/dataset/cfgnew_png/dit_fid_imagenet_czq \
#     --idx 6 \


# # 启动分布式训练
# torchrun \
#     --nproc_per_node=${NGPUS} \
#     --nnodes=${WORLD_SIZE} \
#     --node_rank=${RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     fid_ddp.py \
#     --cfg-scale 1. \
#     --ckpt /home/zqchen/code/mask_dit/results/002-mask-0.6-60000-8-myDiT-XL-2/checkpoints/$i.pt \
#     --image-size 256 \
#     --num-pattern 8 \
#     --hc-path equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
#     --scale 0 1 2 3 4 \
#     --save_path /share/project/dataset/cfgnew_png/dit_fid_imagenet_czq \
#     --idx 0 \

done


# # 设置condition列表
# for i in 0 1 2 3
# do
#   echo " $i "
# # 启动分布式训练
# torchrun \
#     --nproc_per_node=${NGPUS} \
#     --nnodes=${WORLD_SIZE} \
#     --node_rank=${RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     fid_ddp.py \
#     --cfg-scale $i \
#     --ckpt /home/zqchen/code/mask_dit/results/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/5000000.pt \
#     --image-size 256 \
#     --num-pattern 8 \
#     --hc-path equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
#     --scale $i 4 \
#     --save_path /share/project/dataset/dit_fid_imagenet_czq \
#     --idx $((i + 2))
# done

# # 启动分布式训练
# torchrun \
#     --nproc_per_node=${NGPUS} \
#     --nnodes=${WORLD_SIZE} \
#     --node_rank=${RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     fid_ddp.py \
#     --cfg-scale $i \
#     --ckpt /home/zqchen/code/mask_dit/results/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/5000000.pt \
#     --image-size 256 \
#     --num-pattern 8 \
#     --hc-path equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1 \
#     --scale 4 \
#     --save_path /share/project/dataset/dit_fid_imagenet_czq \
#     --idx 6