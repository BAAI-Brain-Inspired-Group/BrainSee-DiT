#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh

conda activate dit

# for i in '0010000' '0020000' '0040000' '0060000' '0080000' '0100000' '0120000'
# for i in '0140000' '0160000' '0180000' '0200000' '0220000' '0240000' '0260000'
# for i in '0300000' '0350000' '0400000' '0450000' '0500000' '0550000' '0600000' '0650000'
# for i in '0550000' '0600000' '0650000'
for i in '6930000'
do
  echo " $i "

# python sample2.py --image-size 256 --ckpt "/home/zqchen/code/mask_dit/results/017-mask-1.0-0-8-256x256-myDiT-XL-2/checkpoints/$i.pt"  --num-pattern   8   --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1  --catGP=True

python sample2.py --image-size 256 --ckpt "/home/zqchen/code/mask_dit/results/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/$i.pt"  --num-pattern   8   --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1  --catGP=True

# python sample2.py --image-size 256 --ckpt "/home/zqchen/code/mask_dit/results/016-mask-0.6-60000-8-256x256-myDiT-XL-2/checkpoints/$i.pt"  --num-pattern   8   --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1  --catGP=True

# python sample2.py --image-size 512 --ckpt "/home/zqchen/code/mask_dit/results/004-mask-0.94-60000-8-512x512-myDiT-XL-2/checkpoints/$i.pt"  --num-pattern   8   --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1  --catGP=True

# python sample2.py --image-size 512 --ckpt "/home/zqchen/code/mask_dit/results/005-mask-1.0-0-8-512x512-myDiT-XL-2/checkpoints/$i.pt"  --num-pattern   8   --hc-path   equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1  --catGP=True

done
