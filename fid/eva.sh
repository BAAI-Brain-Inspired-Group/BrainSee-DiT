#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh

conda activate dit


python evaluator.py VIRTUAL_imagenet256_labeled.npz /share/project/dataset/cfgnew_png/dit_fid_imagenet_czq_04/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/6930000_1.0/zero

python evaluator.py VIRTUAL_imagenet256_labeled.npz /share/project/dataset/cfgnew_png/dit_fid_imagenet_czq_04/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/6930000_1.25/zero

python evaluator.py VIRTUAL_imagenet256_labeled.npz /share/project/dataset/cfgnew_png/dit_fid_imagenet_czq_04/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/6930000_1.5/zero


python evaluator.py VIRTUAL_imagenet256_labeled.npz /share/project/dataset/cfgnew_png/dit_fid_imagenet_czq_04/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/6930000_1.0/zero

python evaluator.py VIRTUAL_imagenet256_labeled.npz /share/project/dataset/cfgnew_png/dit_fid_imagenet_czq_04/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/6930000_1.25/zero

python evaluator.py VIRTUAL_imagenet256_labeled.npz /share/project/dataset/cfgnew_png/dit_fid_imagenet_czq_04/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/6930000_1.5/zero

