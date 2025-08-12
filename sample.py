# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from train import center_crop_arr, downsample_hc
from torchvision import transforms
from PIL import Image
from hypercolumn import HyperColumnLGN
import argparse
import json
import os


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    model = DiT_models[args.model](
        input_size=2,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"/home/bsliu/gitprojects/dit/hc_and_scale/pretrain‗model/vae-{args.vae}").to(device)
    hpc = HyperColumnLGN().to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # class_labels = [0,1,2,3,4,5,6,7]
    # class_labels = [979]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, 1364, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    # import pdb;pdb.set_trace()
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample

    b, c, s = samples.shape
    output, start = [], 0
    for i in range(5):
        temp = samples[:,:,start:start+4**(i+1)]
        temp = temp.reshape(shape=(b, c, 2**(i+1), 2**(i+1)))
        start += 4**(i+1)
        temp = vae.decode(temp / 0.18215).sample
        output.append(temp)
    # import pdb;pdb.set_trace()
    # output = hpc.recon_up(output)

        # Save and display images:
        # save_image(temp[0:1], f"/home/bsliu/gitprojects/dit/myDiT/results/000-myDiT-XL-2/{i}_0024000_{class_labels[0]}.png", nrow=4, normalize=True, value_range=(-1, 1))
        # save_image(temp[5:6], f"/home/bsliu/gitprojects/dit/myDiT/results/000-myDiT-XL-2/{i}_0024000_{class_labels[5]}.png", nrow=4, normalize=True, value_range=(-1, 1))
        save_image(temp, f"/home/bsliu/gitprojects/dit/myDiT/results/001-myDiT-XL-2/result/2000/{i}.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="myDiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=128)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='./results/006-myDiT-XL-2/checkpoints/0070000.pt',
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
