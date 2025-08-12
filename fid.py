# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from train import center_crop_arr
from torchvision import transforms, models, datasets
from PIL import Image
from hypercolumn import HyperColumnLGN
import argparse
import json
import os


def make_fid_images():
    file_path = '/share/project/dataset/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/ILSVRC2012_devkit_t12/data/val_label.txt'
    integer_array = []
    with open(file_path, 'r') as file:
        for line in file:
            integer_array.append(int(line.strip()))

    folder_path = '/share/project/dataset/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/val'
    jpeg_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.JPEG'):
            jpeg_files.append(file_name)
    jpeg_files.sort()
    return integer_array, jpeg_files
    

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    num_pattern = args.num_pattern * 2 if args.catGP else args.num_pattern
    model = DiT_models[args.model](
        input_size=2,
        num_classes=args.num_classes,
        num_pattern=num_pattern,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    moto_path = "/share/project/dataset/dit_fid_imagenet_czq"
    save_path = os.path.join(moto_path, "/".join(ckpt_path.split("/")[-2:])[:-3]+"_"+str(args.cfg_scale))
    # items = ["img", "zero", "all", "0", "1", "2", "3", "4"]
    items = ["img", "zero"]
    for i in items:
        temp_path = os.path.join(save_path, i)
        os.makedirs(temp_path, exist_ok=True)
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"/home/zqchen/code/mask_dit/pretrain_model/vae-{args.vae}").to(device)
    hpc = HyperColumnLGN(restore_ckpt='/home/zqchen/code/mask_dit/hypercolumn/checkpoint/imagenet/' + args.hc_path).to(device)
    labels, img_name = make_fid_images()
    cuda_num = args.cuda_num
    start = 12500 * cuda_num
    end = 12500 * (cuda_num + 1)
    # start = 13286 + cuda_num*684
    # end = 13286 + (cuda_num+1)*684
    step_len = 20
    # start = 49776
    # end = 50000
    # step_len = 16
    while start < end:
        # Labels to condition the model with (feel free to change):
        class_labels = labels[start:start+step_len]
        num_hc_tensor = [[0,6]] * step_len

        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 4, 1364, device=device)
        y = torch.tensor(class_labels, device=device)
        num_hc_tensor = torch.tensor(num_hc_tensor, device=device)

        # Make hc conditoin:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        hc_x = []
        for i in range(step_len):
            image_path = os.path.join(args.data_path, args.train_or_val, img_name[start+i])
            img = Image.open(image_path).convert("RGB")
            img = transform(img).to(device).unsqueeze(0)
            hc_x.append(img)
            save_image(img, os.path.join(save_path, "img", img_name[start+i]), normalize=True, value_range=(-1, 1))
        hc_x = torch.cat(hc_x, 0)
        _, hc_x = hpc.make_GP_and_DoGfeature(hc_x, catGP=args.catGP)
        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        hc_x = [torch.cat([item, item], 0) for item in hc_x]
        num_hc_tensor = torch.cat([num_hc_tensor, num_hc_tensor], 0)
        scale = [[0]*5, [1]*5, [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]
        name = items[1:]

        model_kwargs = dict(y=y, num_hc=num_hc_tensor, cfg_scale=args.cfg_scale, hc=hc_x)

        for i in range(len(name)):
            model_kwargs["num_scale"] = torch.tensor(scale[i], device=device).unsqueeze(0).repeat(2*n, 1)
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            b, c, s = samples.shape
            gen_size = (args.image_size // 8) ** 2

            temp = samples[:,:,-gen_size:]
            temp = temp.reshape(shape=(b, c, args.image_size // 8, args.image_size // 8))
            temp = vae.decode(temp / 0.18215).sample
            for j in range(len(temp)):
                ulti_temp = temp[j].unsqueeze(0)
                save_image(ulti_temp, os.path.join(save_path, name[i], img_name[start+j]), normalize=True, value_range=(-1, 1))
        start += step_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="myDiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=128)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.25)
    parser.add_argument("--num-sampling-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-path", type=str, default='/share/project/dataset/OpenDataLab___ImageNet-1K/raw/ImageNet-1K')
    parser.add_argument("--train-or-val", type=str, choices=["train", "val"], default="val")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--num-pattern", type=int, default=32)
    parser.add_argument("--cuda-num", type=int, default=0)
    parser.add_argument("--hc-path", type=str, default='equ_nv32_vl4_rn1_Vanilla_ks17_norm_RQVQ_level5_256_share_3_12_-1')
    parser.add_argument("--catGP", type=bool, default=True)
    args = parser.parse_args()
    main(args)