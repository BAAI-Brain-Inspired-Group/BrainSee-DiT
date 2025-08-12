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
    
    num_pattern = args.num_pattern * 2 if args.catGP else args.num_pattern
    model = DiT_models[args.model](
        input_size= int(args.image_size / 128),
        num_classes=args.num_classes,
        num_pattern=num_pattern,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    step = ckpt_path.split("/")[-1].split(".")[0]
    save_path = os.path.join("/".join(ckpt_path.split("/")[:-2]), "results", step)
    os.makedirs(save_path, exist_ok=True)
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"/home/zqchen/code/mask_dit/pretrain_model/vae-{args.vae}").to(device)
    hpc = HyperColumnLGN(restore_ckpt='/home/zqchen/code/mask_dit/hypercolumn/checkpoint/imagenet/' + args.hc_path).to(device)
    # import pdb;pdb.set_trace()
    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # class_labels = [277,332,360,387,277,332,360,387]
    # num_hc_tensor = [[0,31], [15,27], [16,22], [9,24], [8,18], [0,8], [6,7], [23,31]]
    # num_hc_tensor = [[16,27], [16,27], [16,27], [16,27], [16,27], [16,27], [16,27], [16,27]]
    # num_hc_tensor = [[6,7], [6,7], [6,7], [6,7], [6,7], [6,7], [6,7], [6,7]]
    num_hc_tensor = [[0,6], [0,6], [0,6], [0,6], [0,6], [0,6], [0,6], [0,6]]
    # num_hc_tensor = [[5] for i in range(8)]
    num_hc = 2

    # Create sampling noise:
    n = len(class_labels)
    scale = [2,4]
    sz = 0
    for i in scale:
        sz += (int(args.image_size/128)*2**i)**2
    z = torch.randn(n, 4, sz, device=device)
    # z = torch.randn(n, 4, 1364*int(args.image_size/256)**2, device=device)
    
    y = torch.tensor(class_labels, device=device)
    # y = torch.tensor([1000 for i in class_labels],device=device)
    num_hc_tensor = torch.tensor(num_hc_tensor, device=device)

    # Make hc conditoin:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    hc_x = []
    with open(os.path.join(args.data_path, 'imagenet_class_index.json'), 'r') as file:
        folders = json.load(file)
    for label_index in class_labels:
        folder_path = folders[str(label_index)][0]
        # images = [f for f in os.listdir(os.path.join(args.data_path, args.train_or_val, folder_path)) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP'))]
        # images = [f for f in os.listdir(os.path.join("/share/project/dataset/ImageNet-Sketch/raw/sketch", folder_path)) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP'))]
        images = [f for f in os.listdir(os.path.join("/share/project/dataset/val_jibuli", folder_path)) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP'))]
        # images = [f for f in os.listdir(os.path.join("/share/project/dataset/val_niantu", folder_path)) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP'))]
        # images = [f for f in os.listdir(os.path.join("/share/project/dataset/val_rickandmorty", folder_path)) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP'))]
        # images = [f for f in os.listdir(os.path.join("/share/project/dataset/val_dongman", folder_path)) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP'))]
        random_image = images[torch.randint(0, len(images), (1,)).item()]
        # random_image = images[0]
        # image_path = os.path.join(args.data_path, args.train_or_val, folder_path, random_image)
        # image_path = os.path.join("/share/project/dataset/ImageNet-Sketch/raw/sketch", folder_path, random_image)
        image_path = os.path.join("/share/project/dataset/val_jibuli", folder_path, random_image)
        # image_path = os.path.join("/share/project/dataset/val_niantu", folder_path, random_image)
        # image_path = os.path.join("/share/project/dataset/val_rickandmorty", folder_path, random_image)
        # image_path = os.path.join("/share/project/dataset/val_dongman", folder_path, random_image)
        img = Image.open(image_path).convert("RGB")
        img = transform(img).to(device).unsqueeze(0)
        hc_x.append(img)
    hc_x = torch.cat(hc_x, 0)
    # hc_output = hpc.make_features(hc_x, num_hc_tensor, num_hc, args.num_pattern)
    # import pdb; pdb.set_trace()
    save_image(hc_x, os.path.join(save_path, "img_jibuli.png"), nrow=4, normalize=True, value_range=(-1, 1))
    # save_image(hc_output, os.path.join(save_path, "hc.png"), nrow=4, normalize=True, value_range=(0, 1))
    _, hc_x = hpc.make_GP_and_DoGfeature(hc_x,catGP=args.catGP)
    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    # y = torch.cat([y, y], 0)
    hc_x = [torch.cat([item, item], 0) for item in hc_x]
    num_hc_tensor = torch.cat([num_hc_tensor, num_hc_tensor], 0)
    scale_hc = [[0]*5, [1]*5, [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]
    name = ["zero", "all", "0", "1", "2", "3", "4"]
    idx = [4]

    model_kwargs = dict(y=y, num_hc=num_hc_tensor, cfg_scale=args.cfg_scale, hc=hc_x,scale=scale)
    print(model.mask_token[0][0][0][:100])

    for i in idx:
        model_kwargs["num_scale"] = torch.tensor(scale_hc[i], device=device).unsqueeze(0).repeat(2*n, 1)
        # model_kwargs["num_scale"] = torch.cat([torch.tensor(scale_hc[i], device=device).unsqueeze(0).repeat(n, 1),torch.tensor(scale_hc[0], device=device).unsqueeze(0).repeat(n, 1)],dim=0)
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        b, c, s = samples.shape
        gen_size = (args.image_size // 8) ** 2

        temp = samples[:,:,-gen_size:]
        temp = temp.reshape(shape=(b, c, args.image_size // 8, args.image_size // 8))
        temp = vae.decode(temp / 0.18215).sample

        save_image(temp, os.path.join(save_path, f"{name[i]}_{step}_2_jibuli.png"), nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="myDiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=128)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-path", type=str, default='/share/project/dataset/OpenDataLab___ImageNet-1K/raw/ImageNet-1K')
    parser.add_argument("--train-or-val", type=str, choices=["train", "val"], default="val")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--num-pattern", type=int, default=32)
    parser.add_argument("--hc-path", type=str, default='equ_nv32_vl4_rn1_Vanilla_ks17_norm_RQVQ_level5_256_share_3_12_-1')
    parser.add_argument("--catGP", type=bool, default=False)
    args = parser.parse_args()
    main(args)