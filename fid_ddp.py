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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


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


def get_data_range(total_size, rank, world_size):
    """根据rank和world_size计算当前进程的数据范围"""
    per_rank = total_size // world_size
    remainder = total_size % world_size
    
    start = rank * per_rank + min(rank, remainder)
    end = start + per_rank + (1 if rank < remainder else 0)
    
    return start, end


def main(args):
    assert torch.cuda.is_available(), "FID testing currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    # torch.manual_seed(args.seed)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)  # 与原始版本保持一致
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    
    num_pattern = args.num_pattern * 2 if args.catGP else args.num_pattern
    model = DiT_models[args.model](
        input_size=2,
        num_classes=args.num_classes,
        num_pattern=num_pattern,
    )
    
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    moto_path = args.save_path
    save_path = os.path.join(moto_path, "/".join(ckpt_path.split("/")[-3:])[:-3]+"_"+str(args.cfg_scale))
    
    # 只在主进程创建目录
    if rank == 0:
        items = ['img','zero','all','0','1','2','3','4']
        for i in items:
            temp_path = os.path.join(save_path, i)
            os.makedirs(temp_path, exist_ok=True)
    
    # 等待主进程创建完目录
    dist.barrier()
    
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    
    # Use DDP wrapper
    model = DDP(model.to(device), device_ids=[rank])
    
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"/home/zqchen/code/mask_dit/pretrain_model/vae-{args.vae}").to(device)
    hpc = HyperColumnLGN(restore_ckpt='/home/zqchen/code/mask_dit/hypercolumn/checkpoint/imagenet/' + args.hc_path).to(device)
    
    labels, img_name = make_fid_images()
    
    # 根据分布式设置计算数据范围
    start, end = get_data_range(len(labels), rank, dist.get_world_size())
    print(f"Rank {rank}: processing images {start:,} to {end:,} (total: {end-start:,})")
    
    step_len = args.batchsize
    
    while start < end:
        # Labels to condition the model with (feel free to change):
        class_labels = labels[start:start+step_len]
        num_hc_tensor = [[0,6]] * step_len

        # Create sampling noise:
        n = len(class_labels)
        scale = args.scale
        sz = 0
        for i in scale:
            sz += (int(args.image_size/128)*2**i)**2
        z = torch.randn(n, 4, sz, device=device)
        # z = torch.randn(n, 4, 1364, device=device)
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
            # 只在主进程保存原始图像
            # if rank == 0:
            # import pdb;pdb.set_trace()
            # save_image(img, os.path.join(save_path, "img", img_name[start+i][:-4]+'png'), normalize=True, value_range=(-1, 1))
            img = torch.clamp(127.5 * img + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            Image.fromarray(img[0]).save(os.path.join(save_path, "img", img_name[start+i][:-4]+'png'))
        hc_x = torch.cat(hc_x, 0)
        _, hc_x = hpc.make_GP_and_DoGfeature(hc_x, catGP=args.catGP)
        
        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        hc_x = [torch.cat([item, item], 0) for item in hc_x]
        num_hc_tensor = torch.cat([num_hc_tensor, num_hc_tensor], 0)
        scale_ = [[0]*5, [1]*5, [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]
        name = ["zero",'all','0','1','2','3','4']  # 简化，只生成zero类型
        idx = args.idx

        model_kwargs = dict(y=y, num_hc=num_hc_tensor, cfg_scale=args.cfg_scale, hc=hc_x,scale=scale)

        for i in idx:
            model_kwargs["num_scale"] = torch.tensor(scale_[i], device=device).unsqueeze(0).repeat(2*n, 1)
            
            # Generate samples using DDP model
            samples = diffusion.p_sample_loop(
                model.module.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            b, c, s = samples.shape
            gen_size = (args.image_size // 8) ** 2

            temp = samples[:,:,-gen_size:]
            temp = temp.reshape(shape=(b, c, args.image_size // 8, args.image_size // 8))
            temp = vae.decode(temp / 0.18215).sample
            temp = torch.clamp(127.5 * temp + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            
            # Save generated images
            for j in range(len(temp)):
                # ulti_temp = temp[j].unsqueeze(0)
                # save_image(ulti_temp, os.path.join(save_path, name[i], img_name[start+j]), normalize=True, value_range=(-1, 1))
                Image.fromarray(temp[j]).save(os.path.join(save_path, name[i], img_name[start+j][:-4]+'png'))
            
        print(f"Rank {rank}: {start} Saved!")
        start += step_len
    
    print(f"Rank {rank}: Completed!")
    
    # Wait for all processes to finish
    dist.barrier()
    
    if rank == 0:
        print("All processes completed FID image generation!")
    
    cleanup()


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

    parser.add_argument("--batchsize", type=int, default=125)
    parser.add_argument('--scale', 
                    type=int, 
                    nargs='+',        # 至少1个整数
                    default=[0, 1, 2, 3, 4], # 默认值为[1,2,3]
                    help='输入选择的尺度，必须包含尺度4')
    
    parser.add_argument('--idx', 
                    type=int, 
                    nargs='+',        # 至少1个整数
                    default=[0], # 默认值为[0,1,2,3,4,5,6]
                    )    
    parser.add_argument("--save_path", type=str, default="/share/project/dataset/dit_fid_imagenet_czq")
    args = parser.parse_args()
    main(args)