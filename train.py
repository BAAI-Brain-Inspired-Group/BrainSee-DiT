# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import torch.multiprocessing as mp

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from hypercolumn import HyperColumnLGN


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def single_max_min_norm(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def sample_hc_id(batch_size, set_size, num_range, device):
    sets = torch.randint(0, num_range, (batch_size, set_size))
    mask = (sets[:, :, None] == sets[:, None, :]).sum(dim=-1)
    while mask.sum(dim=1).max() > set_size:
        duplicate_indices = mask.sum(dim=1) > set_size
        sets[duplicate_indices] = torch.randint(0, num_range, (duplicate_indices.sum().item(), set_size))
        mask = (sets[:, :, None] == sets[:, None, :]).sum(dim=-1)
    return sets.to(device)

# sample_sc = [[1,1,1,1,1], [0,0,0,0,1], [0,0,0,1,0], [0,0,1,0,0], [0,1,0,0,0], [1,0,0,0,0], [0,0,0,0,0]]
# scale_num = [8,8,8,8,8,8]
# t_flag = [0]

# def sample_scale_2(bsz, t, p, dt):
#     total = sum(scale_num)
#     if total / bsz < p:
#         return scale_num + [bsz - total]
#     if t % int(dt / (bsz*(1-p))) == 0 and t != 0:
#         scale_num[t_flag[0] % 6] -= 1
#         t_flag[0] += 1
#         return scale_num + [bsz - total + 1]
#     return scale_num + [bsz - total]


# def sample_everything(batch_size, num_hc, num_range, step, t_sum, p, device):
#     hc = sample_hc_id(batch_size, num_hc, num_range, device)
#     total = sample_scale_2(batch_size, step, p, t_sum)
#     vectors = []
#     for i in range(7):
#         vectors += [sample_sc[i]] * total[i]
#     np.random.shuffle(vectors)
#     return hc, torch.tensor(vectors, device=device), total


def sample_scale_id2(bsz, p_zero, t, time, device="cpu"):
    p_zero = p_zero if time == 0 else (p_zero if t > time else p_zero * t / time)
    p_other = 1 - p_zero
    bsz_ = 1
    random_choice = torch.multinomial(torch.tensor([p_zero, p_other * 5 / 6, p_other / 6]), bsz_, replacement=True)
    all_zeros = torch.zeros(bsz_, 5)
    all_ones = torch.ones(bsz_, 5)
    one_hot_indices = torch.randint(0, 5, (bsz_,))
    one_hot_vectors = torch.zeros(bsz_, 5)
    one_hot_vectors[torch.arange(bsz_), one_hot_indices] = 1
    result = torch.where(random_choice.unsqueeze(1) == 0, all_zeros, 
                         torch.where(random_choice.unsqueeze(1) == 1, one_hot_vectors, all_ones))

    # import pdb;pdb.set_trace()
    result = result.repeat(bsz,1)
    if 4 not in one_hot_indices:
        one_hot_indices = one_hot_indices.repeat(2)
        one_hot_indices[1] = 4
    # import pdb;pdb.set_trace()
    return result.to(device), one_hot_indices.to(device), torch.bincount(random_choice, minlength=3).tolist()


def sample_everything(batch_size, num_hc, num_range, p_zero, t, time, device):
    hc = sample_hc_id(batch_size, num_hc, num_range, device)
    scale_hc, scale, total = sample_scale_id2(batch_size, p_zero, t, time, device)
    return hc, scale_hc, scale, total

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-mask-{args.p_zero}-{args.time}-{args.global_batch_size}-{args.image_size}x{args.image_size}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    num_pattern = args.num_pattern * 2 if args.catGP else args.num_pattern
    model = DiT_models[args.model](
        input_size= int(args.image_size / 128),
        num_classes=args.num_classes,
        num_pattern=num_pattern,
    )
    if args.ckpt is not None:
        state_dict = find_model(args.ckpt)
        model.load_state_dict(state_dict, strict=False)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device])
    # model = DDP(model.to(device), device_ids=[device])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"/home/zqchen/code/mask_dit/pretrain_model/vae-{args.vae}").to(device)
    hpc = HyperColumnLGN(restore_ckpt='/home/zqchen/code/mask_dit/hypercolumn/checkpoint/imagenet/' + args.hc_path)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        # batch_size=int(args.global_batch_size // dist.get_world_size()),
        batch_size=int(args.global_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"{len(loader)} steps per epoch, Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    num_hc = args.num_hc ###################################

    # Variables for monitoring/logging purposes:
    train_steps = args.numstep
    # train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    scale_total = [0,0,0]

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                gaussian_list, dog_feature_list = hpc.make_GP_and_DoGfeature(x,catGP=args.catGP) ###########################
                x = [vae.encode(item).latent_dist.sample().mul_(0.18215) for item in gaussian_list]
            t = torch.randint(0, diffusion.num_timesteps, (y.shape[0],), device=device)
            hc_tensor, scale_hc_tensor, scale_tensor, scale_data = sample_everything(y.shape[0], num_hc, num_pattern, args.p_zero, train_steps, args.time, device) ######################
            if args.fullscale:
                scale_tensor = torch.IntTensor([0, 1, 2, 3, 4]).to(device)
            # import pdb;pdb.set_trace()
            model_kwargs = dict(y=y, num_hc=hc_tensor, num_scale=scale_hc_tensor, hc=dog_feature_list, scale=scale_tensor) ################################
            # import pdb;pdb.set_trace()
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            scale_total = [i + j for i, j in zip(scale_total, scale_data)]
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) loss: {avg_loss:.4f}, epoch: {(train_steps/len(loader)):.4f}, scale: {scale_total}, speed: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                scale_total = [0,0,0]
                start_time = time()
                # import pdb;pdb.set_trace()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()



if __name__ == "__main__":
    # print(sample_scale_id2(48, 0.99, 202000, 60000))
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/share/project/dataset/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/train')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="myDiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=48)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--num-hc", type=int, default=2)
    parser.add_argument("--num-pattern", type=int, default=32)
    parser.add_argument("--hc-path", type=str, default='equ_nv32_vl4_rn1_Vanilla_ks17_norm_RQVQ_level5_256_share_3_12_-1')
    parser.add_argument("--p-zero", type=float, default=0.99)
    parser.add_argument("--time", type=int, default=60000)
    parser.add_argument("--catGP", type=bool, default=False)
    parser.add_argument("--numstep", type=int, default=0)
    parser.add_argument("--fullscale", action='store_true')
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
    # mp.spawn(main, args=(args.world_size,args), nprocs=args.world_size, join=True)