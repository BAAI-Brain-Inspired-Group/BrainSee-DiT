import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from torchvision import transforms
from hypercolumn.vit_pytorch.train_V1_sep_new import *
import numpy as np
from einops import rearrange, repeat
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torch.serialization
from argparse import Namespace


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


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


class HyperColumnLGN(nn.Module):
    def __init__(self, restore_ckpt='/home/bsliu/gitprojects/dit/hc_and_scale/hypercolumn/checkpoint/imagenet/equ_nv8_vl4_rn1_Bipolar_1x1_norm_GP_RQVQ_level5_256_share_4_9_-1'):
        super().__init__()
        args = torch.load(restore_ckpt + '/args.pth', weights_only=False)
        ckpt = torch.load(restore_ckpt + '/last.ckpt', weights_only=False)
        self.column = Column_GP(args)
        # import pdb;pdb.set_trace()
        self.load_state_dict(ckpt['state_dict'], strict=False)
        self.column.share_column()
        self.lgn_ende = self.column.columns[0].lgn_ende[0].eval()
        self.lgn_ende.train = disabled_train
        for param in self.lgn_ende.parameters():
            param.requires_grad = False

        self.scale_each = np.array([1.0971, 0.0784, 0.0810, 0.0862, 0.1451, 0.0871, 0.0865, 0.0737, 0.1232,
                                    0.3183, 0.1491, 0.0968, 0.0731, 0.0721, 0.0960, 0.1580, 0.5251, 0.0892,
                                    0.1090, 0.3065, 0.1169, 0.1045, 0.2694, 0.7812, 0.4649, 0.0807, 0.0925,
                                    0.3577, 0.0829, 0.0949, 0.2093, 1.2422])
        # self.scale_all = 0.3757
        self.post_norm = transforms.Normalize(np.array([0. for i in range(128)]),repeat(self.scale_each,'n -> (n c)',c=4))
        self.post_denorm = transforms.Normalize(np.array([0. for i in range(128)]),repeat(1./self.scale_each,'n -> (n c)',c=4))
        self.scale_all = 1.

    def single_max_min_norm(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def encode(self,x):
        return self.img_to_feat_GP(x)[0]/self.scale_all
        # return self.post_norm(self.img_to_feat_GP(x)[0])
    
    def decode(self,feat):
        return self.lgn_ende.deconv(feat)*self.scale_all
        # return self.post_denorm(self.lgn_ende.deconv(feat))

    def forward(self, x):
        # import pdb;pdb.set_trace()
        output = self.encode(x)
        output = self.decode(output)
        output = self.single_max_min_norm(output)
        return output
    
    def img_to_DoG(self,img):
        # 将图像转化为高斯差分
        img_GP,img_DoG = self.column.get_img_DoG(img)
        # return img_GP, img_DoG
        return [img_GP[-1]]+img_DoG[::-1]
    
    def make_GP_and_DoGfeature(self, img, catGP=False):
        img_GP = [img]
        for i in range(5):
            img_GP.append(self.column.GP.downPy(img_GP[-1]))
        img_DoG = []
        for i in range(4):
            img_DoG.append(img_GP[i+1]-self.column.GP.upPy(img_GP[i+2]))

        img_seq = [img_GP[-1]]+img_DoG[::-1]
        feat_seq = []
        for im,im_gp in zip(img_seq,img_GP[::-1][0:-1]):
            if catGP:
                feat_seq.append(torch.cat([self.lgn_ende(im),self.lgn_ende(im_gp)],dim=1))
            else:
                feat_seq.append(self.lgn_ende(im))
        return img_GP[::-1][1:], feat_seq
    
    def make_features(self, x, index, n, pattern=32):
        x = self.lgn_ende(x)
        b, c, h, w = x.shape
        hc_x_zeroed = torch.zeros(n*b, pattern, c // pattern, h, w).to(x.device)
        x = x.view(b, pattern, c // pattern, h, w)
        for i in range(b):
            for j in range(n):
                hc_x_zeroed[i*n+j, index[i][j], :, :, :] = x[i, index[i][j], :, :, :]
        output = self.decode(hc_x_zeroed.view(n*b, c, h, w))
        # output = self.decode(x.view(b, c, h, w))
        output = self.single_max_min_norm(output)
        # output1 = output.view(B//n, n, C, H, W)
        return output
    
    def img_to_GP(self,img):
        img_GP,img_DoG = self.column.get_img_DoG(img)
        return img_GP[::-1]
    
    def img_to_feat_GP(self,img):
        img_seq = self.img_to_GP(img)
        feat_seq = []
        lgn = self.lgn_ende
        for im in img_seq:
            feat_seq.append(lgn(im))

        return feat_seq
    
    def img_to_feat_rq(self,img):
        # 将图像转化为特征序列
        img_seq = self.img_to_DoG(img)
        feat_seq = []
        lgn = self.lgn_ende
        for im in img_seq:
            feat_seq.append(lgn.codebook.straight_through(lgn(im))[0])
        
        # import pdb;pdb.set_trace()
        return feat_seq
    
    def img_to_feat(self,img):
        # 将图像转化为高斯差分的特征序列
        img_seq = self.img_to_DoG(img)
        feat_seq = []
        lgn = self.lgn_ende
        for im in img_seq:
            feat_seq.append(lgn(im))
        return feat_seq
    
    def feat_to_img(self,feat_seq):
        # 将特征序列转化为图像
        img_seq = []
        lgn = self.lgn_ende
        for feat in feat_seq:
            img_seq.append(lgn.deconv(feat))
        return img_seq
    
    def img_to_feat_2(self,img):
        # 将图像转化为特征序列,使用上一个尺度创建上采样后和ground残差进行特征提取
        img_GP, img_DoG = self.column.get_img_DoG(img)
        feat_seq = []
        lgn = self.lgn_ende

        feat_seq.append(lgn(img_GP[-1]))
        img0 = lgn.deconv(feat_seq[-1])

        for i in range(self.column.n_level-1):
            j = self.n_level-2-i
            img0 = self.column.GP.upPy(img0)
            res = img_GP[j] - img0
            feat_seq.append(lgn(res))
            img0 += lgn.deconv(feat_seq[-1])

        return feat_seq
    
    def recon_img(self,feat_seq):
        # 将特征序列转化为图像
        img_seq = []
        lgn = self.lgn_ende

        img_seq = []
        img_seq.append(lgn.deconv(feat_seq[0]))
        for feat in feat_seq[1:]:
            img_seq.append(self.column.GP.upPy(img_seq[-1]) + lgn.deconv(feat))
        return img_seq
    
    def recon_up(self, img_seq):
        output = img_seq[0]
        for i in range(1, len(img_seq)):
            output = self.column.GP.upPy(output) + img_seq[i]
        return output


if __name__ == "__main__":
    model = HyperColumnLGN().to('cuda:0')
    image = Image.open('/home/bsliu/gitprojects/dit/hc_and_scale/hypercolumn/result/img.png').convert('RGB')
    transform1 = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_tensor = transform1(image)
    input_tensor = input_tensor.to('cuda:0')
    input_tensor = input_tensor.unsqueeze(0).repeat(8,1,1,1)

    num_hc_tensor = [[0], [1], [2], [3], [4], [5], [6], [7]]
    num_hc_tensor = torch.tensor(num_hc_tensor, device='cuda:0')

    output = model.make_features(input_tensor, num_hc_tensor, 1)
    image.save("./result/aa.jpg")
    save_image(output, "./result/bb.jpg", nrow=4, normalize=True, value_range=(0, 1))

    # o1 = model.img_to_GP(input_tensor)
    # import pdb;pdb.set_trace()
    # for i in range(len(o1)):
    #     output = transform2(model.single_max_min_norm(o1[i].squeeze(0)))
    #     output.save(f"GP{i}.jpg")
    
    # output = model.recon_up(o1)
    # output = transform2(model.single_max_min_norm(output.squeeze(0)))
    # output.save(f"./result/ulti.jpg")
