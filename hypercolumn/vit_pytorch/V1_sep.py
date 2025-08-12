import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import random
import pdb

import torchvision

from einops import rearrange, repeat,reduce
from einops.layers.torch import Rearrange
from hypercolumn.utils.tools import *
from hypercolumn.vqvae.modules import VQEmbedding,HRVQEmbedding
# from utils import *


class Lgn_ende(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg = arg
        self.channel = arg.channel
        self.n_vector = arg.n_vector
        self.vector_length = arg.vector_length
        self.lgn_kernel_size = arg.lgn_kernel_size
        self.lgn_stride = arg.lgn_stride
        self.lgn_padding = arg.lgn_padding
        self.activate_fn = arg.activate_fn

        self.conv = nn.Conv2d(self.channel,self.n_vector*self.vector_length,self.lgn_kernel_size,self.lgn_stride,self.lgn_padding,bias=False)
        # self.bn = nn.BatchNorm2d(arg.n_vector*arg.vector_length,momentum=0.1,affine=False)
        self.to_column = nn.Identity()
        self.to_channel = nn.Identity()
        if self.activate_fn is None:
            self.nonlinear = nn.Identity()
        elif self.activate_fn == 'gelu':
            self.nonlinear = nn.GELU()
        elif self.activate_fn == 'relu':
            self.nonlinear = nn.ReLU()
        # self.nonlinear = nn.Identity()

        # self.bn = nn.BatchNorm2d(arg.n_vector,momentum=0.1,affine=False)
        # self.to_column = Rearrange('b (nv vl) h w -> b nv (vl h) w',vl=arg.vector_length)
        # self.to_channel = Rearrange('b nv (vl h) w -> b (nv vl) h w',vl=arg.vector_length)
        self.bn = nn.Identity()
        
        self.apply(weights_init)

    def forward(self,x):
        return self.nonlinear(self.to_channel(self.bn(self.to_column(self.conv(x)))))
        # return F.gelu(self.bn(self.conv(x)))
        # return self.conv(x)
    
    def deconv(self,f):
        return F.conv_transpose2d(f,self.conv.weight,stride=self.lgn_stride,padding=self.lgn_padding)
        # # print('Lgn_ende deconv running_var:',self.bn.running_var.size(),', running_mean:',self.bn.running_mean.size())
        # return F.conv_transpose2d(self.to_channel(self.to_column(f)*self.bn.running_var.view(1,-1,1,1)+self.bn.running_mean.view(1,-1,1,1)),self.conv.weight,stride=self.arg.lgn_stride,padding=self.arg.lgn_padding)

    def deconv_group(self,f):
        img_group = F.conv_transpose2d(f,self.conv.weight,stride=self.lgn_stride,padding=self.lgn_padding,groups=self.n_vector)
        img = reduce(img_group,'b (nv c) h w -> b c h w','sum',nv=self.n_vector)
        return img_group,img
    
    def fullconv(self,x):
        return self.nonlinear(self.to_channel(self.bn(self.to_column(F.conv2d(x,self.conv.weight,padding=self.lgn_padding)))))
    
    def fulldeconv(self,f):
        return F.conv_transpose2d(f,self.conv.weight,padding=self.lgn_padding)/self.lgn_stride**2
    
    def fulldeconv_group(self,f):
        return F.conv_transpose2d(f,self.conv.weight,padding=self.lgn_padding,groups=self.n_vector)/self.lgn_stride**2
    
    def conv_stride(self,x):
        return self.conv(x)
    
class Gonlin(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg = arg
        self.chl = arg.channel
        self.nv = arg.n_vector
        self.vl = arg.vector_length
        self.eye = nn.Parameter(torch.eye(self.nv*self.vl).unsqueeze(2).unsqueeze(3), requires_grad=False)
        # self.eye = torch.eye(self.nv*self.vl).unsqueeze(2).unsqueeze(3)

class Bipolar(Gonlin):
    def __init__(self,arg):
        super().__init__(arg)
        self.c1_1 = nn.Conv2d(self.chl,self.nv,5,2,2,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c1_2_1 = nn.Conv2d(self.chl,self.nv,5,2,2,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c1_2_2 = nn.Conv2d(self.nv,self.nv,5,1,2,bias=False,groups=self.nv) # w:[nv,1,5,5] input:[b,nv,h,w] output:[b,nv,h,w]
        self.c2 = nn.Conv2d(self.nv,self.nv*self.vl,5,2,2,bias=False,groups=self.nv) # w:[nvvl,1,5,5] input:[b,nv,h,w] output:[b,nvvl,h,w]

        self.get_weight()
    
    def get_weight(self):
        out = F.conv_transpose2d(self.eye,self.c2.weight,stride=self.c2.stride,padding=0,groups=self.c2.groups)
        out = F.pad(out,(2,2,2,2))
        out1 = F.conv_transpose2d(out,self.c1_1.weight,stride=self.c1_1.stride,padding=0,groups=self.c1_1.groups)
        out2 = F.conv_transpose2d(out,self.c1_2_2.weight,stride=self.c1_2_2.stride,padding=2,groups=self.c1_2_2.groups)
        out2 = F.conv_transpose2d(out2,self.c1_2_1.weight,stride=self.c1_2_1.stride,padding=0,groups=self.c1_2_1.groups)
        self.weight = out1 + out2

        return self.weight
    
class Bipolar_ks3(Gonlin):
    def __init__(self,arg):
        super().__init__(arg)
        print('Bipolar:',self.chl,self.nv)
        self.c1_1 = nn.Conv2d(self.chl,self.nv,3,2,2,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c1_2_1 = nn.Conv2d(self.chl,self.nv,3,2,2,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c1_2_2 = nn.Conv2d(self.nv,self.nv,3,1,2,bias=False,groups=self.nv) # w:[nv,1,5,5] input:[b,nv,h,w] output:[b,nv,h,w]
        self.c2 = nn.Conv2d(self.nv,self.nv*self.vl,5,2,2,bias=False,groups=self.nv) # w:[nvvl,1,5,5] input:[b,nv,h,w] output:[b,nvvl,h,w]

        self.get_weight()
    
    def get_weight(self):
        out = F.conv_transpose2d(self.eye,self.c2.weight,stride=self.c2.stride,padding=0,groups=self.c2.groups)
        out = F.pad(out,(1,1,1,1))
        out1 = F.conv_transpose2d(out,self.c1_1.weight,stride=self.c1_1.stride,padding=0,groups=self.c1_1.groups)
        out2 = F.conv_transpose2d(out,self.c1_2_2.weight,stride=self.c1_2_2.stride,padding=1,groups=self.c1_2_2.groups)
        out2 = F.conv_transpose2d(out2,self.c1_2_1.weight,stride=self.c1_2_1.stride,padding=0,groups=self.c1_2_1.groups)
        self.weight = out1 + out2

        return self.weight
    
# class Bipolar_1x1(Gonlin):
#     def __init__(self,arg):
#         super().__init__(arg)
#         self.c0_1 = nn.Conv2d(self.chl,self.nv,1,1,0,bias=False)
#         self.c0_2 = nn.Conv2d(self.chl,self.nv,1,1,0,bias=False)
#         self.c1_1 = nn.Conv2d(self.nv,self.nv,5,2,2,bias=False,groups=self.nv)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
#         self.c1_2_1 = nn.Conv2d(self.nv,self.nv,5,2,2,bias=False,groups=self.nv) # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
#         self.c1_2_2 = nn.Conv2d(self.nv,self.nv,5,1,2,bias=False,groups=self.nv) # w:[nv,1,5,5] input:[b,nv,h,w] output:[b,nv,h,w]
#         self.c2 = nn.Conv2d(self.nv,self.nv*self.vl,5,2,2,bias=False,groups=self.nv) # w:[nvvl,1,5,5] input:[b,nv,h,w] output:[b,nvvl,h,w]

#         self.get_weight()
    
#     def get_weight(self):
#         out = F.conv_transpose2d(self.eye,self.c2.weight,stride=self.c2.stride,padding=0,groups=self.c2.groups)
#         out = F.pad(out,(2,2,2,2))
#         out1 = F.conv_transpose2d(out,self.c1_1.weight,stride=self.c1_1.stride,padding=0,groups=self.c1_1.groups)
#         out1 = F.conv_transpose2d(out1,self.c0_1.weight,stride=self.c0_1.stride,padding=0,groups=self.c0_1.groups)
#         out2 = F.conv_transpose2d(out,self.c1_2_2.weight,stride=self.c1_2_2.stride,padding=2,groups=self.c1_2_2.groups)
#         out2 = F.conv_transpose2d(out2,self.c1_2_1.weight,stride=self.c1_2_1.stride,padding=0,groups=self.c1_2_1.groups)
#         out2 = F.conv_transpose2d(out2,self.c0_2.weight,stride=self.c0_2.stride,padding=0,groups=self.c0_2.groups)
#         self.weight = out1 + out2

#         return self.weight
    
class Bipolar_1x1(Gonlin):
    def __init__(self,arg):
        super().__init__(arg)
        self.c0_1 = nn.Conv2d(self.chl,self.nv,1,1,0,bias=False)
        self.c0_2 = nn.Conv2d(self.chl,self.nv,1,1,0,bias=False)
        self.c1_1 = nn.Conv2d(self.nv,self.nv,5,2,2,bias=False,groups=self.nv)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c1_2_1 = nn.Conv2d(self.nv,self.nv,5,2,2,bias=False,groups=self.nv)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c1_2_2 = nn.Conv2d(self.nv,self.nv,5,1,2,bias=False,groups=self.nv) # w:[nv,1,5,5] input:[b,nv,h,w] output:[b,nv,h,w]
        self.c2 = nn.Conv2d(self.nv,self.nv*self.vl,5,2,2,bias=False,groups=self.nv) # w:[nvvl,1,5,5] input:[b,nv,h,w] output:[b,nvvl,h,w]

        self.get_weight()
    
    def get_weight(self):
        out = F.conv_transpose2d(self.eye,self.c2.weight,stride=self.c2.stride,padding=0,groups=self.c2.groups)
        out = F.pad(out,(2,2,2,2))
        out1 = F.conv_transpose2d(out,self.c1_1.weight,stride=self.c1_1.stride,padding=0,groups=self.c1_1.groups)
        out1 = F.conv_transpose2d(out1,self.c0_1.weight,stride=self.c0_1.stride,padding=0,groups=self.c0_1.groups)
        out2 = F.conv_transpose2d(out,self.c1_2_2.weight,stride=self.c1_2_2.stride,padding=2,groups=self.c1_2_2.groups)
        out2 = F.conv_transpose2d(out2,self.c1_2_1.weight,stride=self.c1_2_1.stride,padding=0,groups=self.c1_2_1.groups)
        out2 = F.conv_transpose2d(out2,self.c0_2.weight,stride=self.c0_2.stride,padding=0,groups=self.c0_2.groups)
        self.weight = out1 + out2

        return self.weight
    
class Vanilla(Gonlin):
    def __init__(self,arg):
        super().__init__(arg)
        self.c1 = nn.Conv2d(self.chl,self.nv*self.vl,arg.lgn_kernel_size,arg.lgn_stride,arg.lgn_padding,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]

        self.get_weight()
    
    def get_weight(self):
        out = F.conv_transpose2d(self.eye,self.c1.weight,stride=self.c1.stride,padding=0,groups=self.c1.groups)
        self.weight = out

        return self.weight
    
class Vanilla_2layer(Gonlin):
    def __init__(self,arg):
        super().__init__(arg)
        self.c1 = nn.Conv2d(self.chl,self.nv,5,2,2,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c2 = nn.Conv2d(self.nv,self.nv*self.vl,5,2,2,bias=False,groups=self.nv)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]

        self.get_weight()
    
    def get_weight(self):
        out = F.conv_transpose2d(self.eye,self.c2.weight,stride=self.c2.stride,padding=0,groups=self.c2.groups)
        out = F.conv_transpose2d(out,self.c1.weight,stride=self.c1.stride,padding=0,groups=self.c1.groups)
        self.weight = out

        return self.weight
    

class Vanilla_2layer_nogroup(Gonlin):
    def __init__(self,arg):
        super().__init__(arg)
        self.c1 = nn.Conv2d(self.chl,self.nv,5,2,2,bias=False)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]
        self.c2 = nn.Conv2d(self.nv,self.nv*self.vl,5,2,2,bias=False,groups=1)  # w:[nv,chl,1,1] input:[b,chl,h,w] output:[b,nv,h,w]

        self.get_weight()
    
    def get_weight(self):
        out = F.conv_transpose2d(self.eye,self.c2.weight,stride=self.c2.stride,padding=0,groups=self.c2.groups)
        out = F.conv_transpose2d(out,self.c1.weight,stride=self.c1.stride,padding=0,groups=self.c1.groups)
        self.weight = out

        return self.weight

class Lgn_ende_multi(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg = arg
        self.channel = arg.channel
        self.n_vector = arg.n_vector
        self.vector_length = arg.vector_length
        self.lgn_kernel_size = arg.lgn_kernel_size
        self.lgn_stride = arg.lgn_stride
        self.activate_fn = arg.activate_fn

        self.conv = eval(arg.gonlin)(arg)
        # self.conv1 = eval(arg.gonlin)(arg)
        # self.conv1 = self.conv
        print('weight_size',self.conv.weight.size())
        arg.lgn_padding = (int((self.conv.weight.size(2)-1)/2),int((self.conv.weight.size(2)-1)/2-self.lgn_stride+1))
        self.lgn_padding = arg.lgn_padding
        print('gangling cell padding:',self.lgn_padding)
        # self.bn = nn.BatchNorm2d(arg.n_vector*arg.vector_length,momentum=0.1,affine=False)
        self.bn = nn.Identity()

        if self.activate_fn is None:
            self.nonlinear = nn.Identity()
        elif self.activate_fn == 'gelu':
            self.nonlinear = nn.GELU()
        elif self.activate_fn == 'relu':
            self.nonlinear = nn.ReLU()
        
        self.apply(weights_init)

    def forward(self,x):
        w = self.conv.get_weight()
        return self.nonlinear(self.bn(F.conv2d(F.pad(x,[self.lgn_padding[0],self.lgn_padding[1],self.lgn_padding[0],self.lgn_padding[1]]),w,stride=self.lgn_stride,padding=0)))
    
    def deconv(self,f):
        w = self.conv.get_weight()
        return F.conv_transpose2d(f,w,stride=self.lgn_stride,padding=0)[:,:,self.lgn_padding[0]:-self.lgn_padding[1],self.lgn_padding[0]:-self.lgn_padding[1]]

    def deconv_group(self,f):
        w = self.conv.get_weight()
        img_group = F.conv_transpose2d(f,w,stride=self.lgn_stride,padding=0,groups=self.n_vector)[:,:,self.lgn_padding[0]:-self.lgn_padding[1],self.lgn_padding[0]:-self.lgn_padding[1]]
        img = reduce(img_group,'b (nv c) h w -> b c h w','sum',nv=self.n_vector)
        return img_group,img
    
    def fullconv(self,x):
        w = self.conv.get_weight()
        return self.nonlinear(F.conv2d(x,w,padding=self.lgn_padding[0]))
    
    def fulldeconv(self,f):
        w = self.conv.get_weight()
        return F.conv_transpose2d(f,w,padding=self.lgn_padding[0])/self.lgn_stride**2
    
    def fulldeconv_group(self,f):
        w = self.conv.get_weight()
        return F.conv_transpose2d(f,w,padding=self.lgn_padding[0],groups=self.n_vector)/self.lgn_stride**2
    

class VQLgn(Lgn_ende_multi):
    def __init__(self,arg):
        super().__init__(arg)
        # self.codebook = HRVQEmbedding([8,64,512], 128)
        self.K = 2048
        self.codebook = VQEmbedding(self.K,int(int(arg.n_vector*arg.vector_length)))
    
    def VQloss(self,z_e_x):
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.deconv(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

    def encode(self,x):
        w = self.conv.get_weight()
        z_e_x = self.nonlinear(self.bn(F.conv2d(F.pad(x,[self.lgn_padding[0],self.lgn_padding[1],self.lgn_padding[0],self.lgn_padding[1]]),w,stride=self.lgn_stride,padding=0)))    
        latents = self.codebook(z_e_x)
        return latents
    

class HRVQLgn(Lgn_ende_multi):
    def __init__(self,arg):
        super().__init__(arg)
        # self.K = [8,64,512]
        self.K = [4,16,64,256,1024]
        self.codebook = HRVQEmbedding(self.K,int(int(arg.n_vector*arg.vector_length)))
    
    def VQloss(self,z_e_x):
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = [self.deconv(z) for z in z_q_x_st]
        return x_tilde, z_e_x, z_q_x

    def encode(self,x):
        w = self.conv.get_weight()
        z_e_x = self.nonlinear(self.bn(F.conv2d(F.pad(x,[self.lgn_padding[0],self.lgn_padding[1],self.lgn_padding[0],self.lgn_padding[1]]),w,stride=self.lgn_stride,padding=0)))    
        latents = self.codebook(z_e_x)
        return latents
        

class Sep_Trans(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg = arg
        self.n_vector = arg.n_vector
        self.vector_length = arg.vector_length
        self.n_sep = arg.n_sep
        self.device = arg.device
        self.pred_ks = arg.pred_ks
        self.num_h = arg.num_h
        self.num_w = arg.num_w
        self.nonpara_init = arg.nonpara_init
        self.lgn_stride = arg.lgn_stride
        self.lgn_padding = arg.lgn_padding
        self.emb_scale = arg.emb_scale
        
        if self.arg.ifsep:
            self.coefx = nn.Conv2d(self.n_vector*self.vector_length**2,1,(1,self.n_sep),bias=False).to(self.device)
            self.coefy = nn.Conv2d(self.n_vector*self.vector_length**2,1,(self.n_sep,1),bias=False).to(self.device)
        else:
            self.coef = nn.Conv2d(self.n_vector*self.vector_length**2,1,(self.n_sep,self.n_sep),bias=False).to(self.device)
        self.unfold = nn.Unfold(kernel_size=self.pred_ks,padding=int((self.pred_ks-1)/2),stride=1)
        #张量展开
        self.init()

    def init(self):
        try:
            if self.arg.ifsep:
                nn.init.uniform_(self.coefx.weight,-self.nonpara_init,self.nonpara_init)
                nn.init.uniform_(self.coefy.weight,-self.nonpara_init,self.nonpara_init)
                print('Trans_Conv weights_inits',self.coefx)
            else:
                nn.init.uniform_(self.coef.weight,-self.nonpara_init,self.nonpara_init)
                print('Trans_Conv weights_inits',self.coef)
        except AttributeError:
            print("Skipping initialization of coef")

    def forward(self,grid1,grid2,theta,lgn):
        b,c,h,w = lgn.size()
        uf_lgn = rearrange(self.unfold(lgn),'b c (h w) -> b c h w',h=h)
        uf_grid1 = rearrange(self.unfold(rearrange(grid1,'b h w c -> b c h w')),'b c (h w) -> b c h w',h=h)
        # uf_grid1 = rearrange(grid1,'b h w c -> b c h w')
        uf_lgn_trans = F.grid_sample(uf_lgn,grid2,mode='nearest')
        uf_grid1_trans = F.grid_sample(uf_grid1,grid2,mode='nearest')
        grid1_ = rearrange(uf_grid1_trans,'b (c ks2) h w -> b c ks2 h w',c=2)
        grid2_ = rearrange(grid2,'b h w c -> b c 1 h w')
        dgrid = rearrange(grid2_-grid1_,'b c ks2 h w -> 1 b (ks2 h w) c')*self.emb_scale
        # [1 (nv vl vl) b (ks2 hw)] -> [b (ks2 h w) (nv vl vl)]
        coef_ = rearrange(F.grid_sample(self.get_coef(),dgrid,padding_mode='zeros'),'1 (nv vl1 vl2) b (ks2 hw) -> b hw ks2 nv vl1 vl2',hw=h*w,nv=self.n_vector,vl1=self.vector_length)   
        lgn_ = rearrange(uf_lgn_trans,'b (nv vl ks2) h w -> b (h w) ks2 nv vl 1',nv=self.n_vector,vl=self.vector_length)
        lgn_trans_pred_ = torch.matmul(coef_,lgn_).sum(dim=2)  # [b hw ks2 nv vl vl]x[b hw ks2 nv vl 1]=[b hw ks2 nv vl 1],[b hw ks2 nv vl 1] ->(sum) [b hw nv vl 1]
        lgn_trans_pred = rearrange(lgn_trans_pred_,'b (h w) nv vl 1 -> b (nv vl) h w',h=h)
        
        return lgn_trans_pred
    
    def get_coef(self):
        if self.arg.ifsep:
            x = rearrange(self.coefx.weight,'1 (nv vl1 vl2) h w -> nv vl1 vl2 h w',vl1=self.vector_length,vl2=self.vector_length)
            y = rearrange(self.coefy.weight,'1 (nv vl1 vl2) h w -> nv vl1 vl2 h w',vl1=self.vector_length,vl2=self.vector_length)
            w = torch.einsum('ijkmn,iklmn->ijlmn',x,y)
            return rearrange(w,'nv vl1 vl2 h w -> 1 (nv vl1 vl2) h w')
        else:
            return self.coef.weight
        # return self.coef.weight
        # return self.coef1.weight*self.coef2.weight

    def cal_coef_equ_loss(self,img,lgn_ende):
        if img.size(0)>5:
            img = img[0:5,:,:,:]
        s = img.size()
        img_channel = rearrange(img,'b c h w -> 1 (b c) h w').repeat(self.n_vector*self.vector_length,1,1,1)
        # img_channel = rearrange(img_repeat,'b c h w -> c b h w')
        grid1 = get_grid_xy_tensor(self.col_xy,img_channel.size()).detach()
        mask = reduce((grid1[:,:,:,0:1]>-0.9)&(grid1[:,:,:,0:1]<0.9)&(grid1[:,:,:,1:2]>-0.9)&(grid1[:,:,:,1:2]<0.9),'(nv vl) h w c -> nv () c (h w)','prod',nv=self.n_vector)
        img_trans = rearrange(F.grid_sample(img_channel,grid1,padding_mode='zeros'),'nvvl (b c) h w -> b (nvvl c) h w',c=s[1])
        lgn = F.conv2d(img_trans,lgn_ende.conv.weight,padding=lgn_ende.lgn_padding[0],groups=self.n_vector*self.vector_length)
        aln = rearrange(lgn,'b (nv vl) h w -> nv vl b (h w)',nv=self.n_vector)*mask
        # equ_loss = F.mse_loss(aln,aln.mean(dim=1,keepdim=True).repeat(1,aln.size(1),1,1))
        equ_loss = F.mse_loss(aln.unsqueeze(dim=1).repeat(1,aln.size(1),1,1,1),aln.unsqueeze(dim=2).repeat(1,1,aln.size(1),1,1))

        # pdb.set_trace()
        
        if equ_loss >= 10.:
            equ_loss = equ_loss * 0.1
        if equ_loss >= 1.:
            equ_loss = equ_loss * 0.1
        if not equ_loss < 1.:
            equ_loss = 0.

        return equ_loss
    
    def cal_sparse_equ_loss(self,weight=None):
        # x0, y0 = np.meshgrid(np.linspace(-0.33,0.33,37), np.linspace(-0.33,0.33,37))
        x0, y0 = np.meshgrid(np.linspace(-0.99*0.9,0.99*0.9,37), np.linspace(-0.99*0.9,0.99*0.9,37))
        # x0, y0 = np.meshgrid(np.linspace(-0.79,0.79,self.arg.n_sep*4), np.linspace(-0.79,0.79,self.arg.n_sep*4))
        # x0, y0 = np.meshgrid(np.linspace(-0.99,0.99,self.arg.n_sep), np.linspace(0.99,0.99,self.arg.n_sep))
        grid = torch.cat([torch.tensor(x0,dtype=torch.float).unsqueeze_(0).unsqueeze_(3),torch.tensor(y0,dtype=torch.float).unsqueeze_(0).unsqueeze_(3)],dim=3).to('cuda')
        coef = F.grid_sample(self.get_coef(),grid)
        coef = rearrange(coef,'1 (nv vl1 vl2) h w -> nv vl1 vl2 (h w)',nv=self.n_vector,vl1=self.vector_length)
        grid_ = rearrange(grid,'b h w c -> b 1 1 (h w) c')
        if self.arg.repel:
            attn = F.softmax(coef*10.,dim=1)
            attn = F.softmax(attn*10.,dim=-1)
        else:
            attn = F.softmax(coef*10.,dim=-1)
        attn_ = attn.unsqueeze(dim=4)
        
        # l = int(self.n_sep/3)-1
        # coef = rearrange(self.coef.weight[:,:,l:-l,l:-l],'1 (nv vl1 vl2) h w -> nv vl1 vl2 (h w)',nv=self.n_vector,vl1=self.vector_length)
        # attn = F.softmax(coef*10.,dim=-1) # [nv vl vl (hw)]
        
        # grid = get_grid_noaffine(self.coef.weight.size())[:,l:-l,l:-l,:] # [1 h w 2]
        # grid_ = rearrange(grid,'b h w c -> b 1 1 (h w) c') # [1 1 1 hw 2]
        # attn_ = attn.unsqueeze(dim=4)  # [nv vl vl hw 2]

        col_xy = (grid_*attn_).sum(dim=3).detach()  # [(nv vl) 2]
        i = np.random.randint(0,self.vector_length)
        self.col_xy = col_xy[:,:,i,:].view(-1,2) / self.emb_scale

        return

class Sep_Rot(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg = arg
        self.n_vector = arg.n_vector
        self.vector_length = arg.vector_length
        self.n_sep = arg.n_sep
        self.device = arg.device
        self.pred_ks = arg.pred_ks
        self.num_h = arg.num_h
        self.num_w = arg.num_w
        self.nonpara_init = arg.nonpara_init
        self.lgn_stride = arg.lgn_stride
        self.lgn_padding = arg.lgn_padding
        self.emb_scale = arg.emb_scale
        
        if self.arg.ifsep:
            self.coefx = nn.Conv3d(self.n_vector*self.vector_length**2,1,(1,self.n_sep,1),bias=False).to(self.device)
            self.coefy = nn.Conv3d(self.n_vector*self.vector_length**2,1,(self.n_sep,1,1),bias=False).to(self.device)
            self.coefr = nn.Conv3d(self.n_vector*self.vector_length**2,1,(1,1,self.n_sep),bias=False).to(self.device)
        else:
            self.coef = nn.Conv3d(self.n_vector*self.vector_length**2,1,(self.n_sep,self.n_sep,self.n_sep),bias=False).to(self.device)
        # self.coefx = nn.Conv3d(self.n_vector*self.vector_length**2,1,(1,self.n_sep,1),bias=False).to(self.device)
        # self.coefy = nn.Conv3d(self.n_vector*self.vector_length**2,1,(self.n_sep,1,1),bias=False).to(self.device)
        # self.coefr = nn.Conv3d(self.n_vector*self.vector_length**2,1,(1,1,self.n_sep),bias=False).to(self.device)
        # # self.coef = nn.Conv3d(self.n_vector*self.vector_length**2,1,(self.n_sep,self.n_sep,self.n_sep),bias=False).to(self.device)
        # self.unfold = nn.Unfold(kernel_size=1,padding=0,stride=1)
        self.unfold = nn.Unfold(kernel_size=self.pred_ks,padding=int((self.pred_ks-1)/2),stride=1)

        self.init()

    def init(self):
        try:
            # nn.init.uniform_(self.coef.weight,-self.nonpara_init,self.nonpara_init)
            # print('Rot_Conv weights_inits',self.coef)
            if self.arg.ifsep:
                nn.init.uniform_(self.coefx.weight,-self.nonpara_init,self.nonpara_init)
                nn.init.uniform_(self.coefy.weight,-self.nonpara_init,self.nonpara_init)
                nn.init.uniform_(self.coefr.weight,-self.nonpara_init,self.nonpara_init)
                print('Rot_Conv weights_inits',self.coefx)
            else:
                nn.init.uniform_(self.coef.weight,-self.nonpara_init,self.nonpara_init)
                print('Rot_Conv weights_inits',self.coef)
        except AttributeError:
            # print("Skipping initialization of ", self.coef)
            print("Skipping initialization of coef")

    def forward(self,grid1,grid2,theta,lgn):
        b,c,h,w = lgn.size()
        uf_lgn = rearrange(self.unfold(lgn),'b c (h w) -> b c h w',h=h)
        uf_grid1 = rearrange(self.unfold(rearrange(grid1,'b h w c -> b c h w')),'b c (h w) -> b c h w',h=h)
        # uf_grid1 = rearrange(grid1,'b h w c -> b c h w')
        uf_lgn_trans = F.grid_sample(uf_lgn,grid2,mode='nearest')
        uf_grid1_trans = F.grid_sample(uf_grid1,grid2,mode='nearest')
        grid1_ = rearrange(uf_grid1_trans,'b (c ks2) h w -> b c ks2 h w',c=2)
        grid2_ = rearrange(grid2,'b h w c -> b c 1 h w')
        dgrid = rearrange(grid2_-grid1_,'b c ks2 h w -> 1 b (ks2 h w) 1 c')*self.emb_scale
        s0,s1,s2,s3,s4 = dgrid.size()
        theta = theta.view(1,-1,1,1,1).repeat(s0,1,s2,s3,1)
        dgrid = self.rot_xy(dgrid,theta) # [1 b (ks2 h w) 1 c]
        coef_ = rearrange(F.grid_sample(self.get_coef(),dgrid,padding_mode='zeros'),'1 (nv vl1 vl2) b (ks2 hw) 1 -> b hw ks2 nv vl1 vl2',hw=h*w,nv=self.n_vector,vl1=self.vector_length)   
        lgn_ = rearrange(uf_lgn_trans,'b (nv vl ks2) h w -> b (h w) ks2 nv vl 1',nv=self.n_vector,vl=self.vector_length)
        lgn_trans_pred_ = torch.matmul(coef_,lgn_).sum(dim=2)  # [b hw ks2 nv vl vl]x[b hw ks2 nv vl 1]=[b hw ks2 nv vl 1],[b hw ks2 nv vl 1] ->(sum) [b hw nv vl 1]
        lgn_trans_pred = rearrange(lgn_trans_pred_,'b (h w) nv vl 1 -> b (nv vl) h w',h=h)

        return lgn_trans_pred

    def get_coef(self):
        if self.arg.ifsep:
            x = rearrange(self.coefx.weight,'1 (nv vl1 vl2) h w t -> nv vl1 vl2 h w t',vl1=self.vector_length,vl2=self.vector_length)
            y = rearrange(self.coefy.weight,'1 (nv vl1 vl2) h w t -> nv vl1 vl2 h w t',vl1=self.vector_length,vl2=self.vector_length)
            r = rearrange(self.coefr.weight,'1 (nv vl1 vl2) h w t -> nv vl1 vl2 h w t',vl1=self.vector_length,vl2=self.vector_length)
            w = torch.einsum('ijkmno,iklmno->ijlmno',x,y)
            w = torch.einsum('ijkmno,iklmno->ijlmno',w,r)
            return rearrange(w,'nv vl1 vl2 h w t -> 1 (nv vl1 vl2) h w t')
        else:
            return self.coef.weight
        # return self.coef.weight
        # return self.coef1.weight*self.coef2.weight
    
    def rot_xy(self,dgrid,theta):

        dgrid_ = torch.cat([dgrid,theta/np.pi],dim=4)

        return dgrid_   # [1 b (ks2 h w) 1 3]
    
    def cal_coef_equ_loss(self,img,lgn_ende):
        if img.size(0)>5:
            img = img[0:5,:,:,:]
        s = img.size()
        img_channel = rearrange(img,'b c h w -> 1 (b c) h w').repeat(self.n_vector*self.vector_length,1,1,1)
        # img_channel = F.pad(img_channel,(7,7,7,7))
        grid1 = get_grid_xyr_tensor(self.col_xy*0,self.col_theta,img_channel.size()).detach()
        img_trans = rearrange(F.grid_sample(img_channel,grid1),'nvvl (b c) h w -> b (nvvl c) h w',c=s[1])
        # lgn = F.conv2d(img_trans,weight,padding=self.lgn_stride,groups=self.n_vector*self.vector_length)
        lgn = F.conv2d(img_trans,lgn_ende.conv.weight,padding=lgn_ende.lgn_padding[0],groups=self.n_vector*self.vector_length)
        lgn_ = rearrange(lgn,'b c h w -> c b h w')
        grid2 = get_grid_xyr_tensor(self.col_xy,-self.col_theta,lgn_.size()).detach()
        mask = reduce((grid2[:,:,:,0:1]>-0.9)&(grid2[:,:,:,0:1]<0.9)&(grid2[:,:,:,1:2]>-0.9)&(grid2[:,:,:,1:2]<0.9),'(nv vl) h w c -> nv () c (h w)','prod',nv=self.n_vector)
        lgn_aln = F.grid_sample(lgn_,grid2)
        aln = rearrange(lgn_aln,'(nv vl) b h w -> nv vl b (h w)',nv=self.n_vector)*mask
        # equ_loss = F.mse_loss(aln,aln.mean(dim=1,keepdim=True).repeat(1,aln.size(1),1,1))
        equ_loss = F.mse_loss(aln.unsqueeze(dim=1).repeat(1,aln.size(1),1,1,1),aln.unsqueeze(dim=2).repeat(1,1,aln.size(1),1,1))

        # pdb.set_trace()

        if equ_loss >= 10.:
            equ_loss = equ_loss * 0.1
        if equ_loss >= 1.:
            equ_loss = equ_loss * 0.1
        if not equ_loss < 1.:
            equ_loss = 0.

        return equ_loss
    
    def cal_sparse_equ_loss(self,weight=None):
        # xyz = np.meshgrid(np.linspace(-0.33,0.33,37), np.linspace(-0.33,0.33,37), np.linspace(-1,1,37))
        xyz = np.meshgrid(np.linspace(-0.99*0.9,0.99*0.9,37), np.linspace(-0.99*0.9,0.99*0.9,37), np.linspace(-1,1,37))
        # xyz = np.meshgrid(np.linspace(-0.8,0.8,self.arg.n_sep*4), np.linspace(-0.8,0.8,self.arg.n_sep*4), np.linspace(-1.,1.,self.arg.n_sep*4))
        # xyz = np.meshgrid(np.linspace(0.99,0.99,self.arg.n_sep), np.linspace(0.99,0.99,self.arg.n_sep), np.linspace(-1.,1.,self.arg.n_sep))
        grid = torch.cat([torch.tensor(x0,dtype=torch.float).unsqueeze_(0).unsqueeze_(4) for x0 in xyz],dim=4).to('cuda')
        coef = F.grid_sample(self.get_coef(),grid)
        coef = rearrange(coef,'1 (nv vl1 vl2) h w t -> nv vl1 vl2 (h w t)',nv=self.n_vector,vl1=self.vector_length)
        grid_ = rearrange(grid,'b h w t c -> b 1 1 (h w t) c')
        if self.arg.repel:
            attn = F.softmax(coef*10.,dim=1)
            attn = F.softmax(attn*10.,dim=-1)
        else:
            attn = F.softmax(coef*10.,dim=-1)
        attn_ = attn.unsqueeze(dim=4)
        # coef = rearrange(self.coef.weight,'1 (nv vl1 vl2) h w t -> nv vl1 vl2 (h w t)',nv=self.n_vector,vl1=self.vector_length)  # [nv vl vl (hwt)]
        # attn = F.softmax(coef*10.,dim=-1) # [nv vl vl (hwt)]
        
        # grid = get_grid_noaffine(self.coef.weight.size()) # [1 h w t 3]
        # grid_ = rearrange(grid,'b h w t c -> b 1 1 (h w t) c') # [1 1 1 hwt 3]
        # attn_ = attn.unsqueeze(dim=4)  # [nv vl vl (hwt) 1]

        col_xyr = (grid_*attn_).sum(dim=3) # [1 1 1 hwt 3]x[nv vl vl hwt 1]=[nv vl vl hwt 3], [nv vl vl hwt 3] ->(sum) [nv vl vl 3]
        i = np.random.randint(0,self.vector_length)
        col_xy = col_xyr[:,:,i,:2].view(-1,2)
        self.col_xy = col_xy / self.emb_scale
        self.col_theta = col_xyr[:,:,i,2:].view(-1,1)*np.pi

        return

    
