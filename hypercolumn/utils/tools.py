
import argparse
import os
# import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torchvision.transforms as transforms
# import pytorch_lightning as pl

# from torch.utils.tensorboard import SummaryWriter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    print('opitm:',args.optim)
    if args.optim == 'OneCycleLR':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.1, cycle_momentum=False, anneal_strategy='cos')
    elif args.optim == 'MultiStepLR':
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay, momentum=0.9)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50000,70000],gamma=0.1)
    elif args.optim == 'CyclicLR':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=args.lr,
                                              mode = 'triangular2',cycle_momentum=False,
                                              step_size_up=4000, step_size_down=6000)

    return optimizer, scheduler


def fetch_optimizer_hrvq(args, model):
    """ Create the optimizer and learning rate scheduler """
    print('opitm:',args.optim)
    if args.optim == 'OneCycleLR':
        if args.HRVQ_cl == -1:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
        else:
            # import pdb; pdb.set_trace()
            para = []
            for m in model:
                for lgn in m.lgn_ende:
                    para += list(lgn.codebook.embedding[args.HRVQ_cl].parameters())
            optimizer = optim.AdamW(para, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='cos')
        
    return optimizer, scheduler

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(m,"bias"):
                if m.bias is not None:
                    m.bias.data.fill_(0)
            print('weights_inits',m)
        except AttributeError:
            print("Skipping initialization of ", classname,m)

    elif classname.find('Norm') != -1:
        try:
            if hasattr(m,"weight"):
                if m.weight is not None:
                    m.weight.data.fill_(1)
            if hasattr(m,"bias"):
                if m.bias is not None:
                    m.bias.data.fill_(0)
            print('weights_inits',m)
        except AttributeError:
            print("Skipping initialization of ", classname)

    elif classname.find('Linear') != -1:
        try:
            # nn.init.normal_(m.weight, std=1e-3)
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m,"bias"):
                if m.bias is not None:
                    m.bias.data.fill_(0)
            print('weights_inits',m)
        except AttributeError:
            print("Skipping initialization of ", classname)


class Logger:
    def __init__(self, model, scheduler,args):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.args = args

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/self.args.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[step:{:6d}, lr:{:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        # metrics_str = ("{:10.6f}, "*len(metrics_data)).format(*metrics_data)
        metrics_str = ''
        for k in sorted(self.running_loss.keys()):
            metrics_str += k
            metrics_str += ':'
            metrics_str += '{}, '.format(self.running_loss[k]/self.args.SUM_FREQ)
        
        # print the training status
        print(training_str + metrics_str)

        # if self.writer is None:
        #     self.writer = SummaryWriter()

        for k in self.running_loss:
        #     self.writer.add_scalar(k, self.running_loss[k]/self.args.SUM_FREQ, self.total_steps)
            self.running_loss[k] = self.running_loss[k]*0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = metrics[key]
            else:
                self.running_loss[key] += metrics[key]



        if self.total_steps % self.args.SUM_FREQ == self.args.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    # def write_dict(self, results):
    #     if self.writer is None:
    #         self.writer = SummaryWriter()

    #     for key in results:
    #         self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        # self.writer.close()
        return

def trans_img(x,theta=0.,scale=1.,tx=0.,ty=0.):
    grid = get_grid(x=x,theta=theta,scale=scale,tx=tx,ty=ty)
    # print(grid.size())
    return F.grid_sample(x, grid)

def get_grid(x,theta=0,scale=1.,tx=0.,ty=0.):
    s0 = x.size(0)
    angle = -theta*np.pi/180
    theta = torch.tensor([
        [np.cos(angle)*scale,np.sin(-angle)*scale,tx],
        [np.sin(angle)*scale,np.cos(angle)*scale ,ty]
    ], dtype=torch.float).to('cuda')
    if theta.size(0) == s0:
        theta = theta
    else:
        theta = theta.unsqueeze(0).repeat(s0,1,1)
    grid = F.affine_grid(theta, x.size())
    return grid

def get_grid_tensor(theta,scale,txy,s):
    theta_ = torch.cat([theta.cos()*scale,-theta.sin()*scale,theta.sin()*scale,theta.cos()*scale],dim=1).reshape(-1,2,2)
    txy_ = txy.reshape(-1,2,1)
    theta_aff = torch.cat([theta_,txy_],dim=2)
    return F.affine_grid(theta_aff,s)

def get_grid_xy_tensor(txy,s):
    theta_ = torch.eye(2).unsqueeze(0).repeat(txy.size(0),1,1).to(txy.device)
    txy_ = txy.reshape(-1,2,1)
    theta_aff = torch.cat([theta_,txy_],dim=2)  # [b 2 3]
    # print('tools, get_grid_xy_tensor, theta_aff:',theta_aff.size())
    return F.affine_grid(theta_aff,s)

def get_grid_xyr_tensor(txy,tr,s):
    theta_ = torch.cat([tr.cos(),-tr.sin(),tr.sin(),tr.cos()],dim=1).reshape(-1,2,2)
    txy_ = txy.reshape(-1,2,1)
    theta_aff = torch.cat([theta_,txy_],dim=2)
    return F.affine_grid(theta_aff,s)

def get_grid_rand_xyr(s,deform_range=[360,1,1]):
    
    s0 = s[0]
    angle = ((np.random.rand(s0,1)-0.5) * deform_range[0])/360*np.pi*2
    scale = np.exp(((np.random.rand(s0,1)*2-1)*np.log(deform_range[1])))
    txy = (np.random.rand(s0,2)-0.5)*2.*deform_range[2]

    txy = torch.tensor(txy,dtype=torch.float).to('cuda')
    angle = torch.tensor(angle,dtype=torch.float).to('cuda')
    scale = torch.tensor(scale,dtype=torch.float).to('cuda')

    return [txy,angle,scale]

def get_grid_rand(s,deform_range=[360.,0.5,0.3,0.3]):
    s0 = s[0]
    angle1 = ((np.random.rand(s0)-0.5) * deform_range[0])/360*np.pi*2#随机生成角度
    scale1 = np.exp(((np.random.rand(s0)*2-1)*np.log(deform_range[1])))#随机生成放缩
    tx1 = (np.random.rand(s0)-0.5)*2.*deform_range[2]#随机生成平移值
    ty1 = (np.random.rand(s0)-0.5)*2.*deform_range[3]

    I = torch.tensor(np.eye(3)[None,:,:], dtype=torch.float).to('cuda')
    # print(I.size())
    R1 = torch.tensor([
        [np.cos(angle1)*scale1,np.sin(-angle1)*scale1,np.zeros(tx1.shape)],
        [np.sin(angle1)*scale1,np.cos(angle1)*scale1 ,np.zeros(tx1.shape)],
        [np.zeros(tx1.shape),np.zeros(tx1.shape) ,np.ones(tx1.shape)]
    ], dtype=torch.float).to('cuda')
    # print(R1.size())
    R1 = rearrange(R1,'h w b -> b h w') #旋转矩阵
    # print(R1.size())
    R1_inv = torch.tensor([
        [np.cos(-angle1)/scale1,np.sin(angle1)/scale1,np.zeros(tx1.shape)],
        [np.sin(-angle1)/scale1,np.cos(-angle1)/scale1 ,np.zeros(tx1.shape)],
        [np.zeros(tx1.shape),np.zeros(tx1.shape) ,np.ones(tx1.shape)]
    ], dtype=torch.float).to('cuda')
    R1_inv = rearrange(R1_inv,'h w b -> b h w') #逆旋转矩阵
    T1 = torch.tensor([
        [np.zeros(tx1.shape),np.zeros(tx1.shape),tx1],
        [np.zeros(tx1.shape),np.zeros(tx1.shape),ty1],
        [np.zeros(tx1.shape),np.zeros(tx1.shape),np.zeros(tx1.shape)]
    ], dtype=torch.float).to('cuda')
    T1 = rearrange(T1,'h w b -> b h w') #平移矩阵
    
    theta1 = (R1+T1)[:,0:2,:]       #仿射变换参数  batch, 2, 3
    grid1 = F.affine_grid(theta1, s)#得到仿射变换的坐标网络 batch,h,w,2

    theta1_inv = torch.matmul(R1_inv,I-T1)
    theta1_inv = theta1_inv[:,0:2,:]
    grid1_inv = F.affine_grid(theta1_inv, s)

    # return grid1,grid1_inv,rearrange(torch.tensor([angle1,scale1,tx1,ty1],dtype=torch.float).to('cuda'),'c b -> b c 1 1')
    return grid1,grid1_inv,theta1,theta1_inv,torch.tensor(angle1,dtype=torch.float).to('cuda').view(-1,1,1,1)

def get_grid_noaffine(s):

    if len(s)==4:
        grid1,_,_,_,_ = get_grid_rand(s,deform_range=[0.,1.,0.,0.])
    elif len(s)==5:
        rot = torch.eye(3).view(1,3,3)
        tran = torch.zeros(1,3,1)
        theta = torch.cat([rot,tran],dim=2).repeat(s[0],1,1)
        grid1 = F.affine_grid(theta,s)

    return grid1.to('cuda')
