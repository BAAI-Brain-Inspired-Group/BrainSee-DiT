import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
import copy
from torchvision.utils import save_image

from einops import rearrange,reduce,repeat
from einops.layers.torch import Rearrange
from hypercolumn.utils.tools import *
import numpy as np

# from vit_pytorch.V1 import *
# from vit_pytorch.V1_sep import Lgn_ende_multi as Lgn_ende
from hypercolumn.vit_pytorch.V1_sep import HRVQLgn as Lgn_ende
from hypercolumn.vit_pytorch.V1_sep import Sep_Rot,Sep_Trans
import pdb
    
class Column_trans_rot(nn.Module):
    def __init__(
        self,
        arg,
        masking_ratio = 0.75,
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        self.arg = arg
        arg_trans = self.get_trans_arg()
        arg_rot = self.get_rot_arg()

        self.lgn_ende = nn.ModuleList([Lgn_ende(arg).to(arg.device) for arg in arg_rot])

        self.apply(weights_init)
        self.trans_conv = nn.ModuleList([Sep_Trans(arg) for arg in arg_trans])
        self.rot_conv = nn.ModuleList([Sep_Rot(arg) for arg in arg_rot])
        print('len(self.lgn_ende):',len(self.lgn_ende))

    def forward(self, img):
        loss_rot,res_rot = self.get_loss(img,self.rot_conv,[self.arg.max_rot,1.,self.arg.max_trans,self.arg.max_trans],'rot')
        loss_trans,res_trans = self.get_loss(img,self.trans_conv,[0.,1.,self.arg.max_trans,self.arg.max_trans],'trans')

        loss = loss_rot + loss_trans
        res_rot[0].update(res_trans[0])
        metric = res_rot[0]
        metric['0:loss'] = loss.item()
        res = [metric] + res_rot[1:] + res_trans[1:]

        return loss,res

    def get_loss(self,img,trans_conv,deform_range,name):
        # data process
        stride = self.arg.lgn_stride
        device = img.device
        s = img.size()
        grid = get_grid_noaffine(s).to(device)
        grid1,grid1_inv,theta1,_,angle = get_grid_rand(s,deform_range=deform_range)   #grid与grid1区别
        # count = torch.sum((grid1[:,:,:,0:1]>-0.9)&(grid1[:,:,:,0:1]<0.9)&(grid1[:,:,:,1:2]>-0.9)&(grid1[:,:,:,1:2]<0.9))
        # print('count',count)
        mask = rearrange((grid1[:,:,:,0:1]>-0.9)&(grid1[:,:,:,0:1]<0.9)&(grid1[:,:,:,1:2]>-0.9)&(grid1[:,:,:,1:2]<0.9),'b h w c -> b c h w')#作用？
        # print('get_loss:',img.device,grid1.device)
        img_trans = F.grid_sample(img,grid1)
        # img_trans_grid = F.grid_sample(img,grid)
        #可视化img_trans
        # get features
        lgn,lgn_trans = [lgn_ende(img) for lgn_ende in self.lgn_ende],[lgn_ende(img_trans) for lgn_ende in self.lgn_ende]
        lgn_trans_pred = [trans_conv[i](grid[:,::stride,::stride,:],grid1[:,::stride,::stride,:],angle,lgn[i]) for i in range(len(lgn))]

        # calculate loss
        [conv.cal_sparse_equ_loss() for conv in trans_conv]
        coef_equ_loss = torch.tensor(np.array([0.])).to(device)
        for i in range(len(self.lgn_ende)):
            coef_equ_loss = coef_equ_loss + trans_conv[i].cal_coef_equ_loss(img,self.lgn_ende[i])
        
        equ_loss = torch.tensor(np.array([0.])).to(device)
        for i in range(len(lgn_trans)):
            equ_loss = equ_loss + F.mse_loss(lgn_trans_pred[i]*mask[:,:,::stride,::stride],lgn_trans[i]*mask[:,:,::stride,::stride])

        recon_loss,recon_img = self.get_recon_loss(img,img_trans,lgn,lgn_trans,lgn_trans_pred,mask)

        loss = self.arg.lambda_equ*equ_loss + self.arg.lambda_coef_equ_loss*coef_equ_loss + self.arg.lambda_recon*recon_loss[0] + self.arg.lambda_recon_trans*recon_loss[1] + self.arg.lambda_recon_trans_pred*recon_loss[2] + self.arg.lambda_recon*recon_loss[3]

        metric = {'1:'+name+'_equ':np.array([equ_loss.item(),coef_equ_loss.item()]),'2:'+name+'_recon':np.array([l.item() for l in recon_loss])}
        
        return loss,[metric,recon_img[0],img_trans,recon_img[1],recon_img[2]]
    
    def get_recon_loss(self,img,img_trans,lgn,lgn_trans,lgn_pred,mask):
        img_recon,img_recon_trans,img_recon_trans_pred = 0.,0.,0.
        
        for i in range(len(lgn)):
            img_rc = self.lgn_ende[i].deconv(lgn[i])
            img_rc_trans = self.lgn_ende[i].deconv(lgn_trans[i])
            img_rc_trans_pred = self.lgn_ende[i].deconv(lgn_pred[i])
            img_recon = img_recon + img_rc
            img_recon_trans = img_recon_trans + img_rc_trans
            img_recon_trans_pred = img_recon_trans_pred + img_rc_trans_pred

        recon_loss = F.mse_loss(img,img_recon) #+ F.mse_loss(img,img_rc)*0.5
        recon_loss_trans = F.mse_loss(img_trans*mask,img_recon_trans*mask) #+ F.mse_loss(img_trans*mask,img_rc_trans*mask)*0.5
        recon_loss_trans_pred = F.mse_loss(img_trans*mask,img_recon_trans_pred*mask) #+ F.mse_loss(img_trans*mask,img_rc_trans_pred*mask)*0.5
        return [recon_loss,recon_loss_trans,recon_loss_trans_pred],[img_recon,img_recon_trans,img_recon_trans_pred]
    
    
    def get_recon_loss_sep(self,img,img_trans,lgn,lgn_trans,lgn_pred,mask):
        img_recon,img_recon_trans,img_recon_trans_pred = [],[],[]
        recon_loss,recon_loss_trans,recon_loss_trans_pred = 0.,0.,0.
        
        for i in range(len(lgn)):
            img_rc = self.lgn_ende[i].deconv(lgn[i])
            img_rc_trans = self.lgn_ende[i].deconv(lgn_trans[i])
            img_rc_trans_pred = self.lgn_ende[i].deconv(lgn_pred[i])
            img_recon = img_recon.append(img_rc)
            img_recon_trans = img_recon_trans.append(img_rc_trans)
            img_recon_trans_pred = img_recon_trans_pred.append(img_rc_trans_pred)
            recon_loss = recon_loss + F.mse_loss(img,img_rc)
            recon_loss_trans = recon_loss_trans + F.mse_loss(img,img_rc_trans*mask)
            recon_loss_trans_pred = recon_loss_trans_pred + F.mse_loss(img,img_rc_trans_pred*mask)


        # recon_loss = F.mse_loss(img,img_recon) #+ F.mse_loss(img,img_rc)*0.5
        # recon_loss_trans = F.mse_loss(img_trans*mask,img_recon_trans*mask) #+ F.mse_loss(img_trans*mask,img_rc_trans*mask)*0.5
        # recon_loss_trans_pred = F.mse_loss(img_trans*mask,img_recon_trans_pred*mask) #+ F.mse_loss(img_trans*mask,img_rc_trans_pred*mask)*0.5
        return [recon_loss,recon_loss_trans,recon_loss_trans_pred],[img_recon,img_recon_trans,img_recon_trans_pred]
    
    def get_rot_arg(self):
        arg = []
        # for nv,vl,rn in self.arg.n_vector,self.arg.vector_length,self.arg.rot_num:
        for i,nv in enumerate(self.arg.n_vector):
            nv,vl,rn = self.arg.n_vector[i],self.arg.vector_length[i],self.arg.rot_num[i]
            #16 2 4; 64 1 8
            arg_rot = copy.deepcopy(self.arg)
            arg_rot.n_vector = int(nv/rn)    #4
            arg_rot.vector_length = int(vl*rn)     #8
            arg.append(arg_rot)

        return arg

    def get_trans_arg(self):
        arg = []
        # for nv,vl,rn in self.arg.n_vector,self.arg.vector_length,self.arg.rot_num:
        for i,nv in enumerate(self.arg.n_vector):
            nv,vl,rn = self.arg.n_vector[i],self.arg.vector_length[i],self.arg.rot_num[i]
            arg_trans = copy.deepcopy(self.arg)
            arg_trans.n_vector = nv
            arg_trans.vector_length = vl
            arg.append(arg_trans)
        
        return arg
    
    def logimg(self,img,file,denorm=None):
        # 判断file是否存在，不存在则创建
        if not os.path.exists(file):
            os.makedirs(file)

        if not denorm:
            # norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
            # norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
            norm_mean = np.array([0.5, 0.5, 0.5])
            norm_std = np.array([0.5, 0.5, 0.5])
            denorm = nn.Sequential(transforms.Normalize(norm_mean*0., 1./norm_std),transforms.Normalize(-norm_mean, [1.,1.,1.]))

        if img.size(0) >32:
            img = img[0:32]

        # pdb.set_trace()
        # 保存输入图像
        save_image(
        denorm(img),
        f'{file}/input_img.png',
        nrow=8
        )
        
        # 计算每组超柱的重建图并保存

        ft_map = [lgn_ende.codebook.straight_through(lgn_ende(img))[0] for lgn_ende in self.lgn_ende]

        recon_group = [self.lgn_ende[i].deconv_group(ft_map[i])[0] for i in range(len(self.lgn_ende))]
        recon_group = [rearrange(rg,'b (c c1) h w -> b c c1 h w',c1=self.arg.channel) for rg in recon_group]
        # recon_group = [rg/reduce(rg.abs(),'b c c1 h w -> b c () () ()','max') for rg in recon_group]
        recon_group = [(rg-reduce(rg,'b c c1 h w -> b c () () ()','min'))/(reduce(rg,'b c c1 h w -> b c () () ()','max')-reduce(rg,'b c c1 h w -> b c () () ()','min')) for rg in recon_group]
        # recon_group = [rearrange(rg,'b c c1 h w -> b (c c1) h w') for rg in recon_group]
        recon_group_save = [rearrange(rg[0:1],'b c c1 h w -> (b c) c1 h w') for rg in recon_group]

        for i,rg in enumerate(recon_group_save):
            save_image(
                rg,
                f'{file}/recon_group_{i}.png',
                nrow=4
            )

        recon = [self.lgn_ende[i].deconv(ft_map[i]) for i in range(len(self.lgn_ende))]
        recon = [denorm(rg) for rg in recon]
        for i,rg in enumerate(recon):
            save_image(
                rg,
                f'{file}/recon_{i}.png',
                nrow=8
            )

        # 保存每组超柱的重建图
        w1 = [lgn_ende.conv.weight for lgn_ende in self.lgn_ende]
        w_norm = [(w-w.min())/(w.max()-w.min()+1e-8) for w in w1]
        
        # pad = nn.Sequential(nn.ConstantPad2d((0,1,0,0),0.),nn.ConstantPad2d((0,0,0,1),0.))
        for i,w in enumerate(w_norm):
            save_image(
                w,
                f'{file}/w_{i}.png',
                nrow=self.lgn_ende[i].vector_length
            )

        return [recon_group_save,recon,w_norm]
    
class VQColumn_trans_rot(Column_trans_rot):
    def __init__(self, arg):
        super().__init__(arg)
        self.arg.commit_loss = 0.1

    def get_recon_loss(self,img,img_trans,lgn,lgn_trans,lgn_pred,mask):
        img_recon,img_recon_trans,img_recon_trans_pred = 0.,0.,0.
        vq_loss = 0.
        
        for i in range(len(lgn)):
            img_rc,z_e_x,z_q_x = self.lgn_ende[i].VQloss(lgn[i])
            vq_loss += F.mse_loss(z_q_x, z_e_x.detach()) + self.arg.commit_loss*F.mse_loss(z_e_x, z_q_x.detach())
            img_rc_trans,z_e_x,z_q_x = self.lgn_ende[i].VQloss(lgn_trans[i])
            vq_loss += F.mse_loss(z_q_x, z_e_x.detach()) + self.arg.commit_loss*F.mse_loss(z_e_x, z_q_x.detach())
            img_rc_trans_pred,z_e_x,z_q_x = self.lgn_ende[i].VQloss(lgn_pred[i])
            vq_loss += F.mse_loss(z_q_x, z_e_x.detach()) + self.arg.commit_loss*F.mse_loss(z_e_x, z_q_x.detach())
            img_recon = img_recon + img_rc
            img_recon_trans = img_recon_trans + img_rc_trans
            img_recon_trans_pred = img_recon_trans_pred + img_rc_trans_pred

        recon_loss = F.mse_loss(img,img_recon) #+ F.mse_loss(img,img_rc)*0.5
        recon_loss_trans = F.mse_loss(img_trans*mask,img_recon_trans*mask) #+ F.mse_loss(img_trans*mask,img_rc_trans*mask)*0.5
        recon_loss_trans_pred = F.mse_loss(img_trans*mask,img_recon_trans_pred*mask) #+ F.mse_loss(img_trans*mask,img_rc_trans_pred*mask)*0.5
        return [recon_loss,recon_loss_trans,recon_loss_trans_pred,vq_loss],[img_recon,img_recon_trans,img_recon_trans_pred]
    
class HRVQColumn_trans_rot(Column_trans_rot):
    def __init__(self, arg):
        super().__init__(arg)
        self.arg.commit_loss = 0.1
        self.HR_decay = 0.7
        self.rl_coef = 0.5

    # def get_recon_loss(self,img,img_trans,lgn,lgn_trans,lgn_pred,mask):
    #     cl = self.arg.HRVQ_cl
    #     if cl == -1:
    #         [recon_loss,recon_loss_trans,recon_loss_trans_pred],[img_recon,img_recon_trans,img_recon_trans_pred] = super().get_recon_loss(img,img_trans,lgn,lgn_trans,lgn_pred,mask)
    #         return [recon_loss,recon_loss_trans,recon_loss_trans_pred,0.*recon_loss.detach()],[img_recon,img_recon_trans,img_recon_trans_pred]
    #     else:
    #         img_recon,img_recon_trans,img_recon_trans_pred = 0.,0.,0.
    #         vq_loss = 0.
            
    #         for i in range(len(lgn)):
    #             img_rc,z_e_x,z_q_x = self.lgn_ende[i].VQloss(lgn[i])
    #             img_rc,z_q_x = img_rc[cl],z_q_x[cl]
    #             vq_loss = vq_loss + F.mse_loss(z_q_x, z_e_x.detach())
    #             img_rc_trans,z_e_x,z_q_x = self.lgn_ende[i].VQloss(lgn_trans[i])
    #             img_rc_trans,z_q_x = img_rc_trans[cl],z_q_x[cl]
    #             vq_loss = vq_loss + F.mse_loss(z_q_x, z_e_x.detach())
    #             img_rc_trans_pred,z_e_x,z_q_x = self.lgn_ende[i].VQloss(lgn_pred[i])
    #             img_rc_trans_pred,z_q_x = img_rc_trans_pred[cl],z_q_x[cl]
    #             vq_loss = vq_loss + F.mse_loss(z_q_x, z_e_x.detach())
    #             img_recon = img_recon + img_rc
    #             img_recon_trans = img_recon_trans + img_rc_trans
    #             img_recon_trans_pred = img_recon_trans_pred + img_rc_trans_pred

    #         recon_loss = F.mse_loss(img,img_recon) #+ F.mse_loss(img,img_rc)*0.5
    #         recon_loss_trans = F.mse_loss(img_trans*mask,img_recon_trans*mask) #+ F.mse_loss(img_trans*mask,img_rc_trans*mask)*0.5
    #         recon_loss_trans_pred = F.mse_loss(img_trans*mask,img_recon_trans_pred*mask) #+ F.mse_loss(img_trans*mask,img_rc_trans_pred*mask)*0.5
    #         return [recon_loss,recon_loss_trans,recon_loss_trans_pred,vq_loss],[img_recon,img_recon_trans,img_recon_trans_pred]
        
    def get_recon_loss(self,img,img_trans,lgn,lgn_trans,lgn_pred,mask):
        K = len(self.lgn_ende[0].K)
        hd = self.HR_decay
        img_recon,img_recon_trans,img_recon_trans_pred = [0. for i in range(K)],[0. for i in range(K)],[0. for i in range(K)]
        vq_loss = 0.
        
        for i in range(len(lgn)):
            img_rc,z_e_x,z_q_x = self.lgn_ende[i].VQloss(lgn[i])
            for k,q in enumerate(z_q_x):
                vq_loss += F.mse_loss(q, z_e_x.detach())*(hd**(K-1-k)) + self.arg.commit_loss*F.mse_loss(z_e_x, q.detach())*(hd**(K-1-k))
            img_rc_trans,z_e_x,z_q_x = self.lgn_ende[i].VQloss(lgn_trans[i])
            for k,q in enumerate(z_q_x):
                vq_loss += F.mse_loss(q, z_e_x.detach())*(hd**(K-1-k)) + self.arg.commit_loss*F.mse_loss(z_e_x, q.detach())*(hd**(K-1-k))
            img_rc_trans_pred,z_e_x,z_q_x = self.lgn_ende[i].VQloss(lgn_pred[i])
            for k,q in enumerate(z_q_x):
                vq_loss += F.mse_loss(q, z_e_x.detach())*(hd**(K-1-k)) + self.arg.commit_loss*F.mse_loss(z_e_x, q.detach())*(hd**(K-1-k))
            for j,_ in enumerate(img_rc):
                img_recon[j] += img_rc[j]
                img_recon_trans[j] += img_rc_trans[j]
                img_recon_trans_pred[j] += img_rc_trans_pred[j]
        [rl,rlt,rltp],_ = super().get_recon_loss(img,img_trans,lgn,lgn_trans,lgn_pred,mask)

        recon_loss = torch.sum(torch.stack([F.mse_loss(img,rc)*(hd**(K-1-i)) for i,rc in enumerate(img_recon)]))*self.rl_coef + rl
        recon_loss_trans = torch.sum(torch.stack([F.mse_loss(img_trans*mask,rc)*(hd**(K-1-i)) for i,rc in enumerate(img_recon_trans)]))*self.rl_coef + rlt
        recon_loss_trans_pred = torch.sum(torch.stack([F.mse_loss(img_trans*mask,rc)*(hd**(K-1-i)) for i,rc in enumerate(img_recon_trans_pred)]))*self.rl_coef + rltp
        return [recon_loss,recon_loss_trans,recon_loss_trans_pred,vq_loss*self.rl_coef],[img_recon,img_recon_trans,img_recon_trans_pred]

    def logimg(self,img,file,denorm=None):
        # 判断file是否存在，不存在则创建
        if not os.path.exists(file):
            os.makedirs(file)

        if not denorm:
            # norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
            # norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
            norm_mean = np.array([0.5, 0.5, 0.5])
            norm_std = np.array([0.5, 0.5, 0.5])
            denorm = nn.Sequential(transforms.Normalize(norm_mean*0., 1./norm_std),transforms.Normalize(-norm_mean, [1.,1.,1.]))

        if img.size(0) >32:
            img = img[0:32]

        # pdb.set_trace()
        # 保存输入图像
        save_image(
        denorm(img),
        f'{file}/{self.arg.HRVQ_cl}_input_img.png',
        nrow=8
        )
        
        # 计算每组超柱的重建图并保存

        if self.arg.HRVQ_cl == -1:
            ft_map = [lgn_ende(img) for lgn_ende in self.lgn_ende]
        else:
            ft_map = [lgn_ende.codebook.straight_through(lgn_ende(img))[0][self.arg.HRVQ_cl] for lgn_ende in self.lgn_ende]

        recon_group = [self.lgn_ende[i].deconv_group(ft_map[i])[0] for i in range(len(self.lgn_ende))]
        recon_group = [rearrange(rg,'b (c c1) h w -> b c c1 h w',c1=self.arg.channel) for rg in recon_group]
        # recon_group = [rg/reduce(rg.abs(),'b c c1 h w -> b c () () ()','max') for rg in recon_group]
        recon_group = [(rg-reduce(rg,'b c c1 h w -> b c () () ()','min'))/(reduce(rg,'b c c1 h w -> b c () () ()','max')-reduce(rg,'b c c1 h w -> b c () () ()','min')) for rg in recon_group]
        # recon_group = [rearrange(rg,'b c c1 h w -> b (c c1) h w') for rg in recon_group]
        recon_group_save = [rearrange(rg[0:1],'b c c1 h w -> (b c) c1 h w') for rg in recon_group]

        for i,rg in enumerate(recon_group_save):
            save_image(
                rg,
                f'{file}/{self.arg.HRVQ_cl}_recon_group_{i}.png',
                nrow=4
            )

        recon = [self.lgn_ende[i].deconv(ft_map[i]) for i in range(len(self.lgn_ende))]
        recon = [denorm(rg) for rg in recon]
        for i,rg in enumerate(recon):
            save_image(
                rg,
                f'{file}/{self.arg.HRVQ_cl}_recon_{i}.png',
                nrow=8
            )

        # 保存每组超柱的重建图
        w1 = [lgn_ende.conv.weight for lgn_ende in self.lgn_ende]
        w_norm = [(w-w.min())/(w.max()-w.min()+1e-8) for w in w1]
        
        # pad = nn.Sequential(nn.ConstantPad2d((0,1,0,0),0.),nn.ConstantPad2d((0,0,0,1),0.))
        for i,w in enumerate(w_norm):
            save_image(
                w,
                f'{file}/{self.arg.HRVQ_cl}_w_{i}.png',
                nrow=self.lgn_ende[i].vector_length
            )
        # import pdb; pdb.set_trace()

        return [recon_group_save,recon,w_norm]

    
class Column_GP(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.GP = GaussianPyramid()
        self.arg = arg
        self.n_level = 5
    
        _columns = []
        for i in range(self.n_level):
            _args = copy.deepcopy(arg)
            _args.emb_scale = _args.emb_scale / (2**i)
            _columns.append(_args)
            
        self.columns = nn.ModuleList([HRVQColumn_trans_rot(_columns[i]) for i in range(self.n_level)])

        ## share lgn_ende for all columns
        for i in range(self.n_level):
            if i != 0:
                self.columns[i].lgn_ende = self.columns[0].lgn_ende

    def share_column(self):
        for i in range(self.n_level):
            if i!= 0:
                self.columns[i].lgn_ende = self.columns[0].lgn_ende

    def forward(self,img):
        img_GP,img_DoG = self.get_img_DoG(img)
        losses = []
        reses = []
        for i in range(self.n_level-1):

            # pdb.set_trace()
            loss,res = self.columns[i](img_DoG[i])
            losses.append(loss)
            reses.append(res)
        loss,res = self.columns[-1](img_GP[-1])
        losses.append(loss)
        reses.append(res)
            
        return losses,reses

    def get_img_DoG(self,img):
        img_GP = [img]
        for i in range(self.n_level-1):
            img_GP.append(self.GP.downPy(img_GP[-1]))

        img_DoG = []
        for i in range(self.n_level-1):
            img_DoG.append(img_GP[i]-self.GP.upPy(img_GP[i+1]))

        return img_GP,img_DoG
    
    def logimg(self,img,file,denorm=None):
        if not denorm:
            # norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
            # norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
            norm_mean = np.array([0.5, 0.5, 0.5])
            norm_std = np.array([0.5, 0.5, 0.5])
            denorm = nn.Sequential(transforms.Normalize(norm_mean*0., 1./norm_std),transforms.Normalize(-norm_mean, [1.,1.,1.]))

        if not os.path.exists(file):
            os.makedirs(file)

        if img.size(0) > 32:
            img = img[0:32]

        # log Gaussian pyramid images and DoG images
        img_GP,img_DoG = self.get_img_DoG(img)
        for i in range(len(img_GP)):
            save_image(denorm(img_GP[i]),f'{file}/img_GP_{i}.png',nrow=8)

        for i in range(len(img_DoG)):
            save_image(denorm(img_DoG[i]),f'{file}/img_DoG_{i}.png',nrow=8)

        # log each level's column reconstruction, for the specific HRVQ_cl  
        for i in range(self.n_level):
            j = self.n_level-1-i
            self.columns[j].arg.HRVQ_cl = self.arg.HRVQ_cl
            if i == 0:
                self.columns[j].logimg(img_GP[j],file=f'{file}/{i}')
            else:
                self.columns[j].logimg(img_DoG[j],file=f'{file}/{i}')

        cl = self.arg.HRVQ_cl
        if cl == -1:
            Recon = lambda i,img:[self.columns[i].lgn_ende[j].deconv(f) for j,f in enumerate([lgn_ende(img) for lgn_ende in self.columns[i].lgn_ende])][0]
        else:
            Recon = lambda i,img:[self.columns[i].lgn_ende[j].deconv(f) for j,f in enumerate([lgn_ende.codebook.straight_through(lgn_ende(img))[0][cl] for lgn_ende in self.columns[i].lgn_ende])][0]
        
        img0 = Recon(-1,img_GP[-1])
        save_image(denorm(img0),f'{file}/recon0.png',nrow=8)
        for i in range(self.n_level-1):
            j = self.n_level-2-i
            img0 = self.GP.upPy(img0)
            res = img_GP[j] - img0
            save_image(denorm(res),f'{file}/res{i+1}.png',nrow=8)
            img0 += Recon(j,res)
            save_image(denorm(img0),f'{file}/recon{i+1}.png',nrow=8)

        return



class GaussianPyramid(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.414, channels=3):
        super().__init__()
        self.channels = channels
        self.ks = kernel_size
        self.sigma = sigma
        self.pad = (self.ks-1) // 2
        
        self.kernel = self._gauss_kernel()

    def __call__(self, x):
        return self.blur(x)
    
    def _gauss_kernel(self):
        x_data, y_data = np.mgrid[-(self.ks-1) // 2 :(self.ks-1) // 2+1, -(self.ks-1) // 2:(self.ks-1) // 2+1]
        x_ = torch.FloatTensor(x_data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        y_ = torch.FloatTensor(y_data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        g = torch.exp(-((x_ ** 2 + y_ ** 2) / torch.tensor(2.0 * self.sigma ** 2)))
        g = g / torch.sum(g)
        kernel = repeat(torch.FloatTensor(g), '1 1 h w -> c 1 h w', c=self.channels)
        # return nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def downPy(self,x):
        self.kernel = self.kernel.to(x.device)
        # import pdb; pdb.set_trace()
        x = F.conv2d(x, self.kernel, stride=1,  padding=self.pad, groups=self.channels)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return x
    
    def upPy(self,x):
        self.kernel = self.kernel.to(x.device)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.conv2d(x, self.kernel, stride=1,  padding=self.pad, groups=self.channels)
        # x = F.conv_transpose2d(x, self.kernel, stride=2, groups=self.channels)[:,:,self.pad:(-self.pad+1),self.pad:(-self.pad+1)]
        return x
    
    def DoG(self,x):
        return x - self.upPy(self.downPy(x))

    def blur(self,x):
        self.kernel.to(x.device)
        x = F.conv2d(x, self.kernel, stride=1,  padding=self.pad, groups=self.channels)
        return x
    
    def logimg(self,img,file='./logger/imagenet/test_2_20_gauss',denorm=None):
        # 判断file是否存在，不存在则创建
        if not os.path.exists(file):
            os.makedirs(file)

        if not denorm:
            # norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
            # norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
            norm_mean = np.array([0.5, 0.5, 0.5])
            norm_std = np.array([0.5, 0.5, 0.5])
            denorm = nn.Sequential(transforms.Normalize(norm_mean*0., 1./norm_std),transforms.Normalize(-norm_mean, [1.,1.,1.]))

        if img.size(0) >32:
            img = img[0:32]

        # pdb.set_trace()
        # 保存输入图像
        save_image(
        denorm(img),
        f'{file}/input_img.png',
        nrow=8
        )
        
        img_recon = self.upPy(self.downPy(img))
        save_image(
            denorm(img_recon),
            f'{file}/downPy_upPy.png',
            nrow = 8
        )

        DoG = img - img_recon
        save_image(
            denorm(DoG),
            f'{file}/DoG.png',
            nrow = 8
        )

        return [img_recon,DoG]


# class Column_trans_rot_V2(Column_trans_rot):
#     def __init__(
#         self,
#         arg,
#         masking_ratio = 0.75,
#     )-> None:
#         super().__init__(arg,masking_ratio)
#         self.GaussPyrimid = 

#     def get_recon_loss(self,img,img_trans,lgn,lgn_trans,lgn_pred,recon_last=None,mask=None):
#         if recon_last is None:
#             img_recon,img_recon_trans,img_recon_trans_pred = 0.,0.,0.
#         else:
#             img_recon,img_recon_trans,img_recon_trans_pred = recon_last[0].detach(),recon_last[1].detach(),recon_last[2].detach()
        
#         for i in range(len(lgn)):
#             img_rc = self.lgn_ende[i].deconv(lgn[i])
#             img_rc_trans = self.lgn_ende[i].deconv(lgn_trans[i])
#             img_rc_trans_pred = self.lgn_ende[i].deconv(lgn_pred[i])
#             img_recon = img_recon + img_rc
#             img_recon_trans = img_recon_trans + img_rc_trans
#             img_recon_trans_pred = img_recon_trans_pred + img_rc_trans_pred

#         recon_loss = F.mse_loss(img,img_recon) #+ F.mse_loss(img,img_rc)*0.5
#         recon_loss_trans = F.mse_loss(img_trans*mask,img_recon_trans*mask) #+ F.mse_loss(img_trans*mask,img_rc_trans*mask)*0.5
#         recon_loss_trans_pred = F.mse_loss(img_trans*mask,img_recon_trans_pred*mask) #+ F.mse_loss(img_trans*mask,img_rc_trans_pred*mask)*0.5
#         return [recon_loss,recon_loss_trans,recon_loss_trans_pred],[img_recon,img_recon_trans,img_recon_trans_pred]

    # def get_rot_arg(self):
    #     arg = []

    #     arg_rot = copy.deepcopy(self.arg)
    #     arg_rot.n_vector = int(self.arg.n_vector/self.arg.rot_num)
    #     arg_rot.vector_length = int(self.arg.vector_length*self.arg.rot_num)
    #     arg.append(arg_rot)

    #     arg_ivrt = copy.deepcopy(self.arg)
    #     arg_ivrt.n_vector = arg_ivrt.n_vector_invariant
    #     arg_ivrt.vector_length = arg_ivrt.vector_length_invariant
    #     arg.append(arg_ivrt)
    #     return arg

    # def get_trans_arg(self):
    #     arg =[]
        
    #     arg.append(copy.deepcopy(self.arg))

    #     arg_ivrt = copy.deepcopy(self.arg)
    #     arg_ivrt.n_vector = arg_ivrt.n_vector_invariant
    #     arg_ivrt.vector_length = arg_ivrt.vector_length_invariant
    #     arg.append(arg_ivrt)
    #     return arg
    
    
# class Column_trans_rot_lgn(nn.Module):
#     def __init__(
#         self,
#         arg,
#         masking_ratio = 0.75,
#     ):
#         super().__init__()
#         assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
#         self.masking_ratio = masking_ratio

#         self.arg = arg
#         arg_ivrt = self.get_invariant_arg()

#         self.lgn_ende = Lgn_ende(arg).to(arg.device)
#         self.lgn_ende_ivrt = Lgn_ende(arg_ivrt).to(arg_ivrt.device)

#         self.apply(weights_init)

#     def forward(self, img):
#         return self.lgn_ende.conv_stride(img)

#     def get_invariant_arg(self):
#         arg = copy.deepcopy(self.arg)
        
#         arg.n_vector = arg.n_vector_invariant
#         arg.vector_length = arg.vector_length_invariant

#         return arg
    
class Column_trans_rot_fullconv(nn.Module):
    def __init__(
        self,
        arg,
        masking_ratio = 0.75,
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        self.arg = arg
        arg_invariant = copy.deepcopy(arg)
        arg_invariant.n_vector = arg_invariant.n_vector_invariant
        arg_invariant.vector_length = arg_invariant.vector_length_invariant

        self.lgn_ende = Lgn_ende(arg).to(arg.device)
        self.lgn_ende_invariant = Lgn_ende(arg_invariant).to(arg_invariant.device)

        self.apply(weights_init)

    def forward(self, img):
        return self.lgn_ende.fullconv(img)
        
class Column_trans_rot_lgn(nn.Module):
    def __init__(
        self,
        arg,
        masking_ratio = 0.75,
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        self.arg = arg
        print('column_trans_rot: dataset:',self.arg.dataset)
        arg_trans = self.get_trans_arg()
        arg_rot = self.get_rot_arg()

        self.lgn_ende = nn.ModuleList([Lgn_ende(arg) for arg in arg_rot])

        self.apply(weights_init)
        print('len(self.lgn_ende):',len(self.lgn_ende))

    def forward(self,img):
        if 'mnist' in self.arg.dataset or 'cifar' in self.arg.dataset:
            if self.arg.ensemble == 'vanilla' or self.arg.ensemble == 'single' or self.arg.ensemble == 'double':
                lgn = [img]
            elif self.arg.ensemble == 'moe':
                lgn = [lgn_ende.fullconv(img) for lgn_ende in self.lgn_ende]
            elif self.arg.ensemble == 'mix':
                lgn = [lgn_ende.fullconv(img) for lgn_ende in self.lgn_ende]
            elif self.arg.ensemble == 'ensemble':
                lgn = [lgn_ende.fullconv(img) for lgn_ende in self.lgn_ende]
                lgn = [img] + lgn + lgn

        else:
            if self.arg.ensemble == 'vanilla' or self.arg.ensemble == 'single' or self.arg.ensemble == 'double':
                lgn = [img]
            elif self.arg.ensemble == 'moe':
                lgn = [lgn_ende(img) for lgn_ende in self.lgn_ende]
            elif self.arg.ensemble == 'mix':
                lgn = [lgn_ende(img) for lgn_ende in self.lgn_ende]
            elif self.arg.ensemble == 'ensemble':
                lgn = [lgn_ende(img) for lgn_ende in self.lgn_ende]
                lgn = [img] + lgn + lgn

        return lgn

    def get_rot_arg(self):
        arg = []
        # for nv,vl,rn in self.arg.n_vector,self.arg.vector_length,self.arg.rot_num:
        for i,nv in enumerate(self.arg.n_vector):
            nv,vl,rn = self.arg.n_vector[i],self.arg.vector_length[i],self.arg.rot_num[i]
            #16 2 4; 64 1 8
            arg_rot = copy.deepcopy(self.arg)
            arg_rot.n_vector = int(nv/rn)    #4
            arg_rot.vector_length = int(vl*rn)     #8
            arg.append(arg_rot)

        return arg

    def get_trans_arg(self):
        arg = []
        # for nv,vl,rn in self.arg.n_vector,self.arg.vector_length,self.arg.rot_num:
        for i,nv in enumerate(self.arg.n_vector):
            nv,vl,rn = self.arg.n_vector[i],self.arg.vector_length[i],self.arg.rot_num[i]
            arg_trans = copy.deepcopy(self.arg)
            arg_trans.n_vector = nv
            arg_trans.vector_length = vl
            arg.append(arg_trans)
        
        return arg