import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import warnings
from torch import einsum
from einops import rearrange
import numpy as np
import scipy.io as io

class CA(nn.Module):
    '''CA is channel attention'''
    def __init__(self,n_feats,kernel_size=3,bias=True, bn=False, act=nn.ReLU(True),res_scale=1,conv=nn.Conv2d,reduction=4):

        super(CA, self).__init__()
        
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size,padding = kernel_size//2))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        
        self.conv_du = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(n_feats, n_feats // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats // reduction, n_feats, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        

    def forward(self, x):
        # input zk - zk_1
        x = self.body(x)
        CA = self.conv_du(x)
        CA = torch.mul(x, CA)
        return CA

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y


def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)

        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

# Cross Phase MSA
class CS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=28,
            heads=8,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size

        # position embedding
        seq_l = window_size[0] * window_size[1]
        self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
        self.pos_emb2 = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
        trunc_normal_(self.pos_emb)
        trunc_normal_(self.pos_emb2)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_kv2 = nn.Linear(dim, inner_dim * 2, bias=False)
        self.vfusion = nn.Linear(inner_dim * 2, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x , px):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'
        
        x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
        px_inp = rearrange(px, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
            
        q = self.to_q(x_inp)
                
        k, v = self.to_kv(x_inp).chunk(2, dim=-1)
        pk, pv = self.to_kv2(px_inp).chunk(2, dim=-1)
            
        q, k, v, pk, pv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v, pk, pv))
        q *= self.scale
            
            
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
            
        sim2 = einsum('b h i d, b h j d -> b h i j', q, pk)
        sim2 = sim2 + self.pos_emb2
        attn2 = sim2.softmax(dim=-1)
            
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
            
        out2 = einsum('b h i j, b h j d -> b h i d', attn2, pv)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        finalout = self.vfusion(torch.cat([out,out2],dim=2))
        finalout = self.to_out(finalout)
        finalout = rearrange(finalout, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],b0=w_size[0])

        return finalout
        
class CSAB(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, CS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads)),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x,px):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1).contiguous()
        px = px.permute(0, 2, 3, 1).contiguous()
        for (attn, ff) in self.blocks:
            x = attn(x,px) + x
            x = ff(x) + x
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1).contiguous()
        
# cross stage transformer layer
class CSFormer(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28):
        super(CSFormer, self).__init__()
        self.dim = dim
        #self.scales = len(num_blocks)
        dim_scale = dim
        # Input projection
        self.embedding = nn.Conv2d(in_dim, dim_scale, 3, 1, 1, bias=False)
          
        self.CS_MSA = CSAB(dim=dim_scale, num_blocks=1, dim_head=dim, heads=dim_scale // dim)
        self.CS_MSA2 = CSAB(dim=dim_scale, num_blocks=1, dim_head=dim, heads=dim_scale // dim)
        self.CS_MSA4 = CSAB(dim=dim_scale, num_blocks=1, dim_head=dim, heads=dim_scale // dim)

        self.fusion = nn.Conv2d(dim_scale*2,dim_scale,kernel_size=1,bias=False)
        self.fusion2 = nn.Conv2d(dim_scale*2,dim_scale,kernel_size=1,bias=False)
        
        self.down1 = nn.Conv2d(dim_scale, dim_scale, 4, 2, 1, bias=False)
        self.down2 = nn.Conv2d(dim_scale, dim_scale, 4, 2, 1, bias=False)
        self.down3 = nn.Conv2d(dim_scale, dim_scale, 4, 2, 1, bias=False)
        self.down4 = nn.Conv2d(dim_scale, dim_scale, 4, 2, 1, bias=False)
        
        self.up1 = nn.ConvTranspose2d(dim_scale, dim_scale, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.up2 = nn.ConvTranspose2d(dim_scale, dim_scale, stride=2, kernel_size=2, padding=0, output_padding=0)
        # in phase multi-head self-attention IP-MSA

        # Output projection
        self.mapping = nn.Conv2d(dim_scale, out_dim, 3, 1, 1, bias=False)
        
        #### activation function
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x,px):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        px = F.pad(px, [0, pad_w, 0, pad_h], mode='reflect')

        # Embedding
        fea = self.embedding(x)
        pfea = self.embedding(px)
                
        fea2 = self.down1(fea)
        fea4 = self.down2(fea2)
        pfea2 = self.down3(pfea)
        pfea4 = self.down4(pfea2)
        
        fea4 = self.CS_MSA4(fea4,pfea4)
        fea4 = self.up1(fea4)
        fea2 = self.fusion2(torch.cat([fea2,fea4],dim=1))
        
        fea2 = self.CS_MSA2(fea2,pfea2)
        fea2 = self.up2(fea2)

        fea = self.fusion(torch.cat([fea,fea2],dim=1))
        out = self.CS_MSA(fea,pfea)

        # Mapping
        out = self.mapping(out) + x
        return out[:, :, :h_inp, :w_inp]
        
class HyPaNet(nn.Module):
    def __init__(self, in_nc=29, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):
        x = self.down_sample(self.relu(self.fution(x)))
        x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6
        return x
        

class MAUN(nn.Module):

    def __init__(self,num_iterations = 19):
        super(MAUN, self).__init__()
        self.denoisers = nn.ModuleList([])
        self.para_estimator = HyPaNet(in_nc=28, out_nc=num_iterations)
        self.CAs = nn.ModuleList([])      
        for _ in range(num_iterations):
            self.denoisers.append(
                CSFormer(in_dim=28, out_dim=28, dim=28),
            )
            self.CAs.append(
            CA(28),
            )   
            
        self.fution = nn.Conv2d(56, 28, 1, padding=0, bias=True)
        self.num_iterations = num_iterations
        
    def initial(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: temp: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        nC, step = 28, 2
        y = y / nC * 2
        bs,row,col = y.shape
        y_shift = torch.zeros(bs, nC, row, col).cuda().float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fution(torch.cat([y_shift, Phi], dim=1))
        alpha = self.para_estimator(self.fution(torch.cat([y_shift, Phi], dim=1)))
        return z, alpha
    
    def forward(self, y, input_mask=None):
        if input_mask==None:
            Phi = torch.rand((1,28,256,310)).cuda()
            Phi_s = torch.rand((1, 256, 310)).cuda()
        else:
            Phi, Phi_s = input_mask

        # y = y.squeeze(dim=0)
        x, alphas = self.initial(y, Phi)
          
        sp = x # start point
        p_last = x # output of last phase

        for i in range(self.num_iterations):
            alpha = alphas[:,i,:,:]
            Phi_z = A(sp, Phi)
            sp = sp + At(torch.div(y - Phi_z, alpha + Phi_s), Phi) 
            sp = shift_back_3d(sp)
            
            if i == 0:
                cross_phase = sp
            else:
                cross_phase = cross_phase_save
                
            cross_phase_save = sp
            p_cur = self.denoisers[i](sp,cross_phase)
            
            
            if i<self.num_iterations-1:
                # get output of current phase
                p_cur = shift_3d(p_cur)
                # update start point
                sp = p_cur + self.CAs[i](p_cur - p_last)
                # update output of last phase
                p_last = p_cur
                
        return p_cur[:, :, :, 0:256]
