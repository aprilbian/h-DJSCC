import ipdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor

from compressai.layers import ResidualBlockUpsample
from compressai.layers import ResidualBlock, ResidualBlockWithStride

from compressai.ops.parametrizers import NonNegativeParametrizer

class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out

class AFModule(nn.Module):
    def __init__(self, c_in):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=c_in+1,
                      out_features=c_in),

            nn.LeakyReLU(),

            nn.Linear(in_features=c_in,
                      out_features=c_in),

            nn.Sigmoid()
        )

    def forward(self, x, snr):
        B, _, H, W = x.size()
        context = torch.mean(x, dim=(2, 3))
        snr_context = snr.repeat_interleave(B // snr.size(0), dim=0)

        # snr_context = torch.ones(B, 1, requires_grad=True).to(x.device) * snr
        context_input = torch.cat((context, snr_context), dim=1)
        atten_weights = self.layers(context_input).view(B, -1, 1, 1)
        atten_mask = torch.repeat_interleave(atten_weights, H, dim=2)
        atten_mask = torch.repeat_interleave(atten_mask, W, dim=3)
        out = atten_mask * x
        return out

class EncoderCell(nn.Module):
    def __init__(self, c_in, c_feat, c_out, attn=False):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        if attn:
            self._attn_arch()
        else:
            self._regular_arch()


    def _regular_arch(self):
        self.layers = nn.ModuleDict({
            'rbws1': ResidualBlockWithStride(
                in_ch=self.c_in,
                out_ch=self.c_feat,
                stride=2),

            'rb1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rbws2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),


            'rbws3': ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                stride=2),


            'rb4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
        })
    
    def _attn_arch(self):
        self.layers = nn.ModuleDict({
            'rbws1': ResidualBlockWithStride(
                in_ch=self.c_in,
                out_ch=self.c_feat,
                stride=2),

            'rb1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'af1': AFModule(c_in=self.c_feat),

            'rbws2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'af2': AFModule(c_in=self.c_feat),

            'rbws3': ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                stride=2),

            'af3': AFModule(c_in=self.c_feat),

            'rbws4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
            
            'af4': AFModule(c_in=self.c_out),
        })


    def forward(self, x, snr=torch.tensor([[0]])):

        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr.to(x.device))
            else:
                x = self.layers[key](x)
        
        out = x
        return out


class DecoderCell(nn.Module):
    def __init__(self, c_in, c_feat, c_out, attn):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        if attn:
            self._attn_arch()
        else:
            self._reduced_arch()

    def _reduced_arch(self):
        self.layers = nn.ModuleDict({

            'rb1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            'rbu1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rbu2': ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                upsample=2),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rbu4': ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_out,
                upsample=2),
        })
    
    def _attn_arch(self):
        self.layers = nn.ModuleDict({

            'rb1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            'rbu1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'af1': AFModule(c_in=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rbu2': ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                upsample=2),

            'af2': AFModule(c_in=self.c_feat),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),
            
            'af3': AFModule(c_in=self.c_feat),

            'rbu4': ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_out,
                upsample=2),

            'af4': AFModule(c_in=self.c_out),

        })

    def forward(self, x, snr=torch.tensor([[0]])):

        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr.to(x.device))
            else:
                x = self.layers[key](x)

        out = x
        return out


class EncoderCell_lattn(nn.Module):
    def __init__(self, c_in, c_feat, c_out, attn=False):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        self._attn_arch()


    def _attn_arch(self):
        self.layers = nn.ModuleDict({
            'rbws1': ResidualBlockWithStride(
                in_ch=self.c_in,
                out_ch=self.c_feat,
                stride=2),

            'rb1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'af1': AFModule(c_in=self.c_feat),
            'laf1': AFModule(c_in=self.c_feat),

            'rbws2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'af2': AFModule(c_in=self.c_feat),
            'laf2': AFModule(c_in=self.c_feat),

            'rbws3': ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                stride=2),

            'af3': AFModule(c_in=self.c_feat),
            'laf3': AFModule(c_in=self.c_feat),

            'rbws4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
            
            'af4': AFModule(c_in=self.c_out),
            'laf4': AFModule(c_in=self.c_out),
        })


    def forward(self, x, snr, lamda):

        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr.to(x.device))
            elif key[:2] == 'la':
                x = self.layers[key](x, lamda.to(x.device))
            else:
                x = self.layers[key](x)
        
        out = x
        return out


class DecoderCell_lattn(nn.Module):
    def __init__(self, c_in, c_feat, c_out, attn):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        self._attn_arch()

    
    def _attn_arch(self):
        self.layers = nn.ModuleDict({

            'rb1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            'rbu1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'af1': AFModule(c_in=self.c_feat),
            'laf1': AFModule(c_in=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rbu2': ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                upsample=2),

            'af2': AFModule(c_in=self.c_feat),
            'laf2': AFModule(c_in=self.c_feat),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),
            
            'af3': AFModule(c_in=self.c_feat),
            'laf3': AFModule(c_in=self.c_feat),

            'rbu4': ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_out,
                upsample=2),

            'af4': AFModule(c_in=self.c_out),
            'laf4': AFModule(c_in=self.c_out),
        })

    def forward(self, x, snr, lamda):

        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr.to(x.device))
            elif key[:2] == 'la':
                x = self.layers[key](x, lamda.to(x.device))
            else:
                x = self.layers[key](x)

        out = x
        return out

class EncoderCell_tiny(nn.Module):
    def __init__(self, c_in, c_feat, c_out, attn=False):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        if attn:
            self._attn_arch()
        else:
            self._regular_arch()


    def _regular_arch(self):
        self.layers = nn.ModuleDict({
            'rbws1': ResidualBlockWithStride(
                in_ch=self.c_in,
                out_ch=self.c_feat,
                stride=2),

            'rb1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),
            
            'rb4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb5': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
        })
    
    def _attn_arch(self):
        self.layers = nn.ModuleDict({
            

            'rb1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            'af1': AFModule(c_in=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),
            
            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),
            
            'rb4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb5': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rbws1': ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_out,
                stride=2),

            'af2': AFModule(c_in=self.c_out),
        })


    def forward(self, x, snr=torch.tensor([[0]])):

        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr.to(x.device))
            else:
                x = self.layers[key](x)
        
        out = x
        return out


class DecoderCell_tiny(nn.Module):
    def __init__(self, c_in, c_feat, c_out, attn):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        if attn:
            self._attn_arch()
        else:
            self._reduced_arch()

    def _reduced_arch(self):
        self.layers = nn.ModuleDict({

            'rbu1': ResidualBlockUpsample(
                in_ch=self.c_in,
                out_ch=self.c_feat,
                upsample=2),

            'rb1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),
            
            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
        })
    
    def _attn_arch(self):
        self.layers = nn.ModuleDict({

            'rbu1': ResidualBlockUpsample(
                in_ch=self.c_in,
                out_ch=self.c_feat,
                upsample=2),

            'rb1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'af1': AFModule(c_in=self.c_feat),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),
            
            'rb5': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),

            'af2': AFModule(c_in=self.c_out),
        })

    def forward(self, x, snr=torch.tensor([[0]])):

        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr.to(x.device))
            else:
                x = self.layers[key](x)

        out = x
        return out




class Transform_Layer(nn.Module):
    def __init__(self, c_in, c_feat, c_out, attn=False):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        if attn:
            self._attn_arch()
        else:
            self._regular_arch()


    def _regular_arch(self):
        self.layers = nn.ModuleDict({

            'rb1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
        })
    
    def _attn_arch(self):
        self.layers = nn.ModuleDict({

            'rb1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'af1': AFModule(c_in=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'af2': AFModule(c_in=self.c_feat),

            'rb4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
            
            'af3': AFModule(c_in=self.c_out),
        })


    def forward(self, x, snr=0):

        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr)
            else:
                x = self.layers[key](x)
        
        out = x
        return out




class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.percentage = percentage
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.best_epoch = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics, epoch):
        if self.patience == 0:
            return False, self.best, self.best_epoch, self.num_bad_epochs

        if self.best is None:
            self.best = metrics
            self.best_epoch = epoch
            return False, self.best, self.best_epoch, 0

        if torch.isnan(metrics):
            return True, self.best, self.best_epoch, self.num_bad_epochs

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.best_epoch = epoch
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True, self.best, self.best_epoch, self.num_bad_epochs

        return False, self.best, self.best_epoch, self.num_bad_epochs

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

    def get_state_dict(self):
        state_dict = {
            'best': self.best,
            'best_epoch': self.best_epoch,
            'num_bad_epochs': self.num_bad_epochs,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.best_epoch = state_dict['best_epoch']
        self.num_bad_epochs = state_dict['num_bad_epochs']

    def reset(self):
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.best_epoch = None
        self._init_is_better(self.mode, self.min_delta, self.percentage)




class JSCC(nn.Module):
    def __init__(self, enc, dec, args):
        super().__init__()

        self.c_feat = args['c_feat']
        self.c_out = args['cout']
        self.train_snr = args['train_snr']
        
        self.device = args['device']

        self.enc = enc.to(self.device)
        self.dec = dec.to(self.device)



    def forward(self, img, snr):
        # img: (B,3,H,W); snr -- scalar

        train_snr = torch.tensor([[self.train_snr]], requires_grad=False).to(self.device)
        # Encoder
        x = self.enc(img, train_snr)          # (B,C,H,W)
        B,C,H,W = x.shape
        x = x.view(B,-1,2)                    # (B,C*H*W/2,2)

        # AWGN channel
        Es = torch.mean(torch.view_as_complex(x).abs()**2)
        noise_pwr = torch.sqrt(Es*10**(-snr/10)/2)
        noise = noise_pwr*torch.randn_like(x)
        y = x + noise
        y = y.view(B,C,H,W)

        # Decoder
        output = torch.sigmoid(self.dec(y, train_snr))   # (B,3,H,W)

        return output


# The overall model
class JSCCQ(nn.Module):
    def __init__(self, enc, dec, args):
        super().__init__()

        self.c_feat = args['c_feat']
        self.c_out = args['cout']
        
        self.device = args['device']
        self.train_snr = args['train_snr']
        # The constellation -- we will not update it
        self.n_emb = args['n_embed']  
        self.register_buffer('embed', args['embed'].to(self.device))
        self.embedding = self.embed.transpose(1, 0)

        # soft decision
        self.Kg = args['Kg']                 # annealing
        self.sigma_period = args['period']
        self.sigma_max = args['sigma_max']

        # KL div
        self.commitment = args['commitment']


        self.enc = enc.to(self.device)
        self.dec = dec.to(self.device)

    
    
    def quantize(self, x):

        B = x.shape[0]             # (B,C,H,W)
        x = x.view(B,-1,2)         # 2-dimension, QAM,QPSK etc
        flatten = x.view(-1,2)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embedding
            + self.embedding.pow(2).sum(0, keepdim=True)
        )

        soft_assign = F.softmax(-self.sigma * dist, dim=1)
        likelihoods = torch.mean(soft_assign, dim=0)                              # (n_emb,)
        embed_ind = soft_assign.argmax(1)
        embed_ind = embed_ind.view(*x.shape[:-1])                                 # (B, C*H*W/2,)
        quantize = F.embedding(embed_ind, self.embedding.transpose(0, 1))         # (B, C*H*W/2, 2)

        soft_assign = torch.matmul(soft_assign, self.embedding.transpose(0, 1))   # (B, C*H*W/2, 2)
        soft_assign = soft_assign.view(*x.shape)

        quantize = soft_assign + (quantize - soft_assign).detach()

        '''
        self.norm_factor = torch.sqrt(
            torch.matmul(likelihoods,
                         torch.square(torch.view_as_complex(self.embedding.transpose(0, 1)).abs()))
        )

        quantize = quantize / self.norm_factor'''
        return quantize, likelihoods

    def linear_anneal(self,t):
        self.sigma = self.Kg * (t // self.sigma_period + 1)
        self.sigma = np.clip(0, self.sigma_max, self.sigma)

    def forward(self, img, snr):
        # img: (B,3,H,W); snr -- scalar
        train_snr = torch.tensor([[snr]], requires_grad=False).to(self.device)
        # Encoder
        x = self.enc(img)          # (B,cout,H,W)
        (B,C_,H_,W_) = x.shape

        # Quantize
        quantized, likelihoods = self.quantize(x)  #(B, C*H*W/2, 2), (n_emb,)

        # AWGN channel
        quantized = quantized.view(-1, 2)
        Es = torch.mean(torch.view_as_complex(quantized).abs() ** 2)
        noise_pwr = torch.sqrt(Es*10**(-train_snr/10)/2)
        noise = noise_pwr*torch.randn_like(quantized)
        y = quantized + noise
        y = y.view(B,C_,H_,W_)

        # Decoder
        output = torch.sigmoid(self.dec(y))   # (B,3,H,W)

        return output, likelihoods


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.percentage = percentage
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.best_epoch = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics, epoch):
        if self.patience == 0:
            return False, self.best, self.best_epoch, self.num_bad_epochs

        if self.best is None:
            self.best = metrics
            self.best_epoch = epoch
            return False, self.best, self.best_epoch, 0

        if torch.isnan(metrics):
            return True, self.best, self.best_epoch, self.num_bad_epochs

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.best_epoch = epoch
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True, self.best, self.best_epoch, self.num_bad_epochs

        return False, self.best, self.best_epoch, self.num_bad_epochs

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

    def get_state_dict(self):
        state_dict = {
            'best': self.best,
            'best_epoch': self.best_epoch,
            'num_bad_epochs': self.num_bad_epochs,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.best_epoch = state_dict['best_epoch']
        self.num_bad_epochs = state_dict['num_bad_epochs']

    def reset(self):
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.best_epoch = None
        self._init_is_better(self.mode, self.min_delta, self.percentage)