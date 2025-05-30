import numpy as np
from modules import *
from utils import *

# Define digital compressor
import numpy as np
import torch
import math
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

import time
# Define multi-hop semantic communication, 
# The mobile user transmits the signal using DeepJSCC scheme, 
# while the network uses digital scheme to relay the message

class Adhoc(nn.Module):
    def __init__(self,  args):
        super().__init__()

        self.link_quals = args.link_qual
        self.device = args.device

        # DeepJSCC
        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.enc = EncoderCell(3, self.c_feat, self.c_out, attn = args.adapt).to(self.device)     # DeepJSCC encoder
        self.dec = DecoderCell(self.c_out, self.c_feat, 3, attn = args.adapt).to(self.device)     # DeepJSCC decoder

        # latent compressor: latent -> (reconstructed latent, bit cost)
        #self.compressor = GainedMSHyperprior(args).to(self.device)
        self.compressor = GainedMSHyperprior_large(args).to(self.device)
        
        self.args = args

        self.adapt = args.adapt
        self.fading = args.fading

    def forward(self, img, snr, s):
        # img: (B,3,H,W)
        time_list = []
        # DeepJSCC encoding
        if self.fading:
            coef_h = complex_sig([1], self.device)
            abs_h = torch.abs(coef_h)**2
            h_dB = 10*torch.log10(abs_h)
            eff_snr = snr + h_dB

            x = self.enc(img, eff_snr)
        else:
            s_times = time.time()
            x = self.enc(img, snr)
            s_timet = time.time()
            time_list.append(s_timet - s_times)

        B,C,H,W = x.shape

        sig_s = x.view(B,-1,2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s)

        noise_shape = [B, int(C*H*W/2)]
        noise_lvl = snr
        noise_n = 10**(-noise_lvl/20)*complex_sig(noise_shape, self.device)
        if self.args.fading:
            y = torch.abs(coef_h)*sig_s + noise_n
            y = torch.abs(coef_h)*y/(torch.abs(coef_h)**2+10**(-noise_lvl/10))
        else:
            y = sig_s + noise_n


        latent = torch.view_as_real(y).float().contiguous().view(B,C,H,W)
        
        if self.args.fading:
            x_hat = self.dec(latent, eff_snr)
            rec_x, est_bits = self.compressor(x_hat, eff_snr, s)
        else:
            r_times = time.time()
            x_hat = self.dec(latent, snr)
            r_timet = time.time()
            time_list.append(r_timet - r_times)

            d_times = time.time()
            rec_x, est_bits = self.compressor(x_hat, snr, s)
            d_timet = time.time()
            time_list.append(d_timet - d_times)

        return rec_x, est_bits, time_list

class Adhoc_LATTN(nn.Module):
    def __init__(self,  args):
        super().__init__()

        self.link_quals = args.link_qual
        self.device = args.device

        # DeepJSCC
        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.enc = EncoderCell_lattn(3, self.c_feat, self.c_out, attn = args.adapt).to(self.device)     # DeepJSCC encoder
        self.dec = DecoderCell_lattn(self.c_out, self.c_feat, 3, attn = args.adapt).to(self.device)     # DeepJSCC decoder

        # latent compressor: latent -> (reconstructed latent, bit cost)
        self.compressor = GainedMSHyperprior_large(args).to(self.device)
        
        self.args = args

        self.adapt = args.adapt
        self.fading = args.fading

    def forward(self, img, snr, s):
        # img: (B,3,H,W)
        # DeepJSCC encoding
        if self.fading:
            coef_h = complex_sig([1], self.device)
            abs_h = torch.abs(coef_h)**2
            h_dB = 10*torch.log10(abs_h)
            eff_snr = snr + h_dB

            x = self.enc(img, eff_snr, torch.tensor([[s]]))
        else:
            x = self.enc(img, snr, torch.tensor([[s]]))

        B,C,H,W = x.shape

        sig_s = x.view(B,-1,2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s)

        noise_shape = [B, int(C*H*W/2)]
        noise_lvl = snr
        noise_n = 10**(-noise_lvl/20)*complex_sig(noise_shape, self.device)
        if self.args.fading:
            y = torch.abs(coef_h)*sig_s + noise_n
            y = torch.abs(coef_h)*y/(torch.abs(coef_h)**2+10**(-noise_lvl/10))
        else:
            y = sig_s + noise_n


        latent = torch.view_as_real(y).float().contiguous().view(B,C,H,W)
        
        if self.args.fading:
            x_hat = self.dec(latent, eff_snr, torch.tensor([[s]]))
            rec_x, est_bits = self.compressor(x_hat, eff_snr, s)
        else:
            x_hat = self.dec(latent, snr, torch.tensor([[s]]))
            rec_x, est_bits = self.compressor(x_hat, snr, s)


        return rec_x, est_bits


class Adhocv1(nn.Module):
    def __init__(self,  args):
        super().__init__()

        self.link_quals = args.link_qual
        self.device = args.device

        # DeepJSCC
        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.enc = EncoderCell(3, self.c_feat, self.c_out, attn = True).to(self.device)     # DeepJSCC encoder
        self.dec = DecoderCell(self.c_out, self.c_feat, 3, attn = True).to(self.device)     # DeepJSCC decoder

        # latent compressor: latent -> (reconstructed latent, bit cost)
        self.compressor = ScaleHyperprior_large(args).to(self.device)

        # This is for the second hop
        self.enc2 = EncoderCell(3, self.c_feat, self.c_out, attn = True).to(self.device)     # DeepJSCC encoder
        self.dec2 = DecoderCell(self.c_out, self.c_feat, 3, attn = True).to(self.device)     # DeepJSCC decoder
        
        self.args = args

        self.adapt = args.adapt

    def forward(self, img, snr, snr2):
        # img: (B,3,H,W)

        # DeepJSCC encoding
        x = self.enc(img, snr)

        B,C,H,W = x.shape

        sig_s = x.view(B,-1,2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s)

        noise_shape = [B, int(C*H*W/2)]
        noise_lvl = snr
        noise_n = 10**(-noise_lvl/20)*complex_sig(noise_shape, self.device)

        y = sig_s + noise_n


        latent = torch.view_as_real(y).float().contiguous().view(B,C,H,W)

        x_hat = self.dec(latent, snr)

        # Latent compression model
        rec_x, est_bits = self.compressor(x_hat, snr.item())

        x = self.enc2(rec_x, snr2)

        sig_s = x.view(B,-1,2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s)

        noise_lvl = snr2
        noise_n = 10**(-noise_lvl/20)*complex_sig(noise_shape, self.device)

        y = sig_s + noise_n


        latent = torch.view_as_real(y).float().contiguous().view(B,C,H,W)

        x_hat = self.dec2(latent, snr2)

        return x_hat, est_bits

class Wired_to_Wireless(nn.Module):
    def __init__(self,  args):
        super().__init__()

        self.link_quals = args.link_qual
        self.device = args.device

        # DeepJSCC
        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.enc = EncoderCell(3, self.c_feat, self.c_out, attn = True).to(self.device)     # DeepJSCC encoder
        self.dec = DecoderCell(self.c_out, self.c_feat, 3, attn = True).to(self.device)     # DeepJSCC decoder

        # latent compressor: latent -> (reconstructed latent, bit cost)
        self.compressor = ScaleHyperprior_latent(args).to(self.device)

        
        self.args = args

        self.adapt = args.adapt

    def forward(self, img, snr):
        # img: (B,3,H,W)

        # DeepJSCC encoding
        x = self.enc(img, snr)

        rec_x, est_bits = self.compressor(x, snr.item())

        B,C,H,W = x.shape

        sig_s = rec_x.view(B,-1,2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s)

        noise_shape = [B, int(C*H*W/2)]
        noise_lvl = snr
        noise_n = 10**(-noise_lvl/20)*complex_sig(noise_shape, self.device)

        y = sig_s + noise_n


        latent = torch.view_as_real(y).float().contiguous().view(B,C,H,W)

        x_hat = self.dec(latent, snr)


        return x_hat, est_bits


class Adhoc_CSIR(nn.Module):
    def __init__(self,  args):
        super().__init__()

        self.link_quals = args.link_qual
        self.device = args.device

        # DeepJSCC
        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.enc = EncoderCell(3, self.c_feat, self.c_out, attn = args.adapt).to(self.device)     # DeepJSCC encoder
        self.dec = DecoderCell(self.c_out, self.c_feat, 3, attn = args.adapt).to(self.device)     # DeepJSCC decoder

        # latent compressor: latent -> (reconstructed latent, bit cost)
        #self.compressor = GainedMSHyperprior(args).to(self.device)
        self.compressor = GainedMSHyperprior_large(args).to(self.device)
        
        self.args = args

        self.adapt = args.adapt
        self.fading = args.fading

    def forward(self, img, snr, s):
        # img: (B,3,H,W)

        # DeepJSCC encoding
        coef_h = complex_sig([1], self.device)
        abs_h = torch.abs(coef_h)**2
        h_dB = 10*torch.log10(abs_h)
        eff_snr = snr + h_dB

        x = self.enc(img, snr)


        B,C,H,W = x.shape

        sig_s = x.view(B,-1,2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s)

        noise_shape = [B, int(C*H*W/2)]
        noise_lvl = snr
        noise_n = 10**(-noise_lvl/20)*complex_sig(noise_shape, self.device)

        y = coef_h*sig_s + noise_n
        y = torch.conj(coef_h)*y/(torch.abs(coef_h)**2+10**(-noise_lvl/10))

        latent = torch.view_as_real(y).float().contiguous().view(B,C,H,W)


        x_hat = self.dec(latent, snr)
        rec_x, est_bits = self.compressor(x_hat, eff_snr, s)

        return rec_x, est_bits




class Adhoc_CSIR_LATTN(nn.Module):
    def __init__(self,  args):
        super().__init__()

        self.link_quals = args.link_qual
        self.lamda_list = torch.tensor(args.lamdas)
        self.device = args.device

        # DeepJSCC
        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.enc = EncoderCell_lattn(3, self.c_feat, self.c_out, attn = args.adapt).to(self.device)     # DeepJSCC encoder
        self.dec = DecoderCell_lattn(self.c_out, self.c_feat, 3, attn = args.adapt).to(self.device)     # DeepJSCC decoder

        # latent compressor: latent -> (reconstructed latent, bit cost)
        #self.compressor = GainedMSHyperprior(args).to(self.device)
        self.compressor = GainedMSHyperprior_large(args).to(self.device)
        
        self.args = args
        

        self.adapt = args.adapt
        self.fading = args.fading

    def forward(self, img, snr, s):
        # img: (B,3,H,W)

        # DeepJSCC encoding
        coef_h = complex_sig([1], self.device)
        abs_h = torch.abs(coef_h)**2
        h_dB = 10*torch.log10(abs_h)
        eff_snr = snr + h_dB

        #x = self.enc(img, snr, self.lamda_list[s].view(1, 1))
        x = self.enc(img, snr, torch.tensor([[s]]))


        B,C,H,W = x.shape

        sig_s = x.view(B,-1,2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s)

        noise_shape = [B, int(C*H*W/2)]
        noise_lvl = snr
        noise_n = 10**(-noise_lvl/20)*complex_sig(noise_shape, self.device)

        y = coef_h*sig_s + noise_n
        y = torch.conj(coef_h)*y/(torch.abs(coef_h)**2+10**(-noise_lvl/10))

        latent = torch.view_as_real(y).float().contiguous().view(B,C,H,W)


        #x_hat = self.dec(latent, snr, self.lamda_list[s].view(1, 1))
        x_hat = self.dec(latent, snr, torch.tensor([[s]]))

        rec_x, est_bits = self.compressor(x_hat, eff_snr, s)

        return rec_x, est_bits




class Adhoc_Full_Adapt(nn.Module):
    def __init__(self,  args):
        super().__init__()

        self.link_quals = args.link_qual
        self.device = args.device

        # DeepJSCC
        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.enc = EncoderCell(3, self.c_feat, self.c_out, attn = args.adapt).to(self.device)     # DeepJSCC encoder
        self.dec = DecoderCell(self.c_out, self.c_feat, 3, attn = args.adapt).to(self.device)     # DeepJSCC decoder

        # latent compressor: latent -> (reconstructed latent, bit cost)
        #self.compressor = GainedMSHyperprior(args).to(self.device)
        self.compressor = ScaleHyperprior_Full_Adapt(args).to(self.device)
        
        self.args = args

        self.adapt = args.adapt

    def forward(self, img, snr):
        # img: (B,3,H,W)

        # DeepJSCC encoding
        if self.adapt:
            x = self.enc(img, snr)


        B,C,H,W = x.shape

        sig_s = x.view(B,-1,2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s)

        noise_shape = [B, int(C*H*W/2)]
        noise_lvl = snr
        noise_n = 10**(-noise_lvl/20)*complex_sig(noise_shape, self.device)
        if self.args.fading:
            coef_shape = [B, 1]
            coef_h = complex_sig(coef_shape, self.device)
            y = coef_h*sig_s + noise_n
            y = torch.conj(coef_h)*y/(torch.abs(coef_h)**2+10**(-noise_lvl/10))
        else:
            y = sig_s + noise_n

            # optional -> MMSE estimation on y
            y = 1/torch.sqrt(1+10**(-noise_lvl/10))*y

        latent = torch.view_as_real(y).float().contiguous().view(B,C,H,W)
        if self.adapt:
            x_hat = self.dec(latent, snr)


        # Latent compression model
        rec_x, est_bits = self.compressor(x_hat, snr)
        #rec_x, est_bits = self.compressor(x_hat)


        return rec_x, est_bits



class Adhoc_SEP(nn.Module):
    def __init__(self,  args):
        super().__init__()

        self.link_quals = args.link_qual
        self.device = args.device

        # DeepJSCC
        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.enc = EncoderCell(3, self.c_feat, self.c_out, attn = args.adapt).to(self.device)     # DeepJSCC encoder
        self.dec = DecoderCell(self.c_out, self.c_feat, 3, attn = args.adapt).to(self.device)     # DeepJSCC decoder

        # latent compressor: latent -> (reconstructed latent, bit cost)
        #self.compressor = GainedMSHyperprior(args).to(self.device)
        self.compressor = ScaleHyperprior(args).to(self.device)
        
        self.args = args

        self.adapt = args.adapt

    def forward(self, img):
        # img: (B,3,H,W)

        # DeepJSCC encoding
        x = self.enc(img)

        B,C,H,W = x.shape

        sig_s = x.view(B,-1,2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s)

        noise_shape = [B, int(C*H*W/2)]
        noise_lvl = self.link_quals
        noise_n = 10**(-noise_lvl/20)*complex_sig(noise_shape, self.device)
        if self.args.fading:
            coef_shape = [B, 1]
            coef_h = complex_sig(coef_shape, self.device)
            y = coef_h*sig_s + noise_n
            y = torch.conj(coef_h)*y/(torch.abs(coef_h)**2+10**(-noise_lvl/10))
        else:
            y = sig_s + noise_n

            # optional -> MMSE estimation on y
            y = 1/torch.sqrt(1+10**(-noise_lvl/10))*y

        latent = torch.view_as_real(y).float().contiguous().view(B,C,H,W)

        x_hat = self.dec(latent)


        # Latent compression model
        rec_x, est_bits = self.compressor(x_hat)
        #rec_x, est_bits = self.compressor(x_hat)


        return rec_x, est_bits


class ScaleHyperprior_Full_Adapt(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.btneck_feat = args.btneck_feat
        self.btneck_sz = args.btneck_sz

        self.g_a = EncoderCell(3, args.cfeat, self.btneck_feat, attn = args.adapt)
        self.g_s = DecoderCell(self.btneck_feat, args.cfeat, 3, attn = args.adapt)
        self.h_a = nn.Sequential(
            nn.Conv2d(self.btneck_feat+1, self.btneck_sz, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.btneck_sz, self.btneck_sz, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.btneck_sz, self.btneck_sz, 5, stride=2, padding=2),
        )

        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(self.btneck_sz+1, self.btneck_feat, 5,
                               stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.btneck_feat, self.btneck_feat, 5,
                               stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.btneck_feat, 2*self.btneck_feat, 3, stride=1, padding=1)
        )
        self.entropy_bottleneck = EntropyBottleneck(self.btneck_sz)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, s):
        
        y = self.g_a(x, s)
        B, _, H_, W_ = y.shape
        extend_s = torch.ones(B, 1, H_, W_).to(y.device)
        extend_y = torch.cat((y, extend_s), dim = 1)
        z = self.h_a((extend_y))
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        B, _, H_, W_ = z_hat.shape
        extend_s = torch.ones(B, 1, H_, W_).to(z_hat.device)
        extend_z = torch.cat((z_hat, extend_s), dim = 1)
        scales_hat = self.h_s(extend_z)
        scales_hat, means_hat = scales_hat.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)
        x_hat = self.g_s(y_hat, s)

        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2)*32*32) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2)*32*32) / B

        return x_hat, bpp_y + bpp_z


    

class ScaleHyperprior(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.btneck_feat = args.btneck_feat
        self.btneck_sz = args.btneck_sz

        self.g_a = EncoderCell(3, args.cfeat, self.btneck_feat, attn = args.adapt)
        self.g_s = DecoderCell(self.btneck_feat, args.cfeat, 3, attn = args.adapt)
        self.h_a = nn.Sequential(
            nn.Conv2d(self.btneck_feat+1, self.btneck_sz, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.btneck_sz, self.btneck_sz, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.btneck_sz, self.btneck_sz, 5, stride=2, padding=2),
        )

        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(self.btneck_sz+1, self.btneck_feat, 5,
                               stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.btneck_feat, self.btneck_feat, 5,
                               stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.btneck_feat, 2*self.btneck_feat, 3, stride=1, padding=1)
        )
        self.entropy_bottleneck = EntropyBottleneck(self.btneck_sz)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, s):
        s = torch.tensor([[s]])
        
        y = self.g_a(x, s)
        B, _, H_, W_ = y.shape
        extend_s = torch.ones(B, 1, H_, W_).to(y.device)
        extend_y = torch.cat((y, extend_s), dim = 1)
        z = self.h_a(torch.abs(extend_y))
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        B, _, H_, W_ = z_hat.shape
        extend_s = torch.ones(B, 1, H_, W_).to(z_hat.device)
        extend_z = torch.cat((z_hat, extend_s), dim = 1)
        scales_hat = self.h_s(extend_z)
        scales_hat, means_hat = scales_hat.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)
        x_hat = self.g_s(y_hat, s)

        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2)*32*32) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2)*32*32) / B

        return x_hat, bpp_y + bpp_z

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    


class GainedMSHyperprior(nn.Module):
    '''
    Bottleneck scaling version.
    '''
    def __init__(self, args):
        super().__init__()
        

        self.btneck_feat = args.btneck_feat
        self.btneck_sz = args.btneck_sz

        self.g_a = EncoderCell(3, args.cfeat, self.btneck_feat, attn = args.adapt)
        self.g_s = DecoderCell(self.btneck_feat, args.cfeat, 3, attn = args.adapt)
        self.h_a = nn.Sequential(
            ResidualBlock(self.btneck_feat, self.btneck_feat),
            ResidualBlockWithStride(self.btneck_feat, self.btneck_sz),
            ResidualBlockWithStride(self.btneck_sz, self.btneck_sz)
        )

        self.h_s = nn.Sequential(
            ResidualBlock(self.btneck_sz, self.btneck_sz),
            ResidualBlockUpsample(self.btneck_sz, self.btneck_feat),
            ResidualBlockUpsample(self.btneck_feat, 2*self.btneck_feat),
        )

        self.entropy_bottleneck = EntropyBottleneck(self.btneck_sz)
        self.gaussian_conditional = GaussianConditional(None)

        self.lamdas = args.snrs  # mxh add from HUAWEI CVPR2021 Gained...

        # Condition on Latent y, so the gain vector length M
        # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
        # treat all channels the same in initialization
        self.levels = len(self.lamdas) # 8
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, self.btneck_feat]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, self.btneck_feat]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, self.btneck_sz]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, self.btneck_sz]), requires_grad=True)


    def forward(self, x, s):
        '''
            x: input image
            s: random num to choose gain vector
        '''
        snr = torch.tensor([[self.lamdas[s]]]).to(x.device)
        y = self.g_a(x, snr)
        y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]
        z = self.h_a(y)
        z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat, snr)

        B = y.shape[0]

        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2)*32*32) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2)*32*32) / B

        return x_hat, bpp_y + bpp_z
    



class ScaleHyperprior_large(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.btneck_feat = args.btneck_feat
        self.btneck_sz = args.btneck_sz

        self.g_a = EncoderCell(3, args.cfeat, self.btneck_feat, attn = args.adapt)
        self.g_s = DecoderCell(self.btneck_feat, args.cfeat, 3, attn = args.adapt)
        self.h_a = EncoderCell(self.btneck_feat, args.cfeat, self.btneck_sz, attn = args.adapt)
        self.h_s = DecoderCell(self.btneck_sz, args.cfeat, 2*self.btneck_feat, attn = args.adapt)

        self.entropy_bottleneck = EntropyBottleneck(self.btneck_sz)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, s):
        s = torch.tensor([[s]])
        B, _, H, W = x.shape
        
        y = self.g_a(x, s)
        z = self.h_a(y, s)
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        scales_hat = self.h_s(z_hat, s)
        scales_hat, means_hat = scales_hat.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)
        x_hat = self.g_s(y_hat, s)


        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2)*H*W) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2)*H*W) / B

        #z_string = self.entropy_bottleneck.compress(z)
        #y_string = self.gaussian_conditional.compress()

        return x_hat, bpp_y + bpp_z



class ScaleHyperprior_latent(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.btneck_feat = args.btneck_feat
        self.btneck_sz = args.btneck_sz

        self.g_a = EncoderCell_tiny(args.cout, args.cfeat, self.btneck_feat, attn = args.adapt)
        self.g_s = DecoderCell_tiny(self.btneck_feat, args.cfeat, args.cout, attn = args.adapt)
        self.h_a = EncoderCell_tiny(self.btneck_feat, args.cfeat, self.btneck_sz, attn = args.adapt)
        self.h_s = DecoderCell_tiny(self.btneck_sz, args.cfeat, 2*self.btneck_feat, attn = args.adapt)

        self.entropy_bottleneck = EntropyBottleneck(self.btneck_sz)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, s):
        s = torch.tensor([[s]])
        B, _, H, W = x.shape
        
        y = self.g_a(x, s)
        z = self.h_a(y, s)
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        scales_hat = self.h_s(z_hat, s)
        scales_hat, means_hat = scales_hat.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)
        x_hat = self.g_s(y_hat, s)


        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2)*32*32) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2)*32*32) / B

        #z_string = self.entropy_bottleneck.compress(z)
        #y_string = self.gaussian_conditional.compress()

        return x_hat, bpp_y + bpp_z



class ScaleHyperprior_dual(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.btneck_feat = args.btneck_feat
        self.btneck_sz = args.btneck_sz

        self.g_a = EncoderCell(3, args.cfeat, self.btneck_feat, attn = args.adapt)
        self.g_s = DecoderCell(self.btneck_feat, args.cfeat, 3, attn = args.adapt)
        self.h_a = EncoderCell(self.btneck_feat, args.cfeat, self.btneck_sz, attn = args.adapt)
        self.h_s = DecoderCell(self.btneck_sz, args.cfeat, 2*self.btneck_feat, attn = args.adapt)

        self.entropy_bottleneck = EntropyBottleneck(self.btneck_sz)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, s):
        s = torch.tensor([[s]])
        B, _, H, W = x.shape
        
        y = self.g_a(x, s)
        z = self.h_a(y, s)
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        scales_hat = self.h_s(z_hat, s)
        scales_hat, means_hat = scales_hat.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)
        x_hat = self.g_s(y_hat, s)


        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2)*H*W) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2)*H*W) / B

        #z_string = self.entropy_bottleneck.compress(z)
        #y_string = self.gaussian_conditional.compress()

        return x_hat, bpp_y + bpp_z


class GainedMSHyperprior_large(nn.Module):
    '''
    Bottleneck scaling version.
    '''
    def __init__(self, args):
        super().__init__()
        

        self.btneck_feat = args.btneck_feat
        self.btneck_sz = args.btneck_sz

        self.g_a = EncoderCell(3, args.cfeat, self.btneck_feat, attn = args.adapt)
        self.g_s = DecoderCell(self.btneck_feat, args.cfeat, 3, attn = args.adapt)
        self.h_a = EncoderCell(self.btneck_feat, args.cfeat, self.btneck_sz, attn = args.adapt)
        self.h_s = DecoderCell(self.btneck_sz, args.cfeat, 2*self.btneck_feat, attn = args.adapt)

        self.entropy_bottleneck = EntropyBottleneck(self.btneck_sz)
        self.gaussian_conditional = GaussianConditional(None)

        self.lamdas = args.lamdas

        self.levels = len(self.lamdas) # 8
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, self.btneck_feat]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, self.btneck_feat]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, self.btneck_sz]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, self.btneck_sz]), requires_grad=True)


    def forward(self, x, snr, s):
        '''
            x: input image
            s: random num to choose gain vector
        '''
        B, _, H, W = x.shape
        y = self.g_a(x, snr)
        y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]

        z = self.h_a(y, snr)

        z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        gaussian_params = self.h_s(z_hat, snr)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat, snr)


        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2)*H*W) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2)*H*W) / B


        return x_hat, bpp_y + bpp_z