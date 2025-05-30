import os
import cv2
import csv
import numpy as np
from pytorch_msssim import ms_ssim, ssim

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2.0, dtype=torch.float32)

    if mse == 0:
        return torch.tensor([100.0])

    PIXEL_MAX = 255.0

    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def get_imagenet_list(path):
    fns = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            fns.append(row[0])
    # with open('valid_images.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     with futures.ProcessPoolExecutor() as executor:
    #         for fn, result in zip(glob.iglob(path + '/*JPEG'), executor.map(check_size, glob.iglob(path + '/*JPEG'))):
    #             if result:
    #                 writer.writerow([fn])
                    # fns.append(fn)
    # for fn in glob.iglob(path + '/*JPEG'):
    #     fns.append(fn)
        # image = cv2.imread(fn)
        # height, width, _ = image.shape
        # if height >= 256 and width >= 256:
        #     fns.append(fn)
    return fns


def complex_sig(shape, device):
    sig_real = torch.randn(*shape)
    sig_imag = torch.randn(*shape)
    return (torch.complex(sig_real, sig_imag)/np.sqrt(2)).to(device)

def pwr_normalize(sig):
    _, num_ele = sig.shape[0], torch.numel(sig[0])
    pwr_sig = torch.sum(torch.abs(sig)**2, dim=-1)/num_ele
    sig = sig/torch.sqrt(pwr_sig.unsqueeze(-1))

    return sig

def check_size(fn):
    image = cv2.imread(fn)
    height, width, _ = image.shape
    if height >= 256 and width >= 256:
        return True
    else:
        return False


def np_to_torch(img):
    img = np.swapaxes(img, 0, 1)  # w, h, c
    img = np.swapaxes(img, 0, 2)  # c, h, w
    return torch.from_numpy(img).float()


def to_chan_last(img):
    img = img.transpose(1, 2)
    img = img.transpose(2, 3)
    return img


def as_img_array(image):
    image = image.clamp(0, 1) * 255.0
    return torch.round(image)

def save_nets(job_name, nets, epoch):
    path = '{}/{}.pth'.format('models', job_name)

    if not os.path.exists('models'):
        print('Creating model directory: {}'.format('models'))
        os.makedirs('models')

    torch.save({
        'jscc_model': nets.state_dict(),
        'epoch': epoch
    }, path)


def load_weights(job_name, nets, path = None, strict = False):
    if path == None:
        path = '{}/{}.pth'.format('models', job_name)

    cp = torch.load(path, map_location = torch.device('cuda:1'))
    nets.load_state_dict(cp['jscc_model'], strict = strict)
    
    return cp['epoch']



def mysave_nets(job_name, nets, epoch):
    path = '{}/{}.pth'.format('models', job_name)

    if not os.path.exists('models'):
        print('Creating model directory: {}'.format('models'))
        os.makedirs('models')

    torch.save({
        'jsccq_model': nets.state_dict(),
        'epoch': epoch
    }, path)


def myload_weights(job_name, nets, args):
    path = '{}/{}.pth'.format('models', job_name)

    cp = torch.load(path)
    nets.load_state_dict(cp['jsccq_model'])
    
    return cp['epoch']

def calc_loss(prediction, target, loss):
    if loss == 'l2':
        loss = F.mse_loss(prediction, target)
    elif loss == 'msssim':
        loss = 1 - ms_ssim(prediction, target, win_size=3,
                           data_range=1, size_average=True)
    elif loss == 'ssim':
        loss = 1 - ssim(prediction, target,
                        data_range=1, size_average=True)
    else:
        raise NotImplementedError()
    return loss


def calc_psnr(predictions, targets):
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])
        compare = as_img_array(pred)
        val = psnr(original, compare)
        metric.append(val)
    return metric


def calc_msssim(predictions, targets):
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])
        compare = as_img_array(pred)
        # val = msssim(original, compare)
        val = ms_ssim(original, compare, data_range=255,
                      win_size=3, size_average=True)
        metric.append(val)
    return metric


def calc_ssim(predictions, targets):
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])
        compare = as_img_array(pred)
        val = ssim(original, compare, data_range=255,
                   size_average=True)
        metric.append(val)
    return metric



def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)
    im.save(filename)



def generateGrayarr(n):

    # base case
    if (n <= 0):
        return

    # 'arr' will store all generated codes
    arr = list()

    # start with one-bit pattern
    arr.append("0")
    arr.append("1")

    # Every iteration of this loop generates
    # 2*i codes from previously generated i codes.
    i = 2
    j = 0
    while(True):

        if i >= 1 << n:
            break

        # Enter the prviously generated codes
        # again in arr[] in reverse order.
        # Nor arr[] has double number of codes.
        for j in range(i - 1, -1, -1):
            arr.append(arr[j])

        # append 0 to the first half
        for j in range(i):
            arr[j] = "0" + arr[j]

        # append 1 to the second half
        for j in range(i, 2 * i):
            arr[j] = "1" + arr[j]
        i = i << 1

    for i in range(len(arr)):
        binary = [int(j) for j in arr[i]]
        arr[i] = binary
    return arr


def save_frames(frame, fns, out_dir):
    if not os.path.exists(out_dir):
        print('Creating output directory: {}'.format(out_dir))
        os.makedirs(out_dir)

    for idx, pred in enumerate(frame):
        pred = as_img_array(pred.cpu().numpy())
        pred = np.squeeze(pred, axis=0)
        flag = cv2.imwrite(out_dir + '/{}.png'.format(fns[idx][0]), pred)
        assert flag

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, args):

        ctx.save_for_backward(inputs)
        ctx.args = args

        x_lim_abs  = args.enc_value_limit
        x_lim_range = 2.0 * x_lim_abs
        x_input_norm =  torch.clamp(inputs, -x_lim_abs, x_lim_abs)

        if args.enc_quantize_level == 2:
            outputs_int = torch.sign(x_input_norm)
        else:
            outputs_int  = torch.round((x_input_norm +x_lim_abs) * ((args.enc_quantize_level - 1.0)/x_lim_range)) * x_lim_range/(args.enc_quantize_level - 1.0) - x_lim_abs

        return outputs_int

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.args.enc_clipping in ['inputs', 'both']:
            input, = ctx.saved_tensors
            grad_output[input>ctx.args.enc_value_limit]=0
            grad_output[input<-ctx.args.enc_value_limit]=0

        if ctx.args.enc_clipping in ['gradient', 'both']:
            grad_output = torch.clamp(grad_output, -ctx.args.enc_grad_limit, ctx.args.enc_grad_limit)

        if ctx.args.train_channel_mode not in ['group_norm_noisy', 'group_norm_noisy_quantize']:
            grad_input = grad_output.clone()
        else:
            # Experimental pass gradient noise to encoder.
            grad_noise = snr_db2sigma(ctx.args.fb_noise_snr) * torch.randn(grad_output[0].shape, dtype=torch.float)
            ave_temp   = grad_output.mean(dim=0) + grad_noise
            ave_grad   = torch.stack([ave_temp for _ in range(ctx.args.batch_size)], dim=2).permute(2,0,1)
            grad_input = ave_grad + grad_noise

        return grad_input, None