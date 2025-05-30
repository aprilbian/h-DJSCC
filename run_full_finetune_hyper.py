import random
import numpy as np 
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS

from get_args import get_args
from modules import *
from dataset import CIFAR10, ImageNet, Kodak
from utils import *
from multihop_full_adapt import *

from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

###### Parameter Setting
args = get_args()
args.device = device

lamdas = [200, 400, 800, 1600, 3200]
#lamdas = [100, 400, 1600, 6400]
args.lamdas = lamdas

args.snr_min = 1
args.snr_max = 9

job_name = 'JSCC_multihop_full_adaptive_freeze_hyper_large' + '_dataset_' + str(args.dataset) + '_cout_' + str(args.cout) + '_fading_' + str(args.fading)
if args.adapt:
    job_name = job_name + '_lambda_levels_' + str(len(args.lamdas)) + '_start_' + str(args.lamdas[0]) + '_snr_min_' + str(args.snr_min) + '_snr_max_' + str(args.snr_max)


print(args)
print(job_name)

###### The JSCC Model
jscc_model = Adhoc(args)

writter = SummaryWriter('runs/' + job_name)

# select different datasets

if args.dataset == 'cifar':
    train_set = CIFAR10('datasets/cifar-10-batches-py', 'TRAIN')
    valid_set = CIFAR10('datasets/cifar-10-batches-py', 'VALIDATE')
    eval_set = CIFAR10('datasets/cifar-10-batches-py', 'EVALUATE')
else:
    train_set = torchvision.datasets.CelebA(
                root="datasets/", split='train', download=True,
                transform=transforms.Compose([transforms.ToTensor(), torchvision.transforms.Resize((128, 128), antialias=True)]))
    valid_set = torchvision.datasets.CelebA(
                root="datasets/", split='valid', download=True,
                transform=transforms.Compose([transforms.ToTensor(), torchvision.transforms.Resize((128, 128), antialias=True)]))
    eval_set = torchvision.datasets.CelebA(
                root="datasets/", split='test', download=True,
                transform=transforms.Compose([transforms.ToTensor(), torchvision.transforms.Resize((128, 128), antialias=True)]))

# load the weights
if args.dataset == 'cifar':
    if args.fading:
        load_name = 'JSCC_Adhoc_AF_precode_num_hops_3_dataset_cifar_cout_24_link_qual_10_fading_True_is_adapt_True_link_rng_5'
    else:
        load_name = 'JSCC_Adhoc_AF_num_hops_3_dataset_cifar_cout_24_link_qual_5_is_adapt_True_link_rng_4'
else:
    if args.fading:
        load_name = 'JSCC_dataset_celeba_cout_6_fading_True_is_adapt_True_snr_min_5_snr_max_15'
    else:
        load_name = 'JSCC_dataset_celeba_cout_6_fading_False_is_adapt_True_snr_min_1_snr_max_9'

_ = load_weights(load_name, jscc_model, strict=False)


for p in jscc_model.enc.parameters():
    p.requires_grad = False

for p in jscc_model.dec.parameters():
    p.requires_grad = False
''''''

# load pre-trained
if args.resume == False:
    pass
else:
    _ = load_weights(job_name, jscc_model)

solver = optim.Adam(jscc_model.parameters(), lr=args.lr)
scheduler = LS.MultiplicativeLR(solver, lr_lambda=lambda x: 0.9)
es = EarlyStopping(mode='min', min_delta=0, patience=args.train_patience)

###### Dataloader
train_loader = data.DataLoader(
    dataset=train_set,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=2
        )

valid_loader = data.DataLoader(
    dataset=valid_set,
    batch_size=args.val_batch_size,
    shuffle=True,
    num_workers=2
        )

eval_loader = data.DataLoader(
    dataset=eval_set,
    batch_size=args.val_batch_size,
    shuffle=False,
    num_workers=2
)


def train_epoch(loader, model, solvers):

    model.train()

    with tqdm(loader, unit='batch') as tepoch:
        for _, (images, _) in enumerate(tepoch):
            
            epoch_postfix = OrderedDict()

            images = images.to(args.device).float()
            
            solvers.zero_grad()

            s1 = random.randint(0, len(args.lamdas) - 1)
            snr = args.snr_min + random.random()*(args.snr_max - args.snr_min)
            x_hat, est_bits = model(images, torch.tensor([[snr]]).to(args.device), s1)

            loss = args.lamdas[s1]*nn.MSELoss()(x_hat, images) + est_bits
            loss.backward()
            solvers.step()

            epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())
            epoch_postfix['nbits'] = '{:.4f}'.format(est_bits.item())

            tepoch.set_postfix(**epoch_postfix)


def validate_epoch(loader, model, snr, s, disable = False):

    model.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []
    nbit_hist = []

    # list of processing times for different nodes
    s_time = []
    r_time = []
    d_time = []

    with torch.no_grad():
        with tqdm(loader, unit='batch', disable=disable) as tepoch:
            for _, (images, _) in enumerate(tepoch):

                epoch_postfix = OrderedDict()

                images = images.to(args.device).float()

                output, est_bits, times = model(images, torch.tensor([[snr]]).to(args.device), s)
                loss = args.lamdas[s]*nn.MSELoss()(output, images) + est_bits

                epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())
                epoch_postfix['nbits'] = '{:.4f}'.format(est_bits.item())

                ######  Predictions  ######
                predictions = torch.chunk(output, chunks=output.size(0), dim=0)
                target = torch.chunk(images, chunks=images.size(0), dim=0)

                ######  PSNR/SSIM/etc  ######

                psnr_vals = calc_psnr(predictions, target)
                psnr_hist.extend(psnr_vals)
                epoch_postfix['psnr'] = torch.mean(torch.tensor(psnr_vals)).item()

                ssim_vals = calc_ssim(predictions, target)
                ssim_hist.extend(ssim_vals)
                epoch_postfix['ssim'] = torch.mean(torch.tensor(ssim_vals)).item()
                
                # Show the snr/loss/psnr/ssim
                tepoch.set_postfix(**epoch_postfix)

                loss_hist.append(loss.item())
                nbit_hist.append(est_bits.item())
            
            loss_mean = np.nanmean(loss_hist)
            nbit_mean = np.nanmean(nbit_hist)

            psnr_hist = torch.tensor(psnr_hist)
            psnr_mean = torch.mean(psnr_hist).item()
            psnr_std = torch.sqrt(torch.var(psnr_hist)).item()

            ssim_hist = torch.tensor(ssim_hist)
            ssim_mean = torch.mean(ssim_hist).item()
            ssim_std = torch.sqrt(torch.var(ssim_hist)).item()

            predictions = torch.cat(predictions, dim=0)[:, [2, 1, 0]]
            target = torch.cat(target, dim=0)[:, [2, 1, 0]]

            return_aux = {'psnr': psnr_mean,
                            'ssim': ssim_mean,
                            'predictions': predictions,
                            'target': target,
                            'psnr_std': psnr_std,
                            'ssim_std': ssim_std,
                            'nbit_mean':nbit_mean}

        
    return loss_mean, return_aux



def save_images_epoch(loader, model, snr, s, disable = False):

    model.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []
    nbit_hist = []

    with torch.no_grad():
        with tqdm(loader, unit='batch', disable=disable) as tepoch:
            for i, (images, _) in enumerate(tepoch):

                epoch_postfix = OrderedDict()

                images = images.to(args.device).float()

                output, est_bits = model(images, torch.tensor([[snr]]).to(args.device), s)
                loss = args.lamdas[s]*nn.MSELoss()(output, images) + est_bits

                epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())
                epoch_postfix['nbits'] = '{:.4f}'.format(est_bits.item())

                ######  Predictions  ######
                predictions = torch.chunk(output, chunks=output.size(0), dim=0)
                target = torch.chunk(images, chunks=images.size(0), dim=0)

                # By default, we have batch_size = 1
                #save_image_tensor2pillow(output, 'Full_Adapt_JSC/' + str(182638 + i)  + '.png')
                #save_image_tensor2pillow(images, 'Target/' + str(182638 + i) + '.png')

                ######  PSNR/SSIM/etc  ######

                psnr_vals = calc_psnr(predictions, target)
                psnr_hist.extend(psnr_vals)
                epoch_postfix['psnr'] = torch.mean(torch.tensor(psnr_vals)).item()

                ssim_vals = calc_ssim(predictions, target)
                ssim_hist.extend(ssim_vals)
                epoch_postfix['ssim'] = torch.mean(torch.tensor(ssim_vals)).item()
                
                # Show the snr/loss/psnr/ssim
                tepoch.set_postfix(**epoch_postfix)

                loss_hist.append(loss.item())
                nbit_hist.append(est_bits.item())
            
            loss_mean = np.nanmean(loss_hist)
            nbit_mean = np.nanmean(nbit_hist)

            psnr_hist_list = [psnr_hist[i].cpu().numpy() for i in range(len(psnr_hist))]
            ssim_hist_list = [ssim_hist[i].cpu().numpy() for i in range(len(ssim_hist))]
            #bpp_hist_list = [nbit_hist[i].cpu().numpy() for i in range(len(nbit_hist))]

            np.savez('full_adapt_jsc.npz', psnr = np.array(psnr_hist_list), ssim = np.array(ssim_hist_list), bpp = np.array(nbit_hist))
            
            psnr_hist = torch.tensor(psnr_hist)
            psnr_mean = torch.mean(psnr_hist).item()
            psnr_std = torch.sqrt(torch.var(psnr_hist)).item()

            ssim_hist = torch.tensor(ssim_hist)
            ssim_mean = torch.mean(ssim_hist).item()
            ssim_std = torch.sqrt(torch.var(ssim_hist)).item()

            predictions = torch.cat(predictions, dim=0)[:, [2, 1, 0]]
            target = torch.cat(target, dim=0)[:, [2, 1, 0]]

            return_aux = {'psnr': psnr_mean,
                            'ssim': ssim_mean,
                            'predictions': predictions,
                            'target': target,
                            'psnr_std': psnr_std,
                            'ssim_std': ssim_std,
                            'nbit_mean':nbit_mean}

        
    return loss_mean, return_aux





if __name__ == '__main__':
    epoch = 0

    while epoch < args.epoch and not args.resume:
        
        epoch += 1
        
        train_epoch(train_loader, jscc_model, solver)

        valid_loss, valid_aux = validate_epoch(valid_loader, jscc_model, snr = 5, s = 2)

        writter.add_scalar('loss', valid_loss, epoch)
        writter.add_scalar('psnr', valid_aux['psnr'], epoch)



        flag, best, best_epoch, bad_epochs = es.step(torch.Tensor([valid_loss]), epoch)
        if flag:
            print('ES criterion met; loading best weights from epoch {}'.format(best_epoch))
            _ = load_weights(job_name, jscc_model)
            break
        else:
            # TODO put this in trainer
            if bad_epochs == 0:
                print('average l2_loss: ', valid_loss.item())
                print('avg bits: ', valid_aux['nbit_mean'])
                save_nets(job_name, jscc_model, epoch)
                best_epoch = epoch
                print('saving best net weights...')
            elif bad_epochs % (es.patience//3) == 0:
                scheduler.step()
                print('lr updated: {:.5f}'.format(scheduler.get_last_lr()[0]))


    print('evaluating...')
    print(job_name)
    ####### adjust the num of hops; and link power
    
    #for s in [5, 10, 15]:
    for s in [1, 5, 9]:
        for s1 in range(0,len(args.lamdas)):
            _, eval_aux = validate_epoch(eval_loader, jscc_model, s, s1)
            print(eval_aux['psnr'])
            print(eval_aux['ssim'])
            print(eval_aux['nbit_mean'])
    '''
    

    _, eval_aux = save_images_epoch(eval_loader, jscc_model, 5, 2)
    print(eval_aux['psnr'])
    print(eval_aux['ssim'])
    print(eval_aux['nbit_mean'])
    '''