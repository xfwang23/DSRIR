import argparse
import os
import torch
import torch.nn as nn
from test import validation
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import utils
from utils.dataset_RGB import get_training_data
from model.DSRIR import DSRIR
from losses.loss import PSNRLoss, PositiveContrastLoss, UncertaintyLoss
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='1', help='gpu ids for training,')  # 0; 0,1
    # Model
    # mode: all_in_one, noise, rain, haze
    parser.add_argument('--mode', type=str, default='all_in_one', help='training tasks, noise, rain, haze, all_in_one')
    parser.add_argument('--session', type=str, default='DSRIR', help='name of training model')
    # Optim
    parser.add_argument('--batch-size', nargs='+', default=[16, 14, 12, 8], help='batch_size')
    parser.add_argument('--epochs', type=int, default=347, help='epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='min learning rate for training')
    # Training
    parser.add_argument('--val-epochs', type=int, default=5, help='epochs gap for validation')
    parser.add_argument('--resume', action='store_true', help='whether to train from pretrained model')
    # [128, 160, 192, 256]
    parser.add_argument('--patch-size', nargs='+', default=[128, 160, 192, 256], type=int, help='training patch size')
    parser.add_argument('--epoch-chg', nargs='+', default=[135, 238, 314, 347], type=int, help='patch size update at this epoch')
    parser.add_argument('--train-dir', type=str, default='/home/sht/wxf/all_in_one_dataset', help='directory path to training dataset')
    parser.add_argument('--save-dir', type=str, default='./experiments', help='path to save models and images')
    # Testing
    options = parser.parse_args()
    options.output_dir = r'experiments/' + options.mode
    options.output_path = options.output_dir + '/results/DSRIR/'
    options.denoise_path = os.path.join(options.train_dir, 'BSD68/')
    options.derain_path = os.path.join(options.train_dir, 'Rain100L/test/')
    options.dehaze_path = os.path.join(options.train_dir, 'RESIDE_beta/SOTS/outdoor/')

    return options


opt = parser_arguments()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

torch.backends.cudnn.benchmark = True

# ######## Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.mode
session = opt.session

result_dir = os.path.join(opt.save_dir, mode, 'results', session)
model_dir = os.path.join(opt.save_dir, mode, 'models', session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.train_dir

# ######## Model ###########
model_restoration = DSRIR().cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

new_lr = opt.learning_rate

optimizer = optim.AdamW(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), weight_decay=1e-4)

# ######## Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs - warmup_epochs,
                                                        eta_min=opt.min_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

# ######## DataLoaders ###########
dataset_128 = get_training_data(train_dir, mode=opt.mode, img_options={'patch_size': opt.patch_size[0]})
dataset_160 = get_training_data(train_dir, mode=opt.mode, img_options={'patch_size': opt.patch_size[1]})
dataset_192 = get_training_data(train_dir, mode=opt.mode, img_options={'patch_size': opt.patch_size[2]})
dataset_256 = get_training_data(train_dir, mode=opt.mode, img_options={'patch_size': opt.patch_size[3]})
loader_128 = DataLoader(dataset=dataset_128, batch_size=opt.batch_size[0], shuffle=True, num_workers=6, drop_last=True, pin_memory=True)
loader_160 = DataLoader(dataset=dataset_160, batch_size=opt.batch_size[1], shuffle=True, num_workers=6, drop_last=True, pin_memory=True)
loader_192 = DataLoader(dataset=dataset_192, batch_size=opt.batch_size[2], shuffle=True, num_workers=6, drop_last=True, pin_memory=True)
loader_256 = DataLoader(dataset=dataset_256, batch_size=opt.batch_size[3], shuffle=True, num_workers=6, drop_last=True, pin_memory=True)
loader = None
# ######## Resume ###########
if opt.resume:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)
    # select a loader
    if 1 <= start_epoch <= opt.epoch_chg[0]:
        loader = loader_128
    elif opt.epoch_chg[0] + 1 <= start_epoch <= opt.epoch_chg[1]:
        loader = loader_160
    elif opt.epoch_chg[1] + 1 <= start_epoch <= opt.epoch_chg[2]:
        loader = loader_192
    elif opt.epoch_chg[2] + 1 <= start_epoch <= opt.epoch_chg[3]:
        loader = loader_256

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

# ######## Loss ###########
criterion_psnr = PSNRLoss().cuda()
criterion_ul = UncertaintyLoss().cuda()
criterion_pocl = PositiveContrastLoss().cuda()

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.epochs))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0
last_epoch_loss = 0
p_avg = 0
last_p_avg = 0


for epoch in range(start_epoch, opt.epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    model_restoration.train()
    if epoch == 1:  # patch=128, [1, 135]
        loader = loader_128
    elif epoch == opt.epoch_chg[0] + 1:  # patch=160, [136, 238]
        loader = loader_160
    elif epoch == opt.epoch_chg[1] + 1:  # patch=192, [239, 314]
        loader = loader_192
    elif epoch == opt.epoch_chg[2] + 1:  # patch=256, [315, 347]
        loader = loader_256
    tbar = tqdm(loader)
    for data in tbar:
        target, input_ = data[0].cuda(), data[1].cuda()
        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        optimizer.zero_grad()

        restored, vars_ = model_restoration(input_)

        loss_ul = (criterion_ul(target, restored[0], vars_[0]) + criterion_ul(target, restored[1], vars_[1]) + criterion_ul(target, restored[2], vars_[2])) / 3
        loss_psnr = criterion_psnr(target, restored[0]) + 0.5 * (criterion_psnr(target, restored[1]) + criterion_psnr(target, restored[2])) / 2
        loss_pocl = criterion_pocl(restored[0], target, restored[1], restored[2], input_)
        loss = loss_ul + loss_psnr + 0.2 * loss_pocl

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        tbar.set_description(
            f'epoch:{epoch}/{opt.epochs} | '
            f'loss_ul:{loss_ul:.4f} | '
            f'loss_psnr:{loss_psnr:.4f} | '
            f'loss_pocl:{loss_pocl:.4f} | '
            f'lr:{scheduler.get_lr()[0]:.6}'
        )

    scheduler.step()

    avg_epoch_loss = epoch_loss / len(loader)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Epoch: {}/{}\tTime: {:.4f}\t==>> Loss: {:.4f} <<==\t LearningRate {:.6f}".format(epoch, opt.epochs,
          time.time() - epoch_start_time, avg_epoch_loss, scheduler.get_lr()[0]))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    if epoch % opt.val_epochs == 0:
        p_avg, _ = validation(model_restoration, opt=opt, epoch=epoch, save_img=False, write_results=True)
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_{epoch}.pth"))

        if p_avg > last_p_avg:
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))
            last_p_avg = p_avg

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))
