import argparse
import random
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
import numpy as np
import os
import torch
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils.image_io import save_image_tensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model.DSRIR import DSRIR
from utils.matlab_functions import rgb2ycbcr


def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_psnr_ssim(recoverd, clean, y_channel=False):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):

        cl = rgb2ycbcr(clean[i], y_only=True) if y_channel else clean[i]
        re = rgb2ycbcr(recoverd[i], y_only=True) if y_channel else recoverd[i]

        psnr += peak_signal_noise_ratio(cl, re, data_range=1)
        ssim += structural_similarity(cl, re, data_range=1, channel_axis=-1)

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=8)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img

    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain"):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()

        self.set_dataset(task)

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=8)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=8)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


def Denoise(opt, net, dataset, sigma=15, save_img=False):
    output_path = opt.output_path + 'denoise/' + str(sigma) + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=6)

    psnr = AverageMeter()
    ssim = AverageMeter()
    time_list = []

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            t1 = time.time()
            restored, _ = net(degrad_patch)
            t2 = time.time()
            time_list.append(t2 - t1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored[0], clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            if save_img:
                save_image_tensor(restored[0], output_path + clean_name[0] + '.png')

        print('======================================================================')
        print("Deonise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
        print(f"denoising average time: {sum(time_list) / len(time_list):.4f}")
        print('======================================================================')
    return psnr.avg, ssim.avg


def Derain_Dehaze(opt, net, dataset, task="derain", save_img=False):
    output_path = opt.output_path + task + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=6)

    psnr = AverageMeter()
    ssim = AverageMeter()
    time_list = []

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            t1 = time.time()
            restored, _ = net(degrad_patch)
            t2 = time.time()
            time_list.append(t2 - t1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored[0], clean_patch, y_channel=True if task == 'derain' else False)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            if save_img:
                save_image_tensor(restored[0], output_path + degraded_name[0] + '.png')

        print('======================================================================')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        print(f"{task} average time: {sum(time_list) / len(time_list):.4f}")
        print('======================================================================')
    return psnr.avg, ssim.avg


def validation(net, opt, epoch, save_img, write_results):

    denoise_set = DenoiseTestDataset(opt)
    derain_set = DerainDehazeDataset(opt)
    net.eval()
    p15, p25, p50, prain, phaze, p_avg, s_avg = 0, 0, 0, 0, 0, 0, 0
    s15, s25, s50, srain, shaze = 0, 0, 0, 0, 0

    if opt.mode == 'noise' or opt.mode == 'all_in_one':
        print('Start testing Sigma=15...')
        p15, s15 = Denoise(opt, net, denoise_set, sigma=15, save_img=save_img)

        print('Start testing Sigma=25...')
        p25, s25 = Denoise(opt, net, denoise_set, sigma=25, save_img=save_img)

        print('Start testing Sigma=50...')
        p50, s50 = Denoise(opt, net, denoise_set, sigma=50, save_img=save_img)

    if opt.mode == 'rain' or opt.mode == 'all_in_one':
        print('Start testing rain streak removal...')
        prain, srain = Derain_Dehaze(opt, net, derain_set, task="derain", save_img=save_img)

    if opt.mode == 'haze' or opt.mode == 'all_in_one':
        print('Start testing SOTS...')
        phaze, shaze = Derain_Dehaze(opt, net, derain_set, task="dehaze", save_img=save_img)

    if opt.mode == 'all_in_one':
        p_avg = (p15 + p25 + p50 + prain + phaze) / 5
        s_avg = (s15 + s25 + s50 + srain + shaze) / 5
    elif opt.mode == 'noise':
        p_avg = (p15 + p25 + p50) / 3
        s_avg = (s15 + s25 + s50) / 3
    elif opt.mode == 'rain':
        p_avg = prain
        s_avg = srain
    elif opt.mode == 'haze':
        p_avg = phaze
        s_avg = shaze

    if write_results:
        stat_dir = f'experiments/{opt.mode}/'
        stat_path = os.path.join(stat_dir, f'train_{opt.mode}.txt')
        with open(stat_path, 'a') as f:
            f.write(f'Epoch:{epoch}  '
                    f'Denoising:{p15:.2f}/{s15:.4f}, {p25:.2f}/{s25:.4f}, {p50:.2f}/{s50:.4f}  '
                    f'Deraining:{prain:.2f}/{srain:.4f}  '
                    f'Dehazing:{phaze:.2f}/{shaze:.4f}\t'
                    f'Average: {p_avg:.2f}/{s_avg:.4f}\n')
        f.close()

    return p_avg, s_avg


if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--mode', type=str, default='haze')  # all_in_one, noise, rain, haze
    parser.add_argument('--device', type=int, default=0)
    # /home/kiki/B
    # parser.add_argument('--denoise_path', type=str, default="/home/sht/all_in_one_dataset/BSD68/", help='save path of test noisy images')
    # parser.add_argument('--derain_path', type=str, default="/home/sht/all_in_one_dataset/Rain100L/test/", help='save path of test raining images')
    # parser.add_argument('--dehaze_path', type=str, default="/home/sht/all_in_one_dataset/SOTS/outdoor/", help='save path of test hazy images')
    parser.add_argument('--denoise_path', type=str, default=r"D:\WorkofMaster\GraduationThesis\Experiments\Chapter2\all_in_one_dataset/Urban100/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default=r"D:\WorkofMaster\GraduationThesis\Experiments\Chapter2\all_in_one_dataset/Rain100L/test/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default=r"D:\WorkofMaster\GraduationThesis\Experiments\Chapter2\all_in_one_dataset/SOTS/outdoor/", help='save path of test hazy images')

    opts = parser.parse_args()
    opts.output_path = r'experiments/' + opts.mode + '/results/DSRIR/'

    torch.cuda.set_device(opts.device)
    model = DSRIR().cuda()
    state_dict = torch.load(f'./experiments/{opts.mode}/models/DSRIR/model_best.pth', map_location=torch.device(opts.device))
    model.load_state_dict(state_dict['state_dict'])
    p_avg = validation(model, opt=opts, epoch=0, save_img=False, write_results=False)
    print(f'Average PSNR/SSIM is {p_avg[0]:.2f}/{p_avg[1]:.4f}.')
