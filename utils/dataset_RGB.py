import os
import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as tf
import numpy as np
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, mode='all_in_one', img_options=None):
        super(DataLoaderTrain, self).__init__()
        self.rgb_dir = rgb_dir
        self.csv_file_path = f'./datasets/train/train_{mode.lower()}.csv'
        self.image_paths = pd.read_csv(self.csv_file_path)
        self.img_options = img_options

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        ps = self.ps
        tar_path, inp_path = self.image_paths.iloc[index, :]
        if tar_path.startswith("BSD400") or tar_path.startswith("WED"):
            tar_img = Image.open(os.path.join(self.rgb_dir, tar_path)).convert('RGB')
            inp_img = self._add_gaussian_noise(tar_img)
        else:
            tar_img = Image.open(os.path.join(self.rgb_dir, tar_path)).convert('RGB')
            inp_img = Image.open(os.path.join(self.rgb_dir, inp_path)).convert('RGB')

        w, h = tar_img.size
        if w < ps:
            inp_img.resize((ps, h))
            tar_img.resize((ps, h))
        if h < ps:
            inp_img.resize((w, ps))
            tar_img.resize((w, ps))
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        inp_img = tf.to_tensor(inp_img)
        tar_img = tf.to_tensor(tar_img)
        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = tf.pad(inp_img, [0, 0, padw, padh], padding_mode='reflect')
            tar_img = tf.pad(tar_img, [0, 0, padw, padh], padding_mode='reflect')

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(tar_path)[0]

        return tar_img, inp_img, filename

    def _add_gaussian_noise(self, img):
        img = np.array(img)
        sigma = random.randint(0, 55)
        # sigma = random.choice([15, 25, 50])
        noise = np.random.randn(*img.shape)
        noisy_patch = np.clip(img + noise * sigma, 0, 255).astype(np.uint8)
        noisy_patch = Image.fromarray(noisy_patch)
        # noisy_patch.save('datasets/noise.jpg')
        return noisy_patch


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        # Validate on center crop
        if self.ps is not None:
            inp_img = tf.center_crop(inp_img, (ps, ps))
            tar_img = tf.center_crop(tar_img, (ps, ps))

        inp_img = tf.to_tensor(inp_img)
        tar_img = tf.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = tf.to_tensor(inp)
        return inp, filename


def get_training_data(rgb_dir, mode, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, mode=mode, img_options=img_options)


def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)


def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)