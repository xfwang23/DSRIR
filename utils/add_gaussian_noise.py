import cv2
import numpy as np
import random
from PIL import Image
# 读取图像


# 添加高斯噪声
def _add_gaussian_noise(img):
    img = np.array(img)
    sigma = 25 # random.randint(0, 55)
    noise = np.random.randn(*img.shape)
    noisy_patch = np.clip(img + noise * sigma, 0, 255).astype(np.uint8)
    noisy_patch = Image.fromarray(noisy_patch)
    noisy_patch.save('../images/noise.png')


# 保存添加噪声后的图像
image = Image.open('../images/223061.png').convert('RGB')
_add_gaussian_noise(image)

