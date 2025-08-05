import os
import torch
from PIL import Image
from model.DSRIR import DSRIR
import cv2
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms.functional as tf
from test import crop_img


def draw_features(x, savename):
    img = x[0, 0, :, :]
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    img = img.astype(np.uint8)  # 转成unit8
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
    cv2.imwrite(savename, img)


def convert_vars_(var):
    convert = var.new(1, 3, 1, 1)
    convert[0, 0, 0, 0] = 65.738
    convert[0, 1, 0, 0] = 129.057
    convert[0, 2, 0, 0] = 25.064
    var.mul_(convert).div_(256)
    var = var.sum(dim=1, keepdim=True)

    return var


def inference(model, image_path, save_path, save_results=True):
    image_save_dir = save_path + '/' + 'restored'
    var_save_dir = save_path + '/' + 'vars'
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
        os.makedirs(var_save_dir)
    img = crop_img(np.array(Image.open(image_path).convert('RGB')), base=8)
    img = tf.to_tensor(img).unsqueeze(0)
    img = img.cuda()
    model.eval()
    with torch.no_grad():
        restored, vars_ = model(img)
        if save_results:
            for i in range(len(restored)):
                save_image(restored[i], os.path.join(image_save_dir, 'restored_' + str(i) + '.png'))
                draw_features(convert_vars_(vars_[i]).cpu().numpy(), os.path.join(var_save_dir, 'vars_' + str(i) + '.png'))

        else:
            pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = DSRIR().cuda()
    checkpoint = torch.load('./experiments/all_in_one/models/DSRIR/model_best.pth')
    model.load_state_dict(checkpoint['state_dict'])
    inference(model, image_path='images/0148_0.9_0.12.jpg', save_path='experiments/inference', save_results=True)

