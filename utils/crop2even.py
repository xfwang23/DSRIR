import numpy as np
from PIL import Image
import os


def crop_img(image, base=8):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]

# def crop_to_even(image_path):
#     # 打开图像文件
#     image = Image.open(image_path)
#
#     # 获取原始图像的宽度和高度
#     width, height = image.size
#
#     # 计算裁剪后的新宽度和新高度（确保为偶数）
#     new_width = width if width % 2 == 0 else width - 1
#     new_height = height if height % 2 == 0 else height - 1
#
#     # 裁剪图像
#     cropped_image = image.crop((0, 0, new_width, new_height))
#
#     # 覆盖保存裁剪后的图像
#     cropped_image.save(image_path)
#
#     # 关闭图像文件
#     image.close()


# 遍历文件夹下所有文件
folder_path = r"../images"
for file_name in os.listdir(folder_path):
    # 判断是否为图像文件

    # 构造图像文件路径
    image_path = os.path.join(folder_path, file_name)
    image = np.array(Image.open(image_path).convert('RGB'))

    # 进行裁剪并覆盖保存
    cropped = Image.fromarray(crop_img(image, base=8))
    cropped.save(image_path)
