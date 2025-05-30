# -*- coding: utf-8 -*-
"""
@file name:util_dataset.py
@desc: 数据集 dataset
"""

import os
import torch
import torchvision
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------#
#   dataset
# ----------------------------------------------------#
# AutoEncoder任务，不需要labels
class COCO_dataset(Dataset):
    def __init__(self, images_path, transform=None, image_num=None):
        self.images_path = images_path  # 初始化图像文件夹
        self.transform = transform  # 初始化图像变换
        self.image_list = os.listdir(images_path)
        if image_num is not None:
            self.image_list = self.image_list[:image_num]

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.image_list)

    def __getitem__(self, index):
        # 用于加载并返回数据集中给定索引idx的样本
        image_path = os.path.join(self.images_path, self.image_list[index])
        image = read_image(image_path, mode=ImageReadMode.RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image


# ----------------------------------------------------#
#   transform
# ----------------------------------------------------#
def image_transform(resize=256, gray=False):
    if gray:
        transforms_list = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize(400),
                                              transforms.RandomCrop(resize),
                                              transforms.Grayscale(num_output_channels=1),
                                              transforms.ToTensor(),
                                              ])
    else:
        transforms_list = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize(400),
                                              transforms.RandomCrop(resize),
                                              transforms.ToTensor()
                                              ])
    return transforms_list

'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    image_path = 'F:/PycharmProjects/images_fusion/DenseFuse/fusion_test_data/Road/1'
    gray = False

    transform = image_transform(gray=gray)
    coco_dataset = COCO_dataset(images_path=image_path, transform=transform)
    print(coco_dataset.__len__())  # 118287

    image = coco_dataset.__getitem__(20)
    print(type(image))  # <class 'torch.Tensor'>
    print(image.shape)  # torch.Size([3, 256, 256])
    print(image.max())  # tensor(0.9961)
    print(image.min())  # tensor(0.)

    img_np = image.numpy()
    print(type(img_np))  # <class 'numpy.ndarray'>

    plt.axis("off")
    if gray:
        plt.imshow(np.transpose(img_np, (1, 2, 0)), cmap='gray')
    else:
        plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show()
