import os
import numpy as np
import torch
import cv2
from time import time
from dataloader_HMIF import TestData
import torch.utils.data as data
import torchvision.transforms as transforms
import argparse
from args_setting import args
from natsort import natsorted

from model import W_HSSF

DEVICE = args.DEVICE
EPS = 1e-8

def Mytest(model_test=None, img_save_dir=None):
    os.makedirs('./result/' + args.model + '/' + args.task, exist_ok=True)
    if model_test is None:
        model_dir = './modelsave/' + args.model + '/' + args.task + '/'
        model_path_final = './3_ASFEFusion.pth'
    else:
        model_path_final = model_test

    if img_save_dir is None:
        img_save_dir = './result/' + args.model + '/' + args.task
    else:
        img_save_dir = img_save_dir

    os.makedirs(img_save_dir, exist_ok=True)

    net = W_HSSF()
    net.eval()
    net = net.to(DEVICE)
    net.load_state_dict(torch.load(model_path_final, map_location=args.DEVICE))

    transform = transforms.Compose([transforms.ToTensor()])
    test_set = TestData(transform)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False,
                                  num_workers=1, pin_memory=False)
    with torch.no_grad():
        if args.task == 'CT-MRI':
            for batch, [img_name, img1, img2] in enumerate(test_loader):  # CT-MRI Fusion
                print("test for image %s" % img_name[0])
                img1 = img1.to(DEVICE)
                img2 = img2.to(DEVICE)
                fused_img = net(img1, img2)
                fused_img = (fused_img - fused_img.min()) / (fused_img.max() - fused_img.min()) * 255.
                fused_img = fused_img.cpu().numpy().squeeze()
                cv2.imwrite('%s/%s' % (img_save_dir, img_name[0]), fused_img)
        else:
            for batch, [img_name, img1_Y, img2, img1_CrCb] in enumerate(test_loader):  # PET/SPECT-MRI Fusion
                print("test for image %s" % img_name[0])

                img1_Y = img1_Y.to(DEVICE)
                img2 = img2.to(DEVICE)
                print(img1_Y)
                fused_img_Y = net(img1_Y, img2)

                fused_img_Y = (fused_img_Y - fused_img_Y.min()) / (fused_img_Y.max() - fused_img_Y.min()) * 255.

                fused_Y_np = fused_img_Y.squeeze().cpu().numpy().astype(np.uint8)  # 去除批次和通道维度

                # 3. 直接保存Y通道灰度图（文件名添加_Y后缀）
                cv2.imwrite(os.path.join('./fusion_Y', f"{img_name[0]}_Y.png"), fused_Y_np)
                fused_img_Y = fused_img_Y.cpu().numpy()

                fused_img = np.concatenate((fused_img_Y, img1_CrCb), axis=1).squeeze()
                fused_img = np.transpose(fused_img, (1, 2, 0))
                fused_img = fused_img.astype(np.uint8)
                fused_img = cv2.cvtColor(fused_img, cv2.COLOR_YCrCb2BGR)

                cv2.imwrite('%s/%s' % (img_save_dir, img_name[0]), fused_img)

    print('test results in ./%s/' % img_save_dir)
    print('Finish!')


if __name__ == '__main__':
    Mytest()
