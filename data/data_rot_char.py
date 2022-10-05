import os
import random

import cv2
import numpy as np
import torch
import torchvision.datasets
from torch.utils.data import Dataset
from PIL import Image


class data_rot_char(Dataset):
    def __init__(self, root_dir):
        """获取所有文件的相对路径"""
        self.root_dit = root_dir
        self.file_list = []
        # 遍历每个子文件夹索取文件
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                f = os.path.join(root, file)
                self.file_list.append(f)
                pass
        random.shuffle(self.file_list)
        pass

    def __getitem__(self, item):
        f = self.file_list[item]

        #region 1、获取图像
        img = Image.open(f)
        #endregion

        #region 2、获取字符分类标签
        a = os.path.dirname(f)
        _, dir = os.path.split(a)

        if ord('0') <= ord(dir) <= ord('9'):
            label_cls = ord(dir) - ord('0')
        elif ord('A') <= ord(dir) <= ord('Z'):
            label_cls = ord(dir) - ord('A') + 10
        else:#视为大写字母O
            label_cls = int(ord('O')) - int(ord('A')) + 10
        #endregion

        #region 3、随机旋转，获取旋转标签
        img_cv = np.array(img)
        isrot = np.random.randint(0, 2)
        if isrot is 1:
            cv2.rotate(img_cv, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE, dst=img_cv)
            label_rot = [1,0]


        #endregion
        label = label_rot + label_cls
        return img, label

    def __len__(self):
        l = len(self.file_list)
        return l


if __name__ == '__main__':
    #torchvision.datasets.CIFAR10



    aa = data_rot_char('D:/桌面/ocr')
    img, label = aa[0]

    print(label)
    print(img)
    pass
