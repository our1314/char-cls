"""
2022.9.26
从文件夹读取数据
提供dataloader
数据增强
将数据加载到GPU

dataset 为数据集
dataloader 为数据集加载器
"""
import os
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as f


class SquarePad:
    def __call__(self, image):
        # image = torchvision.transforms.Pad()
        w, h = image.size
        max_wh = max(w, h)

        left = int((max_wh - w) / 2)
        right = max_wh - w - left
        top = int((max_wh - h) / 2)
        bottom = max_wh - h - top

        padding = [left, top, right, bottom]
        image = torchvision.transforms.Pad(padding=padding, fill=0)(image)  # left, top, right and bottom
        return image


trans_train = torchvision.transforms.Compose([
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomVerticalFlip(),
    SquarePad(),
    torchvision.transforms.Resize(200),
    torchvision.transforms.RandomAffine(degrees=10, scale=[0.7, 1.0]),
    torchvision.transforms.RandomGrayscale(p=0.3),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trans_val = torchvision.transforms.Compose([
    SquarePad(),
    torchvision.transforms.Resize(200),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

datasets_train = ImageFolder('C:/Users/pc/Desktop/ocr', transform=trans_train)
datasets_val = ImageFolder('C:/Users/pc/Desktop/ocr', transform=trans_val)

dataloader_train = DataLoader(datasets_train, 10, shuffle=True)
dataloader_val = DataLoader(datasets_val, 4, shuffle=True)

if __name__ == '__main__':
    dataloader_test = DataLoader(datasets_train, batch_size=1, shuffle=True)
    for imgs, labels in dataloader_test:
        img1 = imgs[0, :, :, :]
        img1 = torchvision.transforms.ToPILImage()(img1)
        img1.show()
