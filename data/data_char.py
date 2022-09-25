"""
从文件夹读取数据
提供dataloader
数据增强
将数据加载到GPU

dataset 为数据集
dataloader 为数据集加载器
"""
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

trans = torchvision.transforms.Compose([
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.Pad([28, 10]),
    torchvision.transforms.Resize((300, 300)),
    torchvision.transforms.RandomRotation(degrees=10),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

datasets_all = ImageFolder('D:/桌面/ocr', transform=trans)
l = len(datasets_all)
datasets_train = torch.utils.data.Subset(datasets_all, range(int(0.9 * l)))
datasets_val = torch.utils.data.Subset(datasets_all, range(int(0.1 * l)))

dataloader_train = DataLoader(datasets_train, 2, shuffle=True)
dataloader_val = DataLoader(datasets_val, 1, shuffle=True)
