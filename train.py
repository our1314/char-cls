"""
1、损失函数
2、优化器
3、训练过程
    训练
    验证
"""
import argparse
import os
import time

import torch
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter
from torch import nn

from data import data_char
from models.classify_net1 import classify_net1


def train(opt):
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置一些训练可视化的参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_val_step = 0
    # 训练轮数
    epoch_count = 300
    net = classify_net1()
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    writer = SummaryWriter("logs")

    start_epoch = 0
    if opt.resume:
        '''
            断点继续参考：
            https://www.zhihu.com/question/482169025/answer/2081124014
            '''
        lists = os.listdir(opt.model_save_path)#获取模型路径下的模型文件
        if len(lists) > 0:
            lists.sort(key=lambda fn: os.path.getmtime(opt.model_save_path + "\\" + fn))  # 按时间排序
            last_pt_path = os.path.join(opt.model_save_path, lists[len(lists) - 1])
            checkpoint = torch.load(last_pt_path)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['net'])
            opt.load_state_dict(checkpoint['optimizer'])

    print(f"训练集的数量:{len(data_char.datasets_train)}")
    print(f"训练集的数量:{len(data_char.datasets_val)}")

    for epoch in range(start_epoch, epoch_count):
        print(f"----第{epoch + 1}轮训练开始----")

        # 训练
        net.train()
        total_train_accuracy = 0
        total_train_loss = 0
        for imgs, labels in data_char.dataloader_train:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # #region 显示训练图像
            # img1 = imgs[0, :, :, :]
            # img2 = imgs[1, :, :, :]
            # img1 = torchvision.transforms.ToPILImage()(img1)
            # img1.show()
            # img2 = torchvision.transforms.ToPILImage()(img2)
            # img2.show()
            # #endregion

            out = net(imgs)
            loss = loss_fn(out, labels)
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (out.argmax(1) == labels).sum()
            total_train_accuracy += acc
            total_train_loss += loss

        # 验证
        net.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        with torch.no_grad():
            for imgs, labels in data_char.dataloader_val:
                imgs = imgs.to(device)

                #region 显示训练图像
                # img1 = imgs[0, :, :, :]
                # img2 = imgs[1, :, :, :]
                # img1 = torchvision.transforms.ToPILImage()(img1)
                # img1.show()
                # img2 = torchvision.transforms.ToPILImage()(img2)
                # img2.show()
                #endregion

                labels = labels.to(device)
                out = net(imgs)

                loss = loss_fn(out, labels)
                total_val_loss += loss

                acc = (out.argmax(1) == labels).sum()
                total_val_accuracy += acc

        train_acc = total_train_accuracy / len(data_char.datasets_train)
        train_loss = total_train_loss
        val_acc = total_val_accuracy / len(data_char.datasets_val)
        val_loss = total_val_loss

        print(f"epoch:{epoch}, train_acc={train_acc}, train_loss={train_loss}, val_acc={val_acc}, val_loss={val_loss}")

        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)

        # 保存训练模型
        state_dict = {'net': net.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'epoch': epoch}

        p = f'{opt.model_save_path}/{time.strftime("%Y.%m.%d_%H.%M.%S")}-epoch={epoch}-loss={str(round(train_loss.item(), 5))}-acc={str(round(train_acc.item(), 5))}.pth'
        torch.save(state_dict, p)
        # print(f"第{epoch+1}轮模型参数已保存")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--model_save_path', default='run/train', help='save to project/name')

    opt = parser.parse_args()
    train(opt)

