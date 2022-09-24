"""
1、损失函数
2、优化器
3、训练过程
    训练
    验证
"""
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch import nn

from data import data_char
from models.classify_net1 import classify_net1

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
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

writer = SummaryWriter("../logs015_train")

print(f"训练集的数量:{len(data_char.datasets_train)}")
print(f"训练集的数量:{len(data_char.datasets_val)}")

for epoch in range(epoch_count):
    print(f"----第{epoch + 1}轮训练开始----")

    # 训练
    net.train()
    for imgs, labels in data_char.dataloader_train:
        imgs = imgs.to(device)
        labels = labels.to(device)

        out = net(imgs)
        loss = loss_fn(out, labels)
        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

        print(f"整体测试集上的Loss:{total_val_loss}")
        writer.add_scalar("test_loss", total_val_loss, total_val_step)

        print(total_val_accuracy, len(data_char.datasets_val))
        print(f"整体测试集上的Accuracy:{total_val_accuracy / len(data_char.datasets_val)}")
        writer.add_scalar("test_accuracy", total_val_accuracy / len(data_char.datasets_val), epoch+1)# 计算精度需要除验证数据集的长度，而不是其dataloader(数量不对)

        torch.save(net, f"epoch_{epoch + 1}.pth")
        print(f"第{epoch + 1}轮模型参数已保存")
