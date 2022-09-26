import torch.onnx.utils

from models.classify_net1 import classify_net1

path = 'run/train/epoch=51-train_acc=1.0.pth'
f = path.replace('.pth', '.onnx')

x = torch.randn(1, 3, 300, 300)
checkpoint = torch.load(path)

net = classify_net1()
net.load_state_dict(checkpoint['net'])
net.eval()
torch.onnx.export(net, x, f, input_names=['input'], output_names=['output'], verbose='True')
print('export success!')
