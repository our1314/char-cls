import torch.onnx.utils

from models.classify_net1 import classify_net1
from models.net_resnet18 import net_resnet18

path = 'run/train/weights/epoch=299-train_acc=1.0.pth'
f = path.replace('.pth', '.onnx')

x = torch.randn(1, 3, 200, 200)
checkpoint = torch.load(path)

net = net_resnet18()  # classify_net1()
net.load_state_dict(checkpoint['net'])
net.eval()
torch.onnx.export(net,
                  x,
                  f,
                  opset_version=10,
                  # do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  verbose='True')

import onnx

# Checks 参考 yolov7
onnx_model = onnx.load(f)  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model

print('export success!')
