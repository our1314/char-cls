import torch

loss_fn = torch.nn.CrossEntropyLoss()

a = torch.Tensor([[[0.3, 0.4, 0.3]]])
b = torch.Tensor([[[0.29, 0.4, 0.31]]])
l1 = loss_fn(a, b)
pass
a = torch.Tensor([[0.3, 0.4, 0.3]])
b = torch.Tensor([[0.299, 0.4, 0.301]])
l2 = loss_fn(a, b)
pass
a = torch.Tensor([[0.3, 0.4, 0.3]])
b = torch.Tensor([[0.31, 0.38, 0.31]])
l3 = loss_fn(a, b)
pass