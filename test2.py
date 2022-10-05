import torch

loss_fn = torch.nn.CrossEntropyLoss()

a = torch.Tensor([[[1.0, 2.0, 3.0]]])
b = torch.Tensor([[[1.0, 2.0, 3.1]]])
l1 = loss_fn(a, b)
pass
a = torch.Tensor([[1.1, 2.0, 3.0]])
b = torch.Tensor([[1.0, 2.0, 3.1]])
l2 = loss_fn(a, b)
pass
a = torch.Tensor([[1.0, 2.0, 3.0]])
b = torch.Tensor([[1.001, 2.0, 3.001]])
l3 = loss_fn(a, b)
pass