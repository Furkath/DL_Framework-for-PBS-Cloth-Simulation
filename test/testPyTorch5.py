import torch
a = torch.tensor([[[0.01, 0.011], [0.009, 0.9]],[[0.01, 0.011], [0.009, 0.9]]])
mask = a > torch.tensor(0.01)
print(mask.float())
