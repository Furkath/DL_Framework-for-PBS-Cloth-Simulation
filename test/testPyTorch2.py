import torch

# Assuming your tensor with shape [a, b, c, c]
tensor_a = torch.ones((2, 4, 2, 2))*2

# Tensor with shape [a, 1, c, c]
tensor_b = torch.ones((2, 1, 2, 2))

# Broadcasting to perform element-wise division
result_tensor = tensor_a / tensor_b

print(result_tensor.shape)
print(result_tensor)
