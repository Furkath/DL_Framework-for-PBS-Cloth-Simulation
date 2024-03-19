import torch

# Create two tensors on GPU
tensor1_gpu = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
tensor2_gpu = torch.tensor([[2, 2, 2], [4, 4, 4]], device="cuda")

# Perform element-wise comparison and create a new tensor with the result
comparison_result_gpu = (tensor1_gpu > tensor2_gpu).float()

print(comparison_result_gpu)
