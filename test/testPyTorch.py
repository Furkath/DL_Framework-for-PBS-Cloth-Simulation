import torch

# Create a 1D tensor of shape [12]
vector = torch.rand(3)
print(vector)
# Create a 4D tensor of shape [1000, 12, 100, 100]
tensor_4d = torch.ones((2, 3, 2, 2))
print(tensor_4d)
# Multiply the 1D tensor with the 4D tensor along the second dimension
result = tensor_4d * vector.view(1, 3, 1, 1)
print(result)
# Check the shape of the result
#print(result.shape)
