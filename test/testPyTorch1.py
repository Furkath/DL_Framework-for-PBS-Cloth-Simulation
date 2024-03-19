import torch
import torch.nn as nn

class ISRU(nn.Module):
    def __init__(self, c):
        super(ISRU, self).__init__()
        self.register_buffer('c', torch.tensor(c))

    def forward(self, tensor_list):
        denomi_ = torch.zeros_like(tensor_list[0])
        for tensor_ in tensor_list:
            denomi_ += tensor_**2
        denomi_ += self.c * torch.ones_like(tensor_list[0])
        denomi_ = torch.sqrt(denomi_)
        output = []
        for tensor_ in tensor_list:
            output.append(tensor_ / denomi_)
        return output

    # Example usage
c_value = 1e-9 # Replace with your desired constant value
isru_layer = ISRU(c=c_value)

    # Assuming tensor_list is a list of PyTorch tensors
tensor_list = [torch.ones((3,3)), torch.ones((3,3))]

    # Forward pass through the ISRU layer
output_list = isru_layer(tensor_list)

print(output_list)


