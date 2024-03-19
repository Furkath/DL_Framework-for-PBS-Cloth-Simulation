
import torch

    # Your original tensor
original_tensor = torch.tensor([0, 10, 0])

    # Desired shape [a, 3, b, c]
a, b, c = 2, 2, 2
expanded_tensor = original_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(a, 3, b, c)
    # Unsqueeze to add a dimension at the beginning
#expanded_tensor = torch.unsqueeze(original_tensor, 0)

    # Expand along the specified dimensions
#expanded_tensor = expanded_tensor.expand(a, -1, b, c)

print(expanded_tensor.shape)
print(expanded_tensor)
