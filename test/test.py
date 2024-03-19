import torch 
import torch.nn as nn
import numpy as np
#rsult = torch.cuda.is_available()
#print(rsult)

#a=torch.rand(2,2,2)
#b=a[:,1,:]
#c=a[:,1:,:]
#print(b)
#print(c)

a = torch.tensor([[ [[0,0,0],[1,1,1],[2,2,2]] ],   
                  [ [[1,1,1],[2,2,2],[3,3,3]] ]])

op0=[[[[ 0, 0, 0],
       [-1, 1, 0],
       [ 0, 0, 0]]]]

op1=[[[[ 0,-1, 0],
       [ 0, 1, 0],
       [ 0, 0, 0]]]]

cnnKernel2 = np.concatenate((op0,op1),axis=0)

cnn_2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3,
                                   stride=1, padding=2, dilation=2, bias=False, padding_mode='replicate')

cnn_2.weight = nn.Parameter(torch.DoubleTensor(cnnKernel2), requires_grad=False)


