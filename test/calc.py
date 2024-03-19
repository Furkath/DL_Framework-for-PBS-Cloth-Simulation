import numpy as np

dt=(4e-2)/128
ddrag=np.exp(-3*dt)
print(ddrag)
alpha = -1e4*128/np.sqrt(2)*ddrag
print(alpha)

print(1000/16)
