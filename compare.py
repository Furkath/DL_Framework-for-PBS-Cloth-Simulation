import numpy as np

data = np.load("./data/trainPressData.npz")
#data = np.load("./data/trainData.npz")
xdata = data['dataX']
vdata = data['dataV']
xdata =  np.transpose(xdata, (0, 3, 1, 2))
vdata =  np.transpose(vdata, (0, 3, 1, 2))

simu = np.load("./data/simuPressData.npz")
#simu = np.load("./data/simuCrossData.npz")
#simu = np.load("./data/simuFullData.npz")
#simu = np.load("./data/simuData.npz")
xsimu = simu['dataX']
vsimu = simu['dataV']

print(xdata.shape)
print(vdata.shape)
print(xsimu.shape)
print(vsimu.shape)

mx=0
for i in range(xdata.shape[0]):#5000 #range(4811):
    diff = np.abs( xdata[i,:,:,:]-xsimu[i,:,:,:] )
    jack = np.max(diff,axis=None)
    mx=max(mx,jack)
    diff1 = np.sum(diff, axis=2)
    diff2 = np.sum(diff1, axis=1)
    print(diff2)
print("-------------------------")
print(mx)
