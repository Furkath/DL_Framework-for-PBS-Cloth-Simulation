import numpy as np
from matplotlib import pyplot as plt
#plt.rcParams["font.family"]='DejaVu Sans'

losses = np.load("losses_P.npy")
losses_ = np.load("losses_noP.npy")
appendix = np.loadtxt("losses_noP_A1.txt")
losses_ = np.concatenate((losses_,appendix))
#print(losses.shape)
#print(losses.type)
#print(losses[2])
jack=losses#[:1600]
bob=losses_
#jack = np.log(jack)
#jack = np.log10(jack)
#bob = np.log10(bob)
#plt.plot(jack)
#plt.show()


#fig,axs = plt.subplots(1,2,sharey=True,figsize=(24,10))
fig,axs = plt.subplots(1,2,figsize=(26,10))

axs[0].spines["top"].set_linewidth(3)
axs[0].spines["bottom"].set_linewidth(3)
axs[0].spines["left"].set_linewidth(3)
axs[0].spines["right"].set_linewidth(3)
axs[0].xaxis.set_tick_params(size=10,width=3,labelsize=30,labelfontfamily="serif")
axs[0].yaxis.set_tick_params(size=10,width=3,labelsize=30,labelfontfamily="serif")
axs[0].set_yscale('log')
axs[0].set_xlabel("Epochs",fontsize=32,fontfamily="serif")#,fontweight='bold')
axs[0].set_ylabel("Average Loss",fontsize=32,fontfamily="serif")#,fontweight='bold')
axs[0].text(0.7, 0.85, 'with super-pressure', fontsize=36,fontfamily="serif", horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
axs[0].plot(jack,color='#ee3377')   

axs[1].spines["top"].set_linewidth(3)
axs[1].spines["bottom"].set_linewidth(3)
axs[1].spines["left"].set_linewidth(3)
axs[1].spines["right"].set_linewidth(3)
axs[1].xaxis.set_tick_params(size=10,width=3,labelsize=30,labelfontfamily="serif")
axs[1].yaxis.set_tick_params(size=10,width=3,labelsize=30,labelfontfamily="serif")
axs[1].set_yscale('log')
axs[1].set_xlabel("Epochs",fontsize=32,fontfamily="serif")#,fontweight='bold')
axs[1].set_ylabel("Average Loss",fontsize=32,fontfamily="serif")#,fontweight='bold')
axs[1].text(0.7, 0.85, 'without pressure', fontsize=36,fontfamily="serif", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
axs[1].plot(bob,color='#009988')

plt.show()
