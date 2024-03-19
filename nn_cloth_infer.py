import os
import sys
import time
import numpy as np
#import scipy.io as sio
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

initdatapath=sys.argv[1]
modeldatapath=sys.argv[2]
infereddatapath=sys.argv[3]

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#torch.set_default_dtype(torch.float64)

torch.manual_seed(66)
np.random.seed(66)

# Discrete operator (2D)
op0=[[[[ 0, 0, 0],
       [-1, 1, 0],
       [ 0, 0, 0]]]]

op1=[[[[ 0,-1, 0],
       [ 0, 1, 0],
       [ 0, 0, 0]]]]

op2=[[[[ 0, 0, 0],
       [ 0, 1,-1],
       [ 0, 0, 0]]]]

op3=[[[[ 0, 0, 0],
       [ 0, 1, 0],
       [ 0,-1, 0]]]]

cnnKernel2 = np.transpose( np.concatenate((op0,op1,op2,op3),axis=0) , (0,1,3,2) ) #because in program data, [[x0y0,x0y1],[x1y0,x1y1]], y is the last dimension and lies in row! not like practical intuition~

op4=[[[[ 0, 0, 0],
       [-1, 1, 0],
       [ 0, 0, 0]]]]

op5=[[[[-1, 0, 0],
       [ 0, 1, 0],
       [ 0, 0, 0]]]]

op6=[[[[ 0,-1, 0],
       [ 0, 1, 0],
       [ 0, 0, 0]]]]

op7=[[[[ 0, 0,-1],
       [ 0, 1, 0],
       [ 0, 0, 0]]]]

op8=[[[[ 0, 0, 0],
       [ 0, 1,-1],
       [ 0, 0, 0]]]]

op9=[[[[ 0, 0, 0],
       [ 0, 1, 0],
       [ 0, 0,-1]]]]

op10=[[[[ 0, 0, 0],
        [ 0, 1, 0],
        [ 0,-1, 0]]]]

op11=[[[[ 0, 0, 0],
        [ 0, 1, 0],
        [-1, 0, 0]]]]

cnnKernel1 = np.transpose( np.concatenate((op4,op5,op6,op7,op8,op9,op10,op11),axis=0) , (0,1,3,2) ) #as above

class ISRU(nn.Module):
    def __init__(self,c_=1e-9):
        super(ISRU,self).__init__()
        self.register_buffer('c_',torch.tensor(c_))
        #self.tensor_list=tensor_list
        #self.c_=c_
    def forward(self, tensor_list):
        denomi_ =  torch.zeros_like(tensor_list[0])
        for tensor_ in tensor_list:
            denomi_ += tensor_**2
        denomi_ += self.c_*torch.ones_like(tensor_list[0])
        denomi_ = torch.sqrt(denomi_)
        output = []
        for tensor_ in tensor_list:
            output.append( tensor_/denomi_  )
        return output

class CNNBranch(nn.Module):
    # Convolutional NN Cell 
    def __init__(self,  input_kernel_size=3, input_stride=1, input_padding1=1,input_padding2=0,dt=(4e-2)/128):
        super(CNNBranch, self).__init__()

        # Initial parameters
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding1 = input_padding1
        self.input_padding2 = input_padding2

        # Discretization parameter
        self.dt = dt#.cuda()
        #self.dt = self.dt.cuda()
        ddrag=np.exp(-3*self.dt)

        #coefficient

        self.cnnP1   = torch.nn.Parameter(  torch.ones(8,                               dtype=torch.float64), requires_grad=True)
        self.cnnP2   = torch.nn.Parameter(  torch.ones(4,                               dtype=torch.float64), requires_grad=True)
        self.alpha1  = torch.nn.Parameter( -torch.ones(8,                               dtype=torch.float64), requires_grad=True)
        self.alpha2  = torch.nn.Parameter( -torch.ones(4,                               dtype=torch.float64), requires_grad=True)
        self.beta    = torch.nn.Parameter(  torch.tensor(1,                             dtype=torch.float64), requires_grad=True)
        self.gamma   = torch.nn.Parameter( -torch.tensor(1,                             dtype=torch.float64), requires_grad=True)
        self.delta0  = torch.nn.Parameter(  torch.tensor(1,                             dtype=torch.float64), requires_grad=True)
        self.delta1  = torch.nn.Parameter( -torch.tensor(1,                             dtype=torch.float64), requires_grad=True)
        self.delta2  = torch.nn.Parameter(  torch.tensor(1,                             dtype=torch.float64), requires_grad=True)

        # Conv2d operator as operator
        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=self.input_kernel_size,#3,
                                   stride=self.input_stride, padding=self.input_padding1, bias=False, padding_mode='replicate')
        self.cnn_2 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=self.input_kernel_size,#3,
                                   stride=self.input_stride, padding=self.input_padding2, dilation=2, bias=False)#, padding_mode='replicate')

        self.cnn_1.weight = nn.Parameter(torch.DoubleTensor(cnnKernel1), requires_grad=False)
        self.cnn_2.weight = nn.Parameter(torch.DoubleTensor(cnnKernel2), requires_grad=False)
        #self.init_filter([self.cnn_1,self.cnn_2], c=0.5)

        #print(self.cnn_1.weight.data.shape)

        self.ISRU_=ISRU(c_=1e-9)#.cuda()
    
    def init_filter(self, filter_list, c):
        for filter_ in filter_list:
            filter_.weight.data = torch.DoubleTensor(filter_.weight.size()).uniform_(-c * np.sqrt(1 / np.prod(filter_.weight.shape[:-1])),
                                        c * np.sqrt(1 / np.prod(filter_.weight.shape[:-1])))
            '''
            if filter.bias is not None:
                filter.bias.data.fill_(0.0)
            '''

    def forward(self, h):
        #[batch_size,num_channel,hight,width]
        alpha1_= self.alpha1 *1e6
        alpha2_= self.alpha2 *1e6
        beta_  = self.beta   *1e4
        gamma_ = self.gamma  *1e2
        delta0_= self.delta0 *1e0
        delta1_= self.delta1 *1e1
        delta2_= self.delta2 *1e0
        xx1 = self.cnn_1( h[:, 0:1, ...] )*self.cnnP1.view(1,8,1,1)
        yy1 = self.cnn_1( h[:, 1:2, ...] )*self.cnnP1.view(1,8,1,1)
        zz1 = self.cnn_1( h[:, 2:3, ...] )*self.cnnP1.view(1,8,1,1)
        uu1 = self.cnn_1( h[:, 3:4, ...] )*self.cnnP1.view(1,8,1,1)
        vv1 = self.cnn_1( h[:, 4:5, ...] )*self.cnnP1.view(1,8,1,1)
        ww1 = self.cnn_1( h[:, 5:6, ...] )*self.cnnP1.view(1,8,1,1)
        hx=h[:, 0:1, ...]
        hxpad = torch.cat( (hx[:,:,0:2,:],hx,hx[:,:,-2:,:]) , dim=2 )
        hxpad = torch.cat( (hxpad[:,:,:,0:2],hxpad,hxpad[:,:,:,-2:]) , dim=3 )
        hy=h[:, 1:2, ...]
        hypad = torch.cat( (hy[:,:,0:2,:],hy,hy[:,:,-2:,:]) , dim=2 )
        hypad = torch.cat( (hypad[:,:,:,0:2],hypad,hypad[:,:,:,-2:]) , dim=3 )
        hz=h[:, 2:3, ...]
        hzpad = torch.cat( (hz[:,:,0:2,:],hz,hz[:,:,-2:,:]) , dim=2 )
        hzpad = torch.cat( (hzpad[:,:,:,0:2],hzpad,hzpad[:,:,:,-2:]) , dim=3 )
        hu=h[:, 3:4, ...]
        hupad = torch.cat( (hu[:,:,0:2,:],hu,hu[:,:,-2:,:]) , dim=2 )
        hupad = torch.cat( (hupad[:,:,:,0:2],hupad,hupad[:,:,:,-2:]) , dim=3 )
        hv=h[:, 4:5, ...]
        hvpad = torch.cat( (hv[:,:,0:2,:],hv,hv[:,:,-2:,:]) , dim=2 )
        hvpad = torch.cat( (hvpad[:,:,:,0:2],hvpad,hvpad[:,:,:,-2:]) , dim=3 )
        hw=h[:, 5:6, ...]
        hwpad = torch.cat( (hw[:,:,0:2,:],hw,hw[:,:,-2:,:]) , dim=2 )
        hwpad = torch.cat( (hwpad[:,:,:,0:2],hwpad,hwpad[:,:,:,-2:]) , dim=3 )
        xx2 = self.cnn_2( hxpad )*self.cnnP2.view(1,4,1,1)
        yy2 = self.cnn_2( hypad )*self.cnnP2.view(1,4,1,1)
        zz2 = self.cnn_2( hzpad )*self.cnnP2.view(1,4,1,1)
        uu2 = self.cnn_2( hupad )*self.cnnP2.view(1,4,1,1)
        vv2 = self.cnn_2( hvpad )*self.cnnP2.view(1,4,1,1)
        ww2 = self.cnn_2( hwpad )*self.cnnP2.view(1,4,1,1)

        #print(alpha1.device)
        #linear
        outx = torch.cat(( xx1*alpha1_.view(1,8,1,1) , xx2*alpha2_.view(1,4,1,1) ),dim=1)
        outy = torch.cat(( yy1*alpha1_.view(1,8,1,1) , yy2*alpha2_.view(1,4,1,1) ),dim=1)
        outz = torch.cat(( zz1*alpha1_.view(1,8,1,1) , zz2*alpha2_.view(1,4,1,1) ),dim=1)
        #print(outx.device)

        xISRU = self.ISRU_([ torch.cat((xx1,xx2),dim=1) , torch.cat((yy1,yy2),dim=1) , torch.cat((zz1,zz2),dim=1)  ])
        #print(xISRU[0].device)

        nonlix = xISRU[0] 
        nonliy = xISRU[1]
        nonliz = xISRU[2]
        
        outx += beta_*nonlix
        outy += beta_*nonliy
        outz += beta_*nonliz

        correl = torch.cat((uu1,uu2),dim=1)*nonlix+torch.cat((vv1,vv2),dim=1)*nonliy+torch.cat((ww1,ww2),dim=1)*nonliz
        outx += gamma_*correl*nonlix
        outy += gamma_*correl*nonliy
        outz += gamma_*correl*nonliz
        #print(correl.device)

        outx += delta0_*torch.ones_like(outx)/12 #channel normalization
        outy += delta1_*torch.ones_like(outy)/12
        outz += delta2_*torch.ones_like(outz)/12

        outt = torch.cat((outx.sum(dim=1).unsqueeze(1),outy.sum(dim=1).unsqueeze(1),outz.sum(dim=1).unsqueeze(1)),dim=1)
        #print(outt.device)
        return outt*self.dt


def phyLoss(output,true):
    mse_loss = nn.MSELoss()
    loss = mse_loss(output,true)
    return loss


def train(model, pde, init_state, n_iters, learning_rate, dt, batch_size, save_path):
    # model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3200, 6400], gamma=0.1)
    loss_pre = 1e9
    staDict_pre = copy.deepcopy( model.state_dict() )
    for epoch in range(n_iters):
        data_iter = DataLoader(TensorDataset(init_state), batch_size, drop_last=True)
        loss_iter = []
        for batch_i, init_state_i in enumerate(data_iter):
            init_state_i = torch.cat(tuple(init_state_i)).cuda()
            optimizer.zero_grad()
            # output is a tensor
            output = model(init_state_i)
            true = pde(init_state_i)
            loss = phyLoss(output,true)
            loss.backward()
            #check_device_of_parameters(model)
            optimizer.step()
            loss_iter.append(loss.item())
        scheduler.step()
        loss_mean = sum(loss_iter) / len(loss_iter)
        # print loss in each epoch
        print('[%d/%d %d%%] Epoch loss: %.15f, ' % ((epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), loss_mean))
        for param_group in optimizer.param_groups:
            print('LR: ', param_group['lr'])
        if loss_mean < loss_pre:
            loss_pre = loss_mean
            staDict_pre = copy.deepcopy( model.state_dict() )
        if (epoch+1) % 20 == 0: #quicker debug
            save_model(model, 'nn_'+str(epoch+1), save_path)
    
    save_staDict(staDict_pre,'nn_final', save_path)
            #debug
            #save_staDict(staDict_pre,'nn_final', save_path)


class clothNet(nn.Module):
    def __init__(self,dt_=(4e-2)/128,pressure=0): #can add index and velocity for B.C., and collision setting
        super(clothNet, self).__init__() 
        self.dt = torch.tensor(dt_, dtype=torch.float64).cuda()
        self.drag = torch.tensor(np.exp(-3*dt_), dtype=torch.float64).cuda()
        self.press = torch.tensor(pressure, dtype=torch.float64).cuda()
        self.cnncell = CNNBranch(dt=dt_) #.cuda() can be called outside
        self.ballcenter = torch.tensor([0.,0.,0.], dtype=torch.float64).cuda()
        self.radius = torch.tensor( 0.3, dtype=torch.float64).cuda()

    def crossP(self,x1,x2):#[:,3,:,:]
        ii=x1[:,1:2,:,:]*x2[:,2:3,:,:]-x1[:,2:3,:,:]*x2[:,1:2,:,:]
        jj=x1[:,2:3,:,:]*x2[:,0:1,:,:]-x1[:,0:1,:,:]*x2[:,2:3,:,:]
        kk=x1[:,0:1,:,:]*x2[:,1:2,:,:]-x1[:,1:2,:,:]*x2[:,0:1,:,:]
        return torch.cat((ii,jj,kk),dim=1)


    def pressureCal(self,xx):
        rolli1  = torch.cat((xx[:,:,1:,:],xx[:,:,-1:,:]),dim=2)
        rolli_1 = torch.cat((xx[:,:,0:1,:],xx[:,:,0:-1,:]),dim=2)
        rollj1  = torch.cat((xx[:,:,:,1:],xx[:,:,:,-1:]),dim=3)
        rollj_1 = torch.cat((xx[:,:,:,0:1],xx[:,:,:,0:-1]),dim=3)
        
        rolli1  = xx - rolli1
        rolli_1 = xx - rolli_1
        rollj1  = xx - rollj1
        rollj_1 = xx - rollj_1

        fp=self.crossP(rolli_1,rollj1)+self.crossP(rollj1,rolli1)+self.crossP(rolli1,rollj_1)+self.crossP(rollj_1,rolli_1)
        return fp*self.press

    def forward(self,h):
        #basicf = self.cnncell(h)
        xx = h[:, 0:3, ...]
        v  = h[:, 3:6, ...]
        basicf = self.cnncell(h)
        #print(basicf.shape)
        basicf += self.pressureCal(xx)*self.dt*self.drag
        #print(basicf.shape)
        vout = v*self.drag+basicf
        #print(vout.shape)
        
        #apply constrain
        #if ball
        rr = xx - self.ballcenter.view(1,3,1,1)
        rrnorm = torch.sqrt( rr[:,0:1, ...]**2+rr[:,1:2, ...]**2+rr[:,2:3, ...]**2 + 1e-10*torch.ones_like(rr[:,0:1, ...]) )
        rr = rr/rrnorm
        flag1 = (rrnorm <= self.radius).float()
        rv = vout[:,0:1, ...]*rr[:,0:1, ...]+vout[:,1:2, ...]*rr[:,1:2, ...]+vout[:,2:3, ...]*rr[:,2:3, ...]
        flag2 = (rv < 0.0).float()
        vout-=flag1*flag2*rv*rr
        vout-=flag1*flag2*0.05*vout
        #if B.C. 1
#        vout = torch.cat( ( torch.zeros_like(vout[:,:,0:1,:]) , vout[:,:,1:-1,:]  , torch.zeros_like(vout[:,:,-1:,:])  ),dim=2 )
#        vout = torch.cat( ( torch.zeros_like(vout[:,:,:,0:1]) , vout[:,:,:,1:-1]  , torch.zeros_like(vout[:,:,:,-1:])  ),dim=3 )
        #if B.C. 2
#        vout[:,:,0,0]=0.0                
        output = torch.cat( (xx+self.dt*vout , vout),dim=1)
        return output

class pdeFix(nn.Module):
    def __init__(self, dt=(4e-2)/128):
        super(pdeFix, self).__init__()
        # Initial parameters
        # forward Euler ddt, airdrag
        self.dt = torch.tensor(dt,              dtype=torch.float64).cuda()#.cuda()#4e-2/128 #0.0125  # 10/800
        self.ddrag=torch.tensor(np.exp(-3*dt),  dtype=torch.float64).cuda()
        #physic properties
        self.spriY = torch.tensor( 1e4      ,dtype=torch.float64).cuda()
        self.quadS = torch.tensor(1./128     ,dtype=torch.float64).cuda()
        self.dashD = torch.tensor( 3e4      ,dtype=torch.float64).cuda()
        self.g     = torch.tensor([0,-9.8,0],dtype=torch.float64).cuda()
        self.rolls = [[-2,0],[0,2],[2,0],[0,-2],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]

    def forward(self, h):
        #[batch_size,num_channel,hight,width]
        xx = h[:, 0:3, ...] 
        vv = h[:, 3:6, ...] 
        ff = torch.zeros_like(xx)
        #df = torch.zeros_like(vv)
        for roll_i in self.rolls: #iterations
            #for ind in range(0,2):
            i = roll_i[0]#[ind]
            j = roll_i[1]
            #print(i)
            #print(j)
            rollx,rollv = xx,vv
            if i == 1:
                rollx = torch.cat((rollx[:,:,1:,:],rollx[:,:,-1:,:]),dim=2)
                rollv = torch.cat((rollv[:,:,1:,:],rollv[:,:,-1:,:]),dim=2)
            elif i == 2:
                rollx = torch.cat((rollx[:,:,2:,:],rollx[:,:,-2:,:]),dim=2)
                rollv = torch.cat((rollv[:,:,2:,:],rollv[:,:,-2:,:]),dim=2)
            elif i == -1:
                rollx = torch.cat((rollx[:,:,0:1,:],rollx[:,:,0:-1,:]),dim=2)
                rollv = torch.cat((rollv[:,:,0:1,:],rollv[:,:,0:-1,:]),dim=2)
            elif i == -2: 
                rollx = torch.cat((rollx[:,:,0:2,:],rollx[:,:,0:-2,:]),dim=2)
                rollv = torch.cat((rollv[:,:,0:2,:],rollv[:,:,0:-2,:]),dim=2)
            if j == 1:
                rollx = torch.cat((rollx[:,:,:,1:],rollx[:,:,:,-1:]),dim=3)
                rollv = torch.cat((rollv[:,:,:,1:],rollv[:,:,:,-1:]),dim=3)
            elif j == 2:
                rollx = torch.cat((rollx[:,:,:,2:],rollx[:,:,:,-2:]),dim=3)
                rollv = torch.cat((rollv[:,:,:,2:],rollv[:,:,:,-2:]),dim=3)
            elif j == -1:
                rollx = torch.cat((rollx[:,:,:,0:1],rollx[:,:,:,0:-1]),dim=3)
                rollv = torch.cat((rollv[:,:,:,0:1],rollv[:,:,:,0:-1]),dim=3)
            elif j == -2:
                rollx = torch.cat((rollx[:,:,:,0:2],rollx[:,:,:,0:-2]),dim=3)
                rollv = torch.cat((rollv[:,:,:,0:2],rollv[:,:,:,0:-2]),dim=3)
            rollx = xx - rollx
            rollv = vv - rollv
        #debug
        #    ff+=rollx
        #    df+=rollv
        #return torch.cat((ff,df),dim=1)
            ff += -self.spriY/np.sqrt(i**2+j**2)/self.quadS*rollx
            normx = (rollx**2).sum(dim=1).unsqueeze(1)
            normx += torch.ones_like(normx)*1e-9 
            normx = torch.sqrt(normx)
            xnorm = rollx/normx
            ff += self.spriY*xnorm
            ff += -self.quadS*self.dashD*((rollv*xnorm).sum(dim=1).unsqueeze(1))*xnorm
        ff += self.g.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(ff.shape[0], 3, ff.shape[2], ff.shape[3])  # or I can use .view(~)
        ff *= self.ddrag*self.dt
        return ff


def simulate(clonet,hh,num):
    out=hh.clone()
    #out=hh.clone().detach().cpu()
    for i in range(num):
        print(i)
        #if i==300:        
        #    print("----------record 300 at:",time.time())
        outo = clonet(hh)
        #may put to cpu if too big, but only once a period, otherwise you are let VRAM unused, and io can cost much time
        out=torch.cat((out,outo),dim=0)
        #out=torch.cat((out,outo.clone().detach().cpu()),dim=0)
        hh = outo
    outpy=out.detach().cpu().numpy()
    #outpy=out.numpy()
    np.savez(infereddatapath,dataX=outpy[:,0:3,:,:],dataV=outpy[:,3:6,:,:])
    #np.savez("simuHangData",dataX=outpy[:,0:3,:,:],dataV=outpy[:,3:6,:,:])
    #np.savez("simuBallData",dataX=outpy[:,0:3,:,:],dataV=outpy[:,3:6,:,:])
    #np.savez("simuFullData",dataX=outpy[:,0:3,:,:],dataV=outpy[:,3:6,:,:])
    #np.savez("simuCrossData",dataX=outpy[:,0:3,:,:],dataV=outpy[:,3:6,:,:])
    #np.savez("simuData",dataX=outpy[:,0:3,:,:],dataV=outpy[:,3:6,:,:])
    #np.savez("simuPressData",dataX=outpy[:,0:3,:,:],dataV=outpy[:,3:6,:,:])


def save_model(model, model_name, save_path):
    torch.save(model.state_dict(), save_path + model_name + '.pt')

def save_staDict(sta_Dict, model_name, save_path):
    torch.save(sta_Dict, save_path + model_name + '.pt')

def load_model(model, model_name, save_path):
    model.load_state_dict(torch.load(save_path + model_name + '.pt'))

def check_device_of_parameters(model): # Iterate through all parameters in the model 
    for name, param in model.named_parameters(): 
        # Print the parameter name and its device 
        print(f"Parameter: {name}, Device: {param.dtype}")#param.device

if __name__ == '__main__':

    ################# prepare the input data settings ####################
    dt = 4e-2/128#10.0 / 800
    ################### define the Initial conditions ####################
    #data = np.load("./data/trainData.npz")
    #data = np.load("./data/trainPressData.npz")
    data = np.load( initdatapath )
    xdata = data['dataX']
    vdata = data['dataV']
    init_state =  np.transpose(np.concatenate((xdata, vdata), axis=3), (0, 3, 1, 2))
    print(init_state.shape)
    init_state = torch.tensor(init_state, dtype=torch.float64)
    in_state = init_state[0:1,:,:,:].cuda()
    ################# build the model #####################
    # define the model hyper-parameters
    learning_rate = 1e-3#1e-3
    n_iters = 100#2000
    batch_size =256#128
    save_path = './model/'
    #press = 3e6/2/2/4
    #press = 0
    press = 1e5/16
    #model = CNNBranch().cuda()
    #model = pdeFix().cuda()
    simu  = clothNet(pressure=press).cuda()
    #load_model(simu.cnncell,"nn_final","./model_fineTune/")
    #load_model(simu.cnncell,"nn_final","./model/")
    #load_model(simu.cnncell,"nn_final","./model_full/")
    load_model(simu.cnncell, "", modeldatapath)

    #print(model.state_dict())
    #print(pde.state_dict())

    # infer the model
    start = time.time()
    print("----------start at:",start)
    #train(model, pde, init_state, n_iters, learning_rate, dt, batch_size, save_path)
    with torch.no_grad():
        simulate(simu,in_state,5000)
    end = time.time()
    print('The data collection time is: ', (end - start))


