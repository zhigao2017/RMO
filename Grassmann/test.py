import torch
import torch.nn as nn

from learning_to_learn_tangent import Learning_to_learn_global_training
from LSTM_Optimizee_Model import LSTM_Optimizee_Model
from hand_optimizer.handcraft_optimizer import Hand_Optimizee_Model
from DataSet.MNIST import MNIST
from utils import FastRandomIdentitySampler

import config
import numpy as np
from torch.autograd import Variable

opt = config.parse_opt()
print(opt)



def f(inputs,M):
    
    X=torch.matmul(M,M.permute(0,2,1))
   
    X2=torch.matmul(X,inputs)
    
    L=torch.norm(inputs-X2,dim=1).pow(2)
    
    L=torch.sum(L)

    return L

def retraction(inputs, grad,lr):


    new_point=torch.zeros(inputs.shape).cuda()
    n=inputs.shape[0]

    P=-lr*grad
    PV=inputs+P

    n1=(PV.shape)[1]
    n2=(PV.shape)[2]
    n_min=min(n1,n2)
    
    U,S,Y=torch.svd(PV)
    new_point=torch.matmul(U[:,:,0:n_min],Y.permute(0,2,1))

    return new_point

LSTM_Optimizee = LSTM_Optimizee_Model(opt,opt.DIM, opt.outputDIM, batchsize_data=opt.batchsize_data, batchsize_para=opt.batchsize_para).cuda()



checkpoint = torch.load('')

LSTM_Optimizee.load_state_dict(checkpoint)

DIM = opt.DIM
outputDIM = opt.outputDIM

X_ours=[]
Y_ours=[]
all_iter=0
N=60000
learning_rate = opt.hand_optimizer_lr
Epoches = 8

train_mnist = MNIST(opt.datapath, train=True)

train_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=opt.batchsize_data,shuffle=True, drop_last=True, num_workers=0)
batchsize_para = opt.batchsize_para

M = torch.randn(opt.batchsize_para, DIM, outputDIM)
for i in range(opt.batchsize_para):
    nn.init.orthogonal_(M[i])
 

state = (torch.zeros(opt.batchsize_para, DIM, outputDIM).cuda(),
            torch.zeros(opt.batchsize_para, DIM, outputDIM).cuda(),
            torch.zeros(opt.batchsize_para, DIM, outputDIM).cuda(),
            torch.zeros(opt.batchsize_para, DIM, outputDIM).cuda()
            ) 

batchsize_data=opt.batchsize_data
train_mnist = MNIST(opt.datapath, train=True)
train_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=batchsize_data,shuffle=True, drop_last=True, num_workers=0)

train_data = np.load('data/dim784_training_images_bool.npy')
train_data = torch.Tensor(train_data)
train_data=train_data.view(batchsize_para, train_data.shape[0]//batchsize_para,-1)
        
train_data=train_data.permute(0,2,1)
train_data = train_data.cuda()
        



M = Variable(M)
M.requires_grad = True
M = M.cuda()

for i in range(Epoches):
    
    for j, data in enumerate(train_loader, 0):
        inputs,labels=data
        inputs=inputs.to('cuda:0')
        inputs=inputs.view(batchsize_para,inputs.shape[0]//batchsize_para,-1)
        
        inputs=inputs.permute(0,2,1)
        
        loss = f(inputs, M)
        print('iter:{},loss:{}'.format(all_iter, loss/batchsize_data))
        M.retain_grad()
        loss.backward()
        M_grad = M.grad
        # print(M_grad)
        #projection
        # P=par_projection(theta0,theta_grad)
        
  
        with torch.no_grad():
            
            lr, update, state = LSTM_Optimizee(M_grad, state)
            print('lr_before', lr)
            lr=lr/(1/opt.hand_optimizer_lr)
            


            update=update+M.grad
            M_update=update-torch.matmul(torch.matmul(M,M.permute(0,2,1)),update)

            try:
                weight_after=retraction(M,M_update,lr)
                
            except:
                print('svd error')
                continue

            weight_after_sum = torch.sum(torch.sum(weight_after)).cpu().numpy()
            if np.isnan(weight_after_sum):
                continue
        
        M = weight_after
        M = Variable(M)
        M.requires_grad = True



        loss_test = f(train_data, M)
        
        print('all_iter:{},loss_all:{}'.format(all_iter,loss_test.item()/N))

        loss_test_N = loss_test.item()/N
        

        Y_ours.append(loss_test.item()/N)
                

        all_iter=all_iter+1


    if i>=Epoches:
        print('END')
        break
        
        
X_ours=np.array(X_ours)
Y_ours=np.array(Y_ours)


np.save('loss_final_grassmann.npy',Y_ours)
