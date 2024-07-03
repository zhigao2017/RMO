import torch
import torch.nn as nn
from torch.autograd import Variable as V
from MatrixLSTM.MatrixLSTM import MatrixLSTM
from hand_optimizer.retraction import Retraction

class Hand_Optimizee_Model_csgdm(nn.Module): 
    def __init__(self,lr):
        super(Hand_Optimizee_Model_csgdm,self).__init__()
        self.lr=lr
        self.retraction=Retraction(self.lr)

    def forward(self,grad,M,state,mom):
        mom_M = mom - torch.matmul(torch.matmul(M, M.permute(0,2,1)),grad)
        grad_R=grad-torch.matmul(torch.matmul(M,M.permute(0,2,1)),grad)
        mom_M = 0.1*mom_M + grad_R*self.lr
        M = self.retraction(M,mom_M)
        #grad_R=grad
        #M=M-self.lr*grad_R

        return M,state,mom_M

