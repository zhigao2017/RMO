import torch
import torch.nn as nn
from torch.autograd import Variable as V
from MatrixLSTM.MatrixLSTM import MatrixLSTM
from hand_optimizer.retraction import Retraction

class Hand_Optimizee_Model_rasa(nn.Module): 
    def __init__(self,lr):
        super(Hand_Optimizee_Model_rasa,self).__init__()
        self.lr=lr
        self.retraction=Retraction(self.lr)

    def forward(self,grad,M,state,L_0, R_0):
        n  = M.shape[0]
        D_out = M.shape[2]
        D_in = M.shape[1]
        update = torch.ones_like(M)
        

        for i in range(n):
            U = grad[i]        
            w1 = M[i]
            l0 = L_0[i]
            r0 = R_0[i]
            with torch.no_grad():
                w=w1.mm(torch.transpose(w1,0,1))
                l=0.5*l0+0.5*torch.diag(U.mm(torch.transpose(U,0,1)))/D_out
                ll=torch.max(l0,l)
                l0=ll
                r=0.5*r0+0.5*torch.diag(torch.transpose(U,0,1).mm(U))/D_in
                rr=torch.max(r0,r)
                r0=rr
                LL=torch.diag(ll.pow(-1/4))
                UU=LL.mm(U)
                UU=UU.mm(torch.diag(rr.pow(-1/4)))
                
                PP=UU-w.mm(UU)
                update[i] = PP*self.lr
                # w1 = self.retraction(w1,PP*self.lr)
                

                # M[i] = w1
                L_0[i] = l0
                R_0[i] = r0
        M = self.retraction(M,update*self.lr)

        return   M, state, L_0, R_0

