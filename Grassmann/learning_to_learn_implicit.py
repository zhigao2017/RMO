import torch
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import time
import math, random
import numpy as np


from losses.LOSS import ContrastiveLoss
from ReplyBuffer import ReplayBuffer
from retraction import Retraction

retraction=Retraction(1)





def f(inputs,M):
    
    X=torch.matmul(M,M.permute(0,2,1))
    X2=torch.matmul(X,inputs)
  
    L=torch.norm(inputs-X2,dim=1).pow(2)
    #print('L',torch.norm(inputs-X2,dim=1).shape)
    L=torch.sum(L)

    return L

def optimizee_loss(optimizer):
    ##这个loss并不会用于训练，而是用于将参数的grad转为非none##
    optimizer_param_sum = 0

    for param in optimizer.parameters():
        optimizer_param_sum = optimizer_param_sum + torch.norm(param)
    return optimizer_param_sum

def Hessian_vector_product(M, vector,inputs):
    """
    Performs hessian vector product on the train set in task with the provided vector
    """
    
    # inputs_clone = inputs.clone()
    # labels_clone = labels.clone()
    loss = f(inputs, M)
    grad_M = torch.autograd.grad(loss, M, create_graph=True, retain_graph =True)[0]
    # flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
    vec = vector.cuda()
    h = torch.sum(grad_M * vec)
    hvp = torch.autograd.grad(h, M)[0]
    
    return hvp

def Observe_function(opt, hand_optimizee,hand_optimizee_csgdm,hand_optimizee_rasa, train_loader,optimizer_flag):
    DIM=opt.DIM
    outputDIM=opt.outputDIM
    batchsize_para=opt.batchsize_para
    Observe=opt.Observe
    batchsize_data = opt.batchsize_data
    RB=ReplayBuffer(1000*batchsize_para)

    Square=torch.eye(DIM)

    print('Observe', Observe)
    while(1):
        count=1
        outer_break = True
        inner_break = False

        M=torch.randn(batchsize_para,DIM, outputDIM).cuda()
        mom = torch.zeros_like(M).cuda()
        L_0 = torch.empty(opt.batchsize_para ,DIM, dtype=torch.float,device=torch.device("cuda:0"))
        torch.nn.init.zeros_(L_0)
        
        R_0 = torch.empty(opt.batchsize_para ,outputDIM, dtype=torch.float,device=torch.device("cuda:0"))
        torch.nn.init.zeros_(R_0)

        M_element_number = M.shape[0]*M.shape[1]*M.shape[2]
        for k in range(batchsize_para):
            #print('before MtM', torch.mm(M[k].t(),M[k]))
            nn.init.orthogonal_(M[k])
        # M=torch.randn(batchsize_para,DIM, outputDIM).cuda()
        state = (torch.zeros(batchsize_para,DIM,outputDIM).cuda(),
                                         torch.zeros(batchsize_para,DIM,outputDIM).cuda(),
                                         torch.zeros(batchsize_para,DIM,outputDIM).cuda(),
                                         torch.zeros(batchsize_para,DIM,outputDIM).cuda(),
                                         ) 
                
        M.requires_grad=True
        iteration=torch.zeros(batchsize_para)
        # RB.push(state,M,iteration)  

        for i in range(Observe):
                 
            
            print ('observe finish',count)

            break_flag=False
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels).cuda()
                inputs=inputs.view(batchsize_para,inputs.shape[0]//batchsize_para,-1)
                labels=labels.view(batchsize_para,labels.shape[0]//batchsize_para)
                inputs=inputs.permute(0,2,1)
                loss = f(inputs,M)

                loss.backward()

                g = M.grad.detach()
                gradient_norm = torch.norm(g)
                gradient_norm = gradient_norm**2
                gradient_norm = gradient_norm/M_element_number

                print('count:{}, loss:{}, g:{}'.format(count, loss/batchsize_data, gradient_norm))
                
                if np.isnan(gradient_norm.detach().cpu().numpy()):
                    inner_break = True
                    outer_break = False
                    print('NAN ERROR')

                if inner_break == True:
                    break

                if optimizer_flag == 0:
                    try:  
                        M, state = hand_optimizee(M.grad, M, state)
                    except:
                        inner_break = True
                        outer_break = False
                        print('SVD error')
                        break
                elif optimizer_flag == 1:
                    try:  
                        M, state,mom = hand_optimizee_csgdm(M.grad, M, state,mom)
                    except:
                        inner_break = True
                        outer_break = False
                        print('SVD error')
                        break
                elif optimizer_flag == 2:
                    try:
                        M, state, L_0, R_0 = hand_optimizee_rasa(M.grad, M, state, L_0, R_0)
                    except:
                        inner_break = True
                        outer_break = False
                        print('SVD error')
                        break

                print('-------------------------')


                state = (state[0].detach(),state[1].detach(),state[2].detach(),state[3].detach())
                M=M.detach()
                L_0.detach()
                R_0.detach()
                mom.detach()
                # M.retain_grad()
                M.requires_grad=True

                if count>2000:
                    RB.push(state, M, iteration)


                count=count+1
                print ('loss',loss/batchsize_data)
                print ('observe finish',count)
                localtime = time.asctime( time.localtime(time.time()) )

                if count==Observe:
                    break_flag=True
                    break
            if inner_break == True:
                break

            if break_flag==True:
                break             
        
        if outer_break == True:
            break
        RB.shuffle()
    return RB





def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def Learning_to_learn_global_training(opt,hand_optimizee,hand_optimizee_csgdm,hand_optimizee_rasa,optimizee,train_loader):

    DIM=opt.DIM
    outputDIM=opt.outputDIM
    batchsize_para=opt.batchsize_para
    batchsize_data = opt.batchsize_data
    Observe=opt.Observe
    Epochs=opt.Epochs
    Optimizee_Train_Steps=opt.Optimizee_Train_Steps
    optimizer_lr=opt.optimizer_lr
    Decay=opt.Decay
    Decay_rate=opt.Decay_rate
    Imcrement=opt.Imcrement
    Neumann_alpha = opt.Neumann_alpha
    Neumann_series = opt.Neumann_series
    optimizer_flag = 0

    #adam_global_optimizer = torch.optim.Adam(optimizee.parameters(),lr = optimizer_lr)
    adam_global_optimizer = torch.optim.Adamax(optimizee.parameters(),lr = optimizer_lr)

    RB = Observe_function(opt, hand_optimizee,hand_optimizee_csgdm,hand_optimizee_rasa, train_loader,optimizer_flag)


    check_point=optimizee.state_dict()
    check_point2=optimizee.state_dict()
    check_point3=optimizee.state_dict()
    optimizer_iteration = 0
    optimizer_update = 0
    for i in range(Epochs): 
        print('\n=======> global training steps: {}'.format(i))
        if (i+1) % Decay==0 and (i+1) != 0:
            count=count+1
            adjust_learning_rate(adam_global_optimizer, Decay_rate)

        if opt.Imcrementflag==True:
            if (i+1) % Imcrement==0 and (i+1) != 0:
                Optimizee_Train_Steps=Optimizee_Train_Steps+50

        if (i+1) % opt.modelsave==0 and (i+1) != 0:
            if opt.Pretrain==True:
                torch.save(optimizee.state_dict(), 'state/'+str(i)+'_'+str(opt.optimizer_lr*1000)+'_Decay'+str(opt.Decay)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'.pth')
            else:
                torch.save(optimizee.state_dict(), 'state/'+str(i)+'_'+str(opt.optimizer_lr*1000)+'_Decay'+str(opt.Decay)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'nopretrain_newlr_meanvar_devide2'+'.pth')


        
        # M.retain_grad()
        state, M, iteration = RB.sample(batchsize_para)
        state = (state[0].detach(), state[1].detach(), state[2].detach(), state[3].detach())
        M = M.detach()
        M = Variable(M)
        M.requires_grad = True
        
        flag=False
        break_flag=False
        count=0
        adam_global_optimizer.zero_grad()
        NAN_error = False
        
        while(1):
            for j, data in enumerate(train_loader, 0):
                print('---------------------------------------------------------------------------')
                #print('M',M)
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels).cuda()
                inputs=inputs.view(batchsize_para,inputs.shape[0]//batchsize_para,-1)
                labels=labels.view(batchsize_para,labels.shape[0]//batchsize_para)

                inputs=inputs.permute(0,2,1)
                loss = f(inputs,M)
                loss_test = loss.detach().cpu().numpy()
                if np.isnan(loss_test):
                    NAN_error = True
                    break_flag = True
                    break
                loss.backward()
                print('count',count,'loss',loss/batchsize_data)
                g = M.grad.detach()

                M_element_number = M.shape[0]*M.shape[1]*M.shape[2]
                gradient_norm = torch.norm(g)
                gradient_norm = gradient_norm**2
                gradient_norm = gradient_norm/M_element_number
                
                gradient_norm = gradient_norm.cpu().numpy()
                if np.isnan(gradient_norm):
                    NAN_error = True
                    break_flag = True
                    break

                if count >= opt.train_steps - 1:
                    M = M.detach()
                    M.requires_grad = True
                    

                    loss_temp = f(inputs, M)
                    g_temp = torch.autograd.grad(loss_temp, M, retain_graph = True)[0]
                    
                    g_temp = Variable(g_temp)
                    g_temp.requires_grad = True

                    lr, update, state = optimizee(g_temp, state)

                    lr = abs(lr)
                    lr = lr/(1/opt.hand_optimizer_lr)
                    
                    update = update+g_temp

                
                    update_R=update-torch.matmul(torch.matmul(M,M.permute(0,2,1)),update)
                    
      
                    
                    update_R = -update_R*lr
                    
                    try:
                        M_end = retraction(M, update_R, 1)
                    except:
                        print('SVD ERROR')
                        break_flag = True
                        break
                    

                    vector = g_temp.detach().clone()
                    Jacobi_vector = vector
                    vector_temp_list = []

                    for i in range(Neumann_series):
                        vector_temp = torch.autograd.grad(M_end, update_R, grad_outputs = vector, retain_graph = True)[0]

                        
                        vector_temp = torch.autograd.grad(update_R, g_temp, grad_outputs = vector_temp, retain_graph = True)[0]
                        
                        vector_temp = Hessian_vector_product(M, vector_temp, inputs)
                        vector_temp2 = torch.autograd.grad(M_end, M, grad_outputs = vector, retain_graph = True)[0]

                        vector = vector - Neumann_alpha * (vector_temp+vector_temp2)

                        Jacobi_vector = Jacobi_vector + vector
                    

                    vector_M_update = torch.autograd.grad(M_end, update_R, grad_outputs = Jacobi_vector, retain_graph = True)
                    
                    update_R_optimizee = torch.autograd.grad(update_R, optimizee.parameters(), grad_outputs = vector_M_update, retain_graph = True)

                    check = 0
                    for p in optimizee.parameters():
                        check = check+1 if type(p.grad) == type(None) else check
                    if check>0:
                        print('-------------------------------------')
                        t_temp = time.time()
                        localtime_temp = time.asctime( time.localtime(time.time()) ) 
                        back_loss = optimizee_loss(optimizee)
                        
                        back_loss.backward()

                    for i,p in enumerate(optimizee.parameters()):
                        p.grad = update_R_optimizee[i]

                    
                    adam_global_optimizer.step()
                    
                    optimizer_iteration = optimizer_iteration + 1
                    break_flag = True

                    break

                else:
                    with torch.no_grad():
                        lr, update, state = optimizee(g, state)

                        
                        lr = lr/(1/opt.hand_optimizer_lr)
                        lr = abs(lr)
                        

                        update=update+M.grad

            
                        s=torch.sum(state[0])+torch.sum(state[1])+torch.sum(state[2])+torch.sum(state[3])
                        if s > 100000:
                            break_flag=True
                            flag=True
                            break
            

                        M.grad.data.zero_()

                        update_R=update-torch.matmul(torch.matmul(M,M.permute(0,2,1)),update)
                        update_R = -update_R*lr
                        try:
                            M = retraction(M,update_R,1)
                        except:
                            print('ERROR')
                            break_flag = True
                            break

                        M = M.detach()
                        M.requires_grad = True


                count=count+1
                print('count',count, 'break_flag', break_flag)
            print('break_flag', break_flag)
            if break_flag == True:
                print('-----------------------')
                break
        
        if optimizer_iteration % opt.observe_reset==0 and optimizer_iteration !=0:
            print('Reset the weight of optimizer')
            print('optimizer_iteration:{}'.format(optimizer_iteration))
            optimizer_flag = (optimizer_flag+1)%3
            RB = Observe_function(opt, hand_optimizee,hand_optimizee_csgdm,hand_optimizee_rasa, train_loader,optimizer_flag)

        # if NAN_error == True:
        #     M, state = Observe_function(opt, hand_optimizee, train_loader)

        if flag==False:
            check_point=check_point2
            check_point2=check_point3
            check_point3=optimizee.state_dict()         
        else:
            print('=====>eigenvalue break, reloading check_point')
            optimizee.load_state_dict(check_point)

        