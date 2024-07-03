import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--DIM', type=int, default=784)
    parser.add_argument('--outputDIM', type=int, default=128)
    parser.add_argument('--batchsize_para', type=int, default=10)
    parser.add_argument('--batchsize_data', type=int, default=640)
    parser.add_argument('--datapath', type=str, default='data')
    parser.add_argument('--prepath', type=str, default='')
    parser.add_argument('--prepath2', type=str, default='')
    
    
    parser.add_argument('--nThreads', type=int, default=0)

    parser.add_argument('--Decay', type=int, default=40000)
    parser.add_argument('--modelsave', type=int,default=5000)
    parser.add_argument('--Decay_rate', type=float,default=0.6)

    parser.add_argument('--Pretrain', type=bool,default=False)
    parser.add_argument('--Imcrementflag', type=bool,default=False)
    parser.add_argument('--Imcrement', type=int,default=10000)
    parser.add_argument('--Observe', type=int, default=3000)
    parser.add_argument('--Epochs', type=int, default=1000000)
    parser.add_argument('--Optimizee_Train_Steps', type=int, default=1000)
    parser.add_argument('--train_steps', type=int,default=100)
    parser.add_argument('--Gradient_threshold', type=float, default = 1e-6)



    parser.add_argument('--optimizer_lr', type=float, default=0.001)
    parser.add_argument('--hand_optimizer_lr', type=float, default=0.000001)
    parser.add_argument('--Neumann_alpha', type = float, default=0.1)
    parser.add_argument('--Neumann_series', type = float, default = 5)
    parser.add_argument('--observe_reset', type = int, default = 5000)
    args = parser.parse_args()
    return args