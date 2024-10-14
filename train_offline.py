import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder 
from datetime import datetime
import torch.nn as nn
import os 
from Bert.bert import TransformerDQN
#from Bert.utils import *
from utils.dataset import MovieRatingDataset   
from utils.evaluation import calculate_top_k_accuracies
import argparse
from torch.utils.data import DataLoader 
from torch.nn import functional as F


def compute_lambda_return_torch(rewards_tensor, next_states_tensor, dones_tensor,main_network, target_network, gamma, lambda_, max_n):
    with torch.no_grad():
        # 确保输入张量的维度一致，如果需要
        #assert rewards_tensor.shape == next_states_tensor.shape[:-1] and dones_tensor.shape == rewards_tensor.shape, "Input tensors dimensions mismatch."
        #assert rewards_tensor.shape == next_states_tensor[:-1].shape and dones_tensor.shape == rewards_tensor.shape, "Input tensors dimensions mismatch."
        
        # 初始化张量以存储n步回报
        n_step_returns = torch.zeros_like(rewards_tensor, dtype=torch.float32)
        superloss=0
        # 计算每一步的n步回报，利用张量操作加速
        for n in range(0,max_n):# min(max_n, rewards_tensor.shape[-1])-1):
            discounts = gamma ** torch.arange(n+1, device=rewards_tensor.device, dtype=torch.float32)
            discounted_rewards = discounts * rewards_tensor[..., :n+1]#R
            # 修正索引以正确处理next_states_tensor
            next_state=next_states_tensor[ n,...]
            states_lens=torch.full((next_state.shape[0],), next_state.shape[1])
            next_q_v,_=target_network.getQ2(next_state,states_lens)


            _select,_=main_network.getQ2(next_state,states_lens)
            act_select=torch.argmax(_select,dim=1)
            max_next_q_v=next_q_v.gather(1,act_select.unsqueeze(1)).squeeze(1)
            if n+1!=max_n:
                logtic,actionState = target_network.getAtion2(next_state,states_lens)
                action_n=next_states_tensor[ n+1,...][:,0]
                superloss+= supervisedCriterion(logtic,action_n)     

            #terminal_values = (1-dones_tensor[..., n]) * (gamma**n * next_q_v.max(dim=-1)[0])#gama*V(St+1)
            terminal_values = (1-dones_tensor[..., n]) * (gamma**n * max_next_q_v)#gama*V(St+1)
            n_step_returns[..., n] = discounted_rewards.sum(dim=-1) + terminal_values
        
        # 计算Lambda-return
        #lambda_returns = (1 - lambda_) * torch.sum(lambda_ ** torch.arange(1, max_n + 1, device=rewards_tensor.device, dtype=torch.float32)[:, None] * n_step_returns, dim=0)
        lambda_returns = (1 - lambda_) * torch.sum(lambda_ ** torch.arange(1, max_n ,device=rewards_tensor.device, dtype=torch.float32) * n_step_returns[...,:max_n-1], dim=1)
        truncated=lambda_**max_n*n_step_returns[...,-1]
        lambda_returns+=truncated
    return lambda_returns,superloss

# 采样经验
def sample_experiences(experience_pool, batch_size):
    return random.sample(experience_pool, batch_size)
 



def train(args):   
    epoch=0
    
    #accuracies = calculate_top_k_accuracies(args,user_grouped,testdataloader, policy_network.getAtion,ks=[5, 10, 20, 40])

    
    with tqdm(total=epochs,desc="正在训练"+str(epoch)+"/"+str(epochs)+"@"+str(args.loop)) as pbar:
        for epoch in range(epochs):        
            totalLoss=[]   
            times=0         
            for batch in tqdm(Traindataloader, desc="Training"+str(epoch)+"@"+str(args.loop), total=len(Traindataloader)):
                pointer = np.random.randint(0, 2)
                if pointer == 0:
                    mainQN = policy_network
                    target_QN = target_network
                    optimizer=policy_optimizer
                else:
                    mainQN = target_network
                    target_QN = policy_network
                    optimizer=target_optimizer
                #for states, actions, rewards, next_states, dones in data_loader:
                #batch = sample_experiences(experience_pool, batch_size)
                 
                #states,states_lens, actions, rewards, next_states, dones =batch['state'],batch['state_lens'],batch['action'],batch['rewards'],batch['next_states'],batch['dones']# zip(*batch) 
                states, actions, rewards, next_states, dones =batch['state'],batch['action'],batch['rewards'],batch['next_states'],batch['dones']# zip(*batch) 
                
                states_lens=torch.full((states.shape[0],), states.shape[1])
                if epoch<stage:                
                    # = torch.tensor(states, dtype=torch.int64) 
                    #actions = torch.tensor(actions, dtype=torch.int64)#.unsqueeze(1)
                    logstic,action_state  = mainQN.getAtio2n(states,states_lens)                
                    loss= supervisedCriterion(logstic,actions)       
                else:    
                    #states = torch.tensor(states, dtype=torch.int64) 
                    #actions = torch.tensor(actions, dtype=torch.int64)#.unsqueeze(1)                    
                    logstic,action_state = mainQN.getAtion2(states,states_lens)     
                    loss1= supervisedCriterion(logstic,actions)   
                    # = torch.tensor(states, dtype=torch.int64)
                    #actions = torch.tensor(actions, dtype=torch.int64)                    
                    #rewards = torch.tensor(rewards, dtype=torch.float32)                    
                    #next_states = torch.tensor(next_states, dtype=torch.float32)
                    #dones = torch.tensor(dones, dtype=torch.float32)
                    dones=dones.to(torch.int32)                    
                    # 前向传播
                    #q_values = policy_network.getQ(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                    #q_values = policy_network.getQforAction(actionState).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                    q_values,_=mainQN.getQforAction2(action_state)
                    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)                    
                    # 计算Lambda Return
                    if multity:
                        next_states_tensor = torch.stack([ns for ns in next_states])
                        #next_states_tensor = torch.stack([torch.tensor(ns, dtype=torch.int32) for ns in next_states])
                        next_states_tensor=next_states_tensor.permute(1, 0, 2)                        
                        lambda_returns,superloss = compute_lambda_return_torch(rewards, next_states_tensor, dones,mainQN, target_QN, gamma, lambda_, args.n_steps)
                        Qtarget=lambda_returns
                         
                        #neg sample
                        negative_actions=[]
                        for index in range(states.shape[0]):                    
                            negative_list = []
                            negative_actions_list = [] 
                            
                            for _ in range(args.neg):  # Using underscore (_) as the loop variable since i is not used
                                neg_action = np.random.randint(item_num)
                                while neg_action == actions[index] or neg_action in negative_list:
                                    neg_action = np.random.randint(item_num)                        
                                negative_actions_list.append(neg_action)   
                                #negative_reward.append(-1.0)
                            negative_actions.append(negative_actions_list)     
                       
                        neg_reward = torch.full((states.shape[0],), args.neg, dtype=torch.float32).to(device).unsqueeze(-1)
                        neg_done = torch.full((states.shape[0],), 0, dtype=torch.int32).to(device).unsqueeze(-1) 
                        next_state = next_states[:, 0, :].clone().detach().to(torch.int32).unsqueeze(0)  
                        neg_lambda_returns,superloss = compute_lambda_return_torch(neg_reward, next_state, neg_done,mainQN, target_QN, gamma, lambda_,1)
                        neg_loss=torch.mean((q_value - neg_lambda_returns) ** 2)
                                   
                         ############## 
                    else:
                        reward=rewards[:,0].clone().detach().to(torch.int32)
                        next_state = next_states[:, 0, :].clone().detach().to(torch.int32)  
                        
                        #next_q=target_network.getQ(next_state).max(dim=-1)[0]                       
                        #max_q_values,_ =torch.max(target_network.getQ(next_state),dim=-1) 
                        max_q_values,_ =torch.max(policy_network.getQ(next_state),dim=-1)                         
                        
                        dones=dones.squeeze(-1)
                        Qtarget=max_q_values*gamma*(1-dones)+reward      


                                      
                    
                    # 计算损失
                    loss2 = torch.mean((q_value - Qtarget) ** 2)
                    loss2+=neg_loss
                     
                    if multity:
                        loss2+=superloss
                    #loss2 = F.smooth_l1_loss(positive_q_value, positive_target_q_value)
                    if needQ and needSupvision:
                        #loss=weight*loss1+(1-weight)*loss2         
                        loss=weight*loss1+(1-weight)*loss2                
                    if   not needSupvision and needQ: 
                        loss=loss2         
                    if needSupvision and not needQ  : 
                        loss=loss1  
                 
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                

                times+=1
                    # 假设test_data已经定义
                #if (epoch) % 100 == 0:
                totalLoss.append(loss.detach().cpu().numpy())
                                # 定期更新目标网络（例如每10个epoch）
                #if (times) % 20 == 0:
                #    target_network.load_state_dict(policy_network.state_dict()) 
            #target_network.load_state_dict(policy_network.state_dict())         
            logwriter.add_scalar("loss",np.mean(totalLoss),epoch)
            totalLoss=[]
            policy_network.eval()
            with torch.no_grad():
                if not needSupvision:
                   accuracies = calculate_top_k_accuracies(args,user_grouped,testdataloader, policy_network.getQ2,ks=[5, 10, 20, 40])
                 
                else:
                   accuracies = calculate_top_k_accuracies(args,user_grouped,testdataloader, policy_network.getAtion2,ks=[5, 10, 20, 40])
                
                for key, value in accuracies.items():
                    print(f"{key} Accuracy: {value}")
                    logwriter.add_scalar(key,value,epoch)
                '''
                accuracies_step=evaluateNstep(args,  policy_network.getAtion,device,ks=[5, 10, 20, 40])
                for key, value in accuracies_step.items():
                    print(f"{key} Accuracy: {value}")
                    logwriter.add_scalar(key,value,epoch)
                '''
                logwriter.close()    
                pbar.update(1)

    print("Training completed.")

if __name__ == "__main__": 
    
    data='ml'

    parser = argparse.ArgumentParser(description="Run nive multity step q learning.")
    parser.add_argument("--data", type=str,default=data, help="gpu parameter") 
    parser.add_argument("--loop", type=int, default=0,help="Loop parameter")  
    parser.add_argument("--gpu", type=int,default=-1, help="gpu parameter") 
    
    parser.add_argument("--weight", type=float, default=0.8,help="Loop parameter")  
    parser.add_argument("--needQ",  type=bool, default=True,help="needQ")  
    parser.add_argument("--needSupvision", type=bool, default=True,help="needSupvision")  
    parser.add_argument("--stage", type=int, default=0,help="强化损失开始的阶段")  
    parser.add_argument("--multity", type=bool, default=True,help="multity")  

    parser.add_argument("--neg", type=int, default=10,help="neg") 
    #parser.add_argument("--data", type=str, default=data_directory,help="data_directory") 

    parser.add_argument("--batch_size", type=int, default=2048,help="batch_size") 
    
    parser.add_argument('--r_click', type=float, default=1,
                        help='reward for the click behavior.')
    parser.add_argument('--r_Add', type=float, default=3,
                        help='reward for the purchase behavior.')    
    parser.add_argument('--r_transaction', type=float, default=5,
                        help='reward for the purchase behavior.')  
    if data=='ml':
        parser.add_argument("--n_steps", type=int, default=6,help="n_steps") 
        data_directory = "../../DataSet/MovieLens/ml-1m/"
        parser.add_argument("--state_size", type=int, default=10,help=" ") 
    else:
        parser.add_argument("--state_size", type=int, default=100,help=" ") 
        data_directory = "../../DataSet/Retailrocket" 
        parser.add_argument("--n_steps", type=int, default=6,help="n_steps") 
    parser.add_argument("--datadir", type=str, default=data_directory,help="data_directory") 

    args = parser.parse_args()
    
    
    epochs = 60
    n_steps =args.n_steps# 5  # TD的n步
        
    multity=args.multity  
    stage=args.stage
    weight=args.weight#0.8
    needQ=args.needQ#True
    needSupvision=args.needSupvision#False

    if torch.cuda.is_available():             
        if args.gpu==-1:
            print(torch.cuda.device_count())
            gpu=input("input gpu:")       
        else:
            gpu=args.gpu    
        #os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        #import os
        #print(os.environ.get('CUDA_VISIBLE_DEVICES'))
        device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu") 
        #device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("No GPU")
        
    # 超参数
    gamma = 0.5
    lambda_ = 0.8
    alpha = 0.001
    batch_size = args.batch_size
    state_dim =args.state_size  # 历史轨迹的长度

    target_update_frequency = 10


   
    if args.data=='ml':
        # 读取ML-1M数据
        ratings = pd.read_csv(data_directory+'ratings.dat', sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

        item_encoder = LabelEncoder()
        user_encoder= LabelEncoder()
        ratings['MovieID'] = item_encoder.fit_transform(ratings.MovieID)
        ratings['UserID'] = user_encoder.fit_transform(ratings.UserID)


    else:
        
        ratings = pd.read_csv(os.path.join(data_directory, 'events.csv'), header=0)
        ratings.columns = ['Timestamp','UserID','behavior','MovieID','transid']#['Timestamp','session_id','behavior','item_id','transid']
        
        view_count = (ratings['behavior'] == 'view').sum()
        # 统计 behavior 字段等于 'addtocart' 的数量
        addtocart_count = (ratings['behavior'] == 'addtocart').sum()
        transaction_count = (ratings['behavior'] == 'transaction').sum()
        #ratings = pd.read_csv(data_directory+'ratings.dat', sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        
        item_encoder = LabelEncoder()
        user_encoder= LabelEncoder()
        ratings['MovieID'] = item_encoder.fit_transform(ratings.MovieID)
        ratings['UserID'] = user_encoder.fit_transform(ratings.UserID)
        '''
        behavior_encoder=LabelEncoder()
        ratings['behavior']=behavior_encoder.fit_transform(ratings.behavior)

       
        original_labels = behavior_encoder.classes_  
        encoded_values = behavior_encoder.transform(original_labels)             
        # 打印对应关系  
        for label, value in zip(original_labels, encoded_values):  
            print(f"原始标签: {label}, 编码后的值: {value}")
        '''


        #ratings['is_buy']=1-ratings['behavior']#将加入购物车认为是购买。为何？
        #ratings = ratings.drop('behavior', axis=1)
    #保留消费历史    
    consume=ratings[['UserID','MovieID']]
    user_grouped =consume.groupby('UserID')
     

    from tensorboardX import SummaryWriter
    time_string=datetime.now().strftime('%Y-%m-%d-%H-%M-%S') 
    #twoStage:第一阶段有监督，第二阶段是两部分损失之和
    #one没有第一阶段，只有两部分损失和
    #only1，只有强化损失
    sub="/stage@"+str(stage)+"+supvision@"+str(needSupvision)+"+TD@"+str(needQ)
    if needQ and needSupvision: 
        logwriter = SummaryWriter( "/logs1008/"+args.data+"/dim@"+str(state_dim)+"/step@"+str(args.n_steps)+sub+'/'+time_string)
    if needQ and not needSupvision: 
        stepname="n-setp" if multity else "oneStep"
        logwriter = SummaryWriter( "/logs1008/"+args.data+"/dim"+str(state_dim)+"@"+stepname+"/onlyQ/"+time_string)   
    if not needQ and needSupvision: 
        logwriter = SummaryWriter( "/logs1008/"+args.data+"/dim"+str(state_dim)+"@onlyBert/"+time_string)   
                
    arg_dict = vars(args) 
    logwriter.add_hparams(arg_dict,{})   
     
    user_ids = ratings['UserID'].unique()

    item_num = ratings['MovieID'].nunique()  # 动作空间与状态空间相同

    dim_feedforward=64
    num_head=1
    num_encoder_layers=1
    dropout_rate=0.1
    learning_rate=0.001
    np.random.shuffle(user_ids)
    fractions = np.array([0.8, 0.1, 0.1])
    # split into 3 parts
    train_ids, val_ids, test_ids = np.array_split(user_ids, (fractions[:-1].cumsum() * len(user_ids)).astype(int))
    #set_template(args)


    policy_network = TransformerDQN( item_num+1,state_dim, item_num,nhead=num_head, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward).to(device)
    target_network = TransformerDQN( item_num+1,state_dim, item_num,nhead=num_head, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward).to(device)
    target_network.load_state_dict(policy_network.state_dict())

    target_optimizer = optim.Adam(target_network.parameters(),lr=learning_rate)
    policy_optimizer=optim.Adam(policy_network.parameters(),lr=learning_rate)

    supervisedCriterion = nn.CrossEntropyLoss()
        
    neg=10
    # 定义超参数
    hparams = {
        'stage':stage,
        'state_step':state_dim,
        'learning_rate':learning_rate,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'dim_feedforward':dim_feedforward,
        'num_head':num_head,
        'num_encoder_layers':num_encoder_layers,
        'gamma':gamma,
        'neg':neg,
        'weight':weight,
         
        'tag':'采用了负样本，不用监督，看看效果'
        
    }
    hyperparameters_str = ', '.join(['%s: %s' % (key, value) for key, value in hparams.items()])
    logwriter.add_text('Super Parameters', hyperparameters_str, global_step=0) 
    

    # 生成经验池
    '''
    name=str(n_steps)+'multiple_Trainbuffer.df'
    experience_pool = create_experience(ratings,train_ids, state_dim, n_steps,name) 
    name=str(n_steps)+'multiple_Testbuffer.df'
    test_pool = create_experience(ratings,test_ids, state_dim, n_steps,name) 
    '''
    name="(STEP@"+str(args.state_size)+")"+str(n_steps)+'multiple_Trainbuffer.df'
    Traindataset = MovieRatingDataset(args, ratings, state_dim,user_ids=train_ids, n_steps=n_steps,dirName=data_directory, filename=name,device=device)

    # 使用DataLoader加载数据
    Traindataloader = DataLoader(Traindataset, args.batch_size, shuffle=True, num_workers=0)
    name="(STEP@"+str(args.state_size)+")"+str(n_steps)+'multiple_Testbuffer.df'

    testdataset = MovieRatingDataset(args, ratings, state_dim,user_ids=test_ids, n_steps=n_steps,dirName=data_directory, filename=name,device=device)

    # 使用DataLoader加载数据
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    

    train(args)
