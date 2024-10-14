import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm 

'''

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist
     
def create_experience(ratings, user_ids,state_dim, n_steps,filename):
    experiences = []
    #user_ids = ratings['UserID'].unique()
    tt=0
    
    if os.path.exists(os.path.join(data_directory,filename)):
        #return pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
        with open(os.path.join(data_directory,filename ), 'rb') as handle:
            loaded_data = pickle.load(handle)
            return loaded_data  
        
    with tqdm(total=len(user_ids),desc="正在处理") as pbar:
        for user_id in user_ids:
             
            if tt>20:
                break
            tt+=1
            
           
            user_data  = ratings[ratings['UserID'] == user_id].sort_values(by='Timestamp')#['Rating'].values
            for i in range(len(user_data) - state_dim - n_steps + 1):  # 确保可以生成n个next_state
                
                totol=user_data.iloc[i:]['MovieID'].tolist() 
                current_state=user_data.iloc[i:i + state_dim]['MovieID'].tolist() 
                action = user_data.iloc[i + state_dim]['MovieID'] 
            
                rewards = user_data.iloc[i + state_dim:i + state_dim + n_steps]['Rating'].tolist()
            
                next_states = [user_data.iloc[i +  step:i +  step + state_dim]['MovieID'].tolist() 
                            for step in range(1, n_steps + 1)]  
                # 终止状态判断逻辑不变
                dones = [False] * (n_steps - 1) + [True] if i + state_dim + n_steps == len(user_data) else [False] * n_steps 
        
                experiences.append((current_state, action, rewards, next_states, dones)) 
            pbar.update(1)
    
    with open(os.path.join(data_directory, filename), 'wb') as handle:
        pickle.dump(experiences, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return experiences
'''
class MovieRatingDataset(Dataset):
    def __init__(self,args, ratings_df, state_dim, n_steps, user_ids=None,dirName=None, filename=None,device=None):
 
        self.device = device
        self.name=filename#str(n_steps)+'multiple_Trainbuffer.df'
        self.data_directory =dirName# "../DataSet/MovieLens/ml-1m/"
        self.ratings_df = ratings_df.sort_values(by='Timestamp')
        self.state_dim = state_dim
        self.n_steps = n_steps
        if user_ids is None:
            self.user_ids = ratings_df['UserID'].unique()
        else:
            self.user_ids = user_ids
        self.experiences = self._generate_experiences(args)
    
    def _generate_experiences(self,args):
        experiences = []
        if os.path.exists(os.path.join(self.data_directory,self.name)):
            #return pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
            with open(os.path.join(self.data_directory,self.name ), 'rb') as handle:
                experiences = pickle.load(handle)
                return experiences 
        else:
            with tqdm(total=len(self.user_ids),desc="正在处理") as pbar: 
                for user_id in self.user_ids:
                    #print(len(self.ratings_df[self.ratings_df['UserID'] == 1].sort_values(by='Timestamp')))
                    user_data = self.ratings_df[self.ratings_df['UserID'] == user_id].sort_values(by='Timestamp')
                    
                    for i in range(len(user_data) - self.state_dim - self.n_steps + 1):#len(user_data)- self.n_steps能索引的位置：len-1+step+1；因为range，所以+1                 
                        current_state = user_data.iloc[i:i + self.state_dim]['MovieID'].tolist()
                        action = user_data.iloc[i + self.state_dim]['MovieID']
                        if args.data=='ml':                            
                            rewards = user_data.iloc[i + self.state_dim:i + self.state_dim + self.n_steps]['Rating'].tolist()
                        else:
                            behavior=user_data.iloc[i + self.state_dim:i + self.state_dim + self.n_steps]['behavior'].tolist()
                            
                            rewards = []
                            for evt in behavior:
                                if evt == 'view' :
                                    rewards.append(args.r_click)
                                elif   evt == 'addtocart':
                                    rewards.append(args.r_Add)
                                elif evt == 'transaction':
                                    rewards.append(args.r_transaction)
                                else:
                                    raise ValueError("error behavior")
                             
                            #rewards = user_data.iloc[i + self.state_dim:i + self.state_dim + self.n_steps]['Rating'].tolist()
                        next_states =[]
                        dones=[]
                        #print(i)
                        if i==len(user_data) - self.state_dim - self.n_steps :
                            #print('end')
                            pass
                        for step in range(1, self.n_steps + 1):                        
                            #next_states.append(user_data.iloc[i + step:i + step + self.state_dim]['MovieID'].tolist())
                            next_states.append(user_data.iloc[i + step:i + step + self.state_dim]['MovieID'].tolist())
                            if i + self.state_dim + self.n_steps == len(user_data):                      
                                dones.append(True)
                            else:
                                dones.append(False)
                        #[user_data.iloc[i + step:i + step + self.state_dim]['MovieID'].tolist() for step in range(1, self.n_steps + 1)]
                        #dones = [False] * (self.n_steps - 1) + [True] if i + self.state_dim + self.n_steps == len(user_data) else [False] * self.n_steps                    
                        experiences.append((user_id,current_state, action, rewards, next_states, dones))
                    pbar.update(1)
                    pass
            
            
            with open(os.path.join(self.data_directory, self.name), 'wb') as handle:
                pickle.dump(experiences, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return experiences
        
        
    
    def __len__(self):
        return len(self.experiences)
    
    def __getitem__(self, idx):
        user, current_state, action, rewards, next_states, dones = self.experiences[idx]
        # 可以考虑将数据转换成Tensor或其他适合模型输入的格式
        return {
            'user': torch.tensor(user, dtype=torch.long).to(self.device),
            'state': torch.tensor(current_state, dtype=torch.long).to(self.device),
            'action': torch.tensor(action, dtype=torch.long).to(self.device),
            'rewards': torch.tensor(rewards, dtype=torch.float).to(self.device),
            'next_states': torch.tensor(next_states, dtype=torch.long).to(self.device),#.view(-1),  # Flatten for simplicity
            'dones': torch.tensor(dones, dtype=torch.bool).to(self.device)
        }
        
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
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder 
if __name__ == "__main__": 
    state_dim=10
    n_steps=3
    name=str(n_steps)+'multiple_Trainbuffer.df' 
    data_directory = "../../DataSet/MovieLens/ml-1m/"
  
    # 读取ML-1M数据
    ratings = pd.read_csv(data_directory+'ratings.dat', sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

    item_encoder = LabelEncoder()
    user_encoder= LabelEncoder()
    ratings['MovieID'] = item_encoder.fit_transform(ratings.MovieID)
    ratings['UserID'] = user_encoder.fit_transform(ratings.UserID)

    from tensorboardX import SummaryWriter
    time_string=datetime.now().strftime('%Y-%m-%d-%H-%M-%S') 

    logwriter = SummaryWriter( "/logs/Mutity/"+time_string)
                    
    alluser_ids = ratings['UserID'].unique()

    action_dim = ratings['MovieID'].nunique()  # 动作空间与状态空间相同

    np.random.shuffle(alluser_ids)
    fractions = np.array([0.8, 0.1, 0.1])
    # split into 3 parts
    train_ids, val_ids, test_ids = np.array_split(alluser_ids, (fractions[:-1].cumsum() * len(alluser_ids)).astype(int))

    dataset = MovieRatingDataset( ratings, state_dim,user_ids=train_ids, n_steps=n_steps,dirName=data_directory, filename=name)

    # 使用DataLoader加载数据
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    # 现在你可以迭代dataloader来获取数据批次
    for batch in dataloader:
        # 训练你的模型...
        pass