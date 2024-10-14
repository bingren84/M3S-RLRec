import os
import pandas as pd
import tensorflow as tf
from utility import to_pickled_df, pad_history
import numpy as np
from tqdm import tqdm 
from sklearn.preprocessing import LabelEncoder  

class BuffData():
    def __init__(self,DATA='Retailrocket',sate_len=10):
        
        self.DATA=DATA
        if DATA=='Retailrocket':
            self.length=10
            data_directory='Retailrocket'            
            event_df = pd.read_csv(os.path.join(data_directory, 'events.csv'), header=0)
            event_df.columns = ['timestamp','session_id','behavior','item_id','transid']
            ###remove transid column
            event_df =event_df[event_df['transid'].isnull()]#？ 只有购买了才有值
            event_df = event_df.drop('transid',axis=1)
            ##########remove users with <=2 interactions
            event_df['valid_session'] = event_df.session_id.map(event_df.groupby('session_id')['item_id'].size() > 2)
            event_df = event_df.loc[event_df.valid_session].drop('valid_session', axis=1)
            ##########remove items with <=2 interactions
            event_df['valid_item'] = event_df.item_id.map(event_df.groupby('item_id')['session_id'].size() > 2)
            event_df = event_df.loc[event_df.valid_item].drop('valid_item', axis=1)
            ######## transform to ids
            item_encoder = LabelEncoder()
            session_encoder= LabelEncoder()
            behavior_encoder=LabelEncoder()
            event_df['behavior']=behavior_encoder.fit_transform(event_df.behavior)
            event_df['item_id'] = item_encoder.fit_transform(event_df.item_id)
            event_df['session_id'] = session_encoder.fit_transform(event_df.session_id)
            original_labels = behavior_encoder.classes_  
            encoded_values = behavior_encoder.transform(original_labels)             
            # 打印对应关系  
            for label, value in zip(original_labels, encoded_values):  
                print(f"原始标签: {label}, 编码后的值: {value}")
            
            ###########sorted by user and timestamp
            event_df['is_buy']=1-event_df['behavior']#将加入购物车认为是购买。为何？
            event_df = event_df.drop('behavior', axis=1)

        else:
            data_directory='MovieLens/ml-1m/'
            self.length=10
            event_df = pd.read_csv(os.path.join(data_directory, 'ratings.dat'), sep='::', names=['session_id', 'item_id', 'Rating', 'timestamp'])
            event_df = event_df.astype({'session_id': np.int16, 'item_id': np.int16, 'Rating': np.int16, 'timestamp': np.uint32})     
            #movies_df = pd.read_csv(os.path.join(data_directory, 'movies.dat'), sep='::', names=['item_id', 'Title', 'Genres'], encoding='latin-1')
            #movies_df['item_id'] = movies_df['item_id'].apply(pd.to_numeric)
            #print(movies_df.head(10))
            #ratings_df = ratings_df.groupby('UserID').filter(lambda x: len(x['MovieID']) > 10)
            item_encoder = LabelEncoder()
            user_encoder= LabelEncoder() 
            event_df['item_id'] = item_encoder.fit_transform(event_df.item_id)
            event_df['session_id'] = user_encoder.fit_transform(event_df.session_id)
            
        self.event_df=event_df
        self.data_directory=data_directory
        item_ids=event_df.item_id.unique()
        session_ids=event_df.session_id.unique()
        print("item数量:"+str(len(item_ids))+" 最大值："+str(np.max(event_df.item_id)))#70851he 70852可见itemID从0到最大连续
    
        print("user数量:"+str(len(session_ids))+" 最大值："+str(np.max(event_df.session_id)))#195524和195523
        print("交互数："+str(len(event_df)))

        # 按照 session_id 分组并计算每个会话的长度
        session_lengths = event_df.groupby('session_id').size()

        # 计算所有会话的平均长度
        average_length_across_sessions = session_lengths.mean()

        print("平均长度"+str(average_length_across_sessions))
        self.pad_item=len(item_ids)
            
    def split(self):       
        data_directory=self.data_directory
        event_df=self.event_df
        sorted_events = event_df.sort_values(by=['session_id', 'timestamp'])
        sorted_events.to_csv(self.data_directory+'/sorted_events.csv', index=None, header=True)
        to_pickled_df(data_directory,'sorted_events', sorted_events)
        self.sorted_events=sorted_events   
        
        total=sorted_events.session_id.unique()
        np.random.shuffle(total)
        fractions = np.array([0.8, 0.1, 0.1])
        # split into 3 parts
        train_ids, val_ids, test_ids = np.array_split(total, (fractions[:-1].cumsum() * len(total)).astype(int))

        train_sessions=sorted_events[sorted_events['session_id'].isin(train_ids)]
        val_sessions=sorted_events[sorted_events['session_id'].isin(val_ids)]
        test_sessions=sorted_events[sorted_events['session_id'].isin(test_ids)]

        to_pickled_df(data_directory, "sampled_train",train_sessions)
        to_pickled_df(data_directory, "sampled_val",val_sessions)
        to_pickled_df(data_directory,"sampled_test",test_sessions)
        self.sorted_events=sorted_events
        
            
    def prepareData(self,dataDF='sampled_train.df'):
        
        data_directory=self.data_directory
        train_sessions = pd.read_pickle(os.path.join(data_directory, dataDF))
        groups=train_sessions.groupby('session_id')
        ids=train_sessions.session_id.unique()

        userid,Rating,state, len_state, action, is_buy, next_state, len_next_state, is_done = [],[],[], [], [], [], [],[],[]
        length=self.length
        with tqdm(total=len(ids),desc="正在处理"+dataDF) as pbar:
            for id in ids:
                group=groups.get_group(id)
                history=[]
                for index, row in group.iterrows():
                    
                    s=list(history)
                    history.append(row['item_id'])                    
                    if len(s)<length:#<1可以解决能启动
                        
                        continue
                    len_state.append(length if len(s)>=length else 1 if len(s)==0 else len(s))
                    s=pad_history(s,length,self.pad_item)
                    a=row['item_id']
                    if 'Retailrocket' ==self.DATA:
                        is_b=row['is_buy']
                        is_buy.append(is_b)
                    elif 'ml-1m'==self.DATA:
                        Rating.append(row['Rating'])
                        userid.append(id)
                        
                    state.append(s)
                    action.append(a)
                    
                    #history.append(row['item_id'])
                    next_s=list(history)
                    len_next_state.append(length if len(next_s)>=length else 1 if len(next_s)==0 else len(next_s))
                    next_s=pad_history(next_s,length,self.pad_item)
                    next_state.append(next_s)
                    is_done.append(False)
                if len(is_done)>0:  
                    is_done[-1]=True
                
                pbar.update(1)
        if 'Retailrocket' in data_directory:
            dic={'state':state,'len_state':len_state,'action':action,'is_buy':is_buy,'next_state':next_state,'len_next_states':len_next_state,'is_done':is_done}
        else:
            dic={'userid':userid,'state':state,'len_state':len_state,'action':action,'Rating':Rating, 'next_state':next_state,'len_next_states':len_next_state,'is_done':is_done}
        reply_buffer=pd.DataFrame(data=dic)
        if "train" in dataDF: 
            to_pickled_df(data_directory, "train_replay_buffer-Step"+str(self.length),reply_buffer)
            dic={'state_size':[length],'item_num':[self.pad_item]}
            data_statis=pd.DataFrame(data=dic)
            to_pickled_df(data_directory,"data_statis-"+str(self.length),data_statis)
        elif "test" in dataDF: 
            to_pickled_df(data_directory, "test_replay_buffer-Step"+str(self.length),reply_buffer)
        else:
            to_pickled_df(data_directory, "val_replay_buffer-Step"+str(self.length),reply_buffer)



        
    def pop(self):
        DATA_DIR=self.data_directory
        replay_buffer_behavior = pd.read_pickle(os.path.join(DATA_DIR, 'sorted_events.df'))
        total_actions=replay_buffer_behavior.shape[0]
        pop_dict={}
        for index, row in replay_buffer_behavior.iterrows():
            action=row['item_id']
            if action in pop_dict:
                pop_dict[action]+=1
            else:
                pop_dict[action]=1
            if index%100000==0:
                print (index/100000)
        for key in pop_dict:
            pop_dict[key]=float(pop_dict[key])/float(total_actions)
        f = open(DATA_DIR+'/pop_dict.txt', 'w')
        f.write(str(pop_dict))
        f.close()
if __name__ == '__main__':
    #pro=BuffData(DATA='Retailrocket')#Retailrocket
    pro=BuffData(DATA='ml-1m')
    pro.split()
    pro.prepareData('sampled_train.df') 
    pro.prepareData('sampled_val.df')
    
    #pro.prepareData('sampled_val.df')

    pro.pop()


