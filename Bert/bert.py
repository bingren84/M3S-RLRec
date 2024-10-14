from torch import nn as nn
from .attention import *
import math

from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch.nn.functional as F

class TransformerDQN(nn.Module):
    def __init__(self,num_items, state_dim, action_dim, nhead=1, num_encoder_layers=1, dim_feedforward=64,device=None,dropout=0.1):
        super(TransformerDQN, self).__init__()
        self.embedding = nn.Embedding(num_items, dim_feedforward)
        self.positional_encoding = nn.Embedding(state_dim, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead,dropout=dropout)#,dropout=0.1,norm_first=True)
        norm = nn.LayerNorm(dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers,norm=norm)
        #self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_feedforward, action_dim)
        self.fc2 = nn.Linear(dim_feedforward, action_dim)
        self.Qactivation = nn.ReLU()  # 定义激活函数

        self.device=device
        
    def extract_axis_1(self,data, ind):  
        """  
        从 PyTorch 张量的第一个轴上获取指定索引的元素。  
        :param data: 将被子集化的 PyTorch 张量。  
        :param ind: 要获取的索引（每个沿着 data 的第 0 轴的元素一个）。  
        :return: 子集化后的张量。  
        """  
        # 确保 ind 是一个一维张量  
        ind = torch.as_tensor(ind, dtype=torch.long, device=data.device)  
        
        # 获取 data 的第一个维度的大小  
        batch_size = data.size(0)  
        
        # 创建一个范围从 0 到 batch_size-1 的一维张量  
        batch_range = torch.arange(batch_size, device=data.device)  
        
        # 将 batch_range 和 ind 堆叠成一个二维张量，其中每一行表示一个索引元组  
        indices = torch.stack([batch_range, ind], dim=1)  
        
        # 使用这些索引从 data 中获取元素  
        res = data[indices[:, 0], indices[:, 1]]  
        
        return res
    
    def forward(self, x,len_state):
        batch_size, sequence_length = x.size()
        positions = torch.arange(sequence_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        embedded_positions = self.positional_encoding(positions)
        x = self.embedding(x) +embedded_positions
        
        #x=self.dropout(x)

        x = self.transformer_encoder(x)        
       # x=self.norm(x)
        x = x.mean(dim=1)  # 平均池化所有时间步
        #x=torch.sum(x, dim=1 )
        #x=self.extract_axis_1(x,len_state-1)
        
        return x
    def getAtion2(self,x,len_state):
        action_state=self.forward(x,len_state)
        logtic=self.fc(action_state)
        
        return logtic,action_state
    def getQ2(self,x,len_state):
        action_state=self.forward(x,len_state)    

        q=self.fc2(action_state)
        #q=self.Qactivation(q)
        return q,None
        
    def getQforAction2(self,action_state): 
        q=self.fc2(action_state)       
        return q,None
        
        

    
    
    
class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = BERTEmbedding(self.args)
        self.model = BERTModel(self.args)
        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.model.named_parameters():
                if not 'layer_norm' in n:
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)
        
    def forward(self, x):
        x1, mask = self.embedding(x)
        scores = self.model(x1, self.embedding.token.weight, mask)
        return scores


class BERTEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 2
        hidden = args.bert_hidden_units
        max_len = args.bert_max_len
        dropout = args.bert_dropout

        self.token = TokenEmbedding(
            vocab_size=vocab_size, embed_size=hidden)
        self.position = PositionalEmbedding(
            max_len=max_len, d_model=hidden)

        self.layer_norm = LayerNorm(features=hidden)
        self.dropout = nn.Dropout(p=dropout)

    def get_mask(self, x):
        if len(x.shape) > 2:
            x = torch.ones(x.shape[:2]).to(x.device)
        return (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

    def forward(self, x):
        mask = self.get_mask(x)
        if len(x.shape) > 2:
            pos = self.position(torch.ones(x.shape[:2]).to(x.device))
            x = torch.matmul(x, self.token.weight) + pos
        else:
            x = self.token(x) + self.position(x)
        return self.dropout(self.layer_norm(x)), mask


class BERTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden = args.bert_hidden_units
        heads = args.bert_num_heads
        head_size = args.bert_head_size
        dropout = args.bert_dropout
        attn_dropout = args.bert_attn_dropout
        layers = args.bert_num_blocks

        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            hidden, heads, head_size, hidden * 4, dropout, attn_dropout) for _ in range(layers)])
        self.linear = nn.Linear(hidden, hidden)
        self.bias = torch.nn.Parameter(torch.zeros(args.num_items + 2))
        self.bias.requires_grad = True
        self.activation = GELU()

    def forward(self, x, embedding_weight, mask):
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        x = self.activation(self.linear(x))
        scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias
        return scores
