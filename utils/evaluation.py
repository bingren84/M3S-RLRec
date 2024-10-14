import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
def calculate_top_k_accuracies(args,user_grouped,testdataloader,  model,ks=[5, 10, 20, 40]):
    """
    计算多个Top-K准确率。
    :param test_data: 测试数据集
    :param ks: 一个包含多个K值的列表
    :param model: 训练好的模型
    :return: 各个Top-K准确率的字典
    """
    top_k_accuracies = {}
    hits = {k: 0 for k in ks}
    purchase= {k: 0 for k in ks}
    total_count = {k: 0 for k in ks}
    ndcg_hit={k: 0 for k in ks}
    ndcg_purchase={k: 0 for k in ks}
     
    with torch.no_grad(), tqdm(total=len(testdataloader.dataset), desc="正在评估") as pbar:
        for batch in tqdm(testdataloader, desc="evaluation", total=len(testdataloader)):
            user, states, actions, rewards, next_states, dones = batch['user'], batch['state'], batch['action'], batch['rewards'], batch['next_states'], batch['dones']
            states_lens = torch.full((states.shape[0],), states.shape[1])
            predicted_scores, _ = model(states, states_lens)

            all_movie_ids = torch.arange(predicted_scores.shape[1])
            # 初始化掩码列表------------------
            masked_scores = []

            for u, u_score, u_action in zip(user, predicted_scores, actions):
                u_cpu = u.cpu().item()
                user_watched = user_grouped.get_group(u_cpu)['MovieID'].values
                mask = torch.zeros_like(all_movie_ids, dtype=torch.bool).to(states.device)

                mask[user_watched] = True
                mask[u_action.item()] = False
                masked_score = u_score * (~mask).float()
                masked_scores.append(masked_score)

            masked_scores = torch.stack(masked_scores)
            #----------------

            _, top_k_indices = torch.topk(masked_scores, max(ks), dim=1)

            for k in ks:
                top_k_indices_per_batch = top_k_indices[:, :k]

                for sample_top_k_indices, sample_action, reward in zip(top_k_indices_per_batch, actions, rewards):
                    if sample_action.item() in sample_top_k_indices.cpu().numpy():
                        hits[k] += 1
                        index = np.where(sample_top_k_indices.cpu().numpy() == sample_action.item())[0][0]
                        ndcg = 1. / np.log2(index + 2)
                        ndcg_hit[k] += ndcg

                        if args.data == 'kaggle' and reward[0] == args.r_transaction:
                            purchase[k] += 1
                            ndcg_purchase[k] += ndcg

                    total_count[k] += 1

            pbar.update(1)
                
    # 计算并存储准确率
    for k in ks:
        top_k_accuracy = hits[k] / total_count[k] if total_count[k] > 0 else 0
        top_k_accuracies[f'Hit@{k}'] = top_k_accuracy
        top_k_accuracies[f'NDCG@{k}'] =  ndcg_hit[k] / total_count[k] if total_count[k] > 0 else 0
        if args.data=='kaggle':
            top_k_purchase = purchase[k] / hits[k]
            top_k_accuracies[f'purchase@{k}'] = top_k_purchase
            top_k_accuracies[f'NDCG_purchase@{k}'] =  ndcg_purchase[k] / purchase[k]
    
    return top_k_accuracies

