import numpy as np
import random
import torch

#from datasets import DATASETS
#from model import *
import argparse


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_template(args):
    args.min_uc = 5
    args.min_sc = 5
    args.split = 'leave_one_out'
    dataset_code = {'1': 'ml-1m', '20': 'ml-20m', 'b': 'beauty', 'bd': 'beauty_dense' , 'g': 'games', 's': 'steam', 'y': 'yoochoose'}
    args.dataset_code = dataset_code['1']#dataset_code[input('Input 1 / 20 for movielens, b for beauty, bd for dense beauty, g for games, s for steam and y for yoochoose: ')]
    if args.dataset_code == 'ml-1m':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.1
        args.bert_attn_dropout = 0.1
        args.bert_max_len = 200
        args.bert_mask_prob = 0.2
        args.bert_max_predictions = 40
    elif args.dataset_code == 'ml-20m':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.1
        args.bert_attn_dropout = 0.1
        args.bert_max_len = 200
        args.bert_mask_prob = 0.2
        args.bert_max_predictions = 20
    elif args.dataset_code in ['beauty', 'beauty_dense']:
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.5
        args.bert_attn_dropout = 0.2
        args.bert_max_len = 50
        args.bert_mask_prob = 0.6
        args.bert_max_predictions = 30
    elif args.dataset_code == 'games':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.5
        args.bert_attn_dropout = 0.5
        args.bert_max_len = 50
        args.bert_mask_prob = 0.5
        args.bert_max_predictions = 25
    elif args.dataset_code == 'steam':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.2
        args.bert_attn_dropout = 0.2
        args.bert_max_len = 50
        args.bert_mask_prob = 0.4
        args.bert_max_predictions = 20
    elif args.dataset_code == 'yoochoose':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 256
        args.bert_dropout = 0.2
        args.bert_attn_dropout = 0.2
        args.bert_max_len = 50
        args.bert_mask_prob = 0.4
        args.bert_max_predictions = 20

    batch = 128
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch
    args.train_negative_sampler_code = 'random'
    args.train_negative_sample_size = 0
    args.train_negative_sampling_seed = 0
    args.test_negative_sampler_code = 'random'
    args.test_negative_sample_size = 100
    args.test_negative_sampling_seed = 98765


parser = argparse.ArgumentParser()



################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
# optimizer & lr#
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'])
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=100)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=True)
parser.add_argument('--warmup_steps', type=int, default=100)
# epochs #
parser.add_argument('--num_epochs', type=int, default=100)
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20])
parser.add_argument('--best_metric', type=str, default='NDCG@10')

################
# Model
################
parser.add_argument('--model_code', type=str, default='bert', choices=['bert', 'sas', 'narm'])
# BERT specs, used for SASRec and NARM as well #
parser.add_argument('--bert_max_len', type=int, default=None)
parser.add_argument('--bert_hidden_units', type=int, default=64)
parser.add_argument('--bert_num_blocks', type=int, default=2)
parser.add_argument('--bert_num_heads', type=int, default=2)
parser.add_argument('--bert_head_size', type=int, default=32)
parser.add_argument('--bert_dropout', type=float, default=0.1)
parser.add_argument('--bert_attn_dropout', type=float, default=0.1)
parser.add_argument('--bert_mask_prob', type=float, default=0.2)

################
# Distillation & Retraining
################
parser.add_argument('--num_generated_seqs', type=int, default=3000)
parser.add_argument('--num_original_seqs', type=int, default=0)
parser.add_argument('--num_poisoned_seqs', type=int, default=100)
parser.add_argument('--num_alter_items', type=int, default=10)

################


args = parser.parse_args()


