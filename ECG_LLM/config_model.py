import torch

# Harry Davies 19_09_2024

# The following code is adapted from a tutorial by Andrej Kapathy, available at https://github.com/karpathy/ng-video-lecture
# The explaination behind this code and the model files can be found in the paper "Interpretable Pre-Trained Transformers for Heart Time-Series Data"
# available at https://arxiv.org/abs/2407.20775

eval_interval = 200 # 2000, sau bao nhieu epoch ites, thi danh gia loss
# save_interval = 10000 # 20000 #how often the model is checkpointed
eval_iters = 100  # 200 so lan data lap de danh gia loss
batch_size = 8 # sequences we process in parellel
max_iters = 100000# 1000000

block_size = 1280 # this is context length
learning_rate = 3e-04
n_embd = 64 # 384 / 6 means every head is 64 dimensional
n_head = 8
n_layer = 8

# n_embd = 128
# n_head = 16
# n_layer = 16

dropout = 0.2
# dropout = 0.0


# GPU is necessary. Training of 8 head, 8 layer model and 500 context length was possible with 12GB VRAM
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#define vocab size. All data was scaled between 0 and 100 and rounded to nearest integer, giving 101 possible token values
# Q/A if don't convert signal  to 0-> 100, what the vocab_size
# vocab_size = 101
vocab_size = 512
# out_features
num_classes = 4