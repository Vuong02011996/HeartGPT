import torch
import torch.nn as nn
from torch.nn import functional as F
import scipy.io
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os


# Harry Davies 19_09_2024

# The following code is adapted from a tutorial by Andrej Kapathy, available at https://github.com/karpathy/ng-video-lecture
# The explaination behind this code and the model files can be found in the paper "Interpretable Pre-Trained Transformers for Heart Time-Series Data"
# available at https://arxiv.org/abs/2407.20775

eval_interval = 200 # 2000, sau bao nhieu epoch ites, thi danh gia loss
# save_interval = 10000 # 20000 #how often the model is checkpointed
eval_iters = 20  # 200 so lan data lap de danh gia loss
batch_size = 32 # sequences we process in parellel
max_iters = 500000# 1000000

block_size = 500 # this is context length
learning_rate = 3e-04
n_embd = 64 # 384 / 6 means every head is 64 dimensional
n_head = 8
n_layer = 8

# n_embd = 128
# n_head = 16
# n_layer = 16

dropout = 0.2


# GPU is necessary. Training of 8 head, 8 layer model and 500 context length was possible with 12GB VRAM
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#define vocab size. All data was scaled between 0 and 100 and rounded to nearest integer, giving 101 possible token values
# Q/A if don't convert signal  to 0-> 100, what the vocab_size
# vocab_size = 101
vocab_size = 1001

# out_features
num_classes = 3

path_model = '/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Model/'
path_save ='/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Data/Data_Study/'

types_beat = [0, 1, 1, 2, 2, 2, 2]
symbols = ['N','S','S', 'V', 'V', 'V', 'V']
split = 'train'
number_type_N = 500000
data = None
labels = None
for i, type_beat in enumerate(types_beat):
    all_windows = np.load(path_save + f'all_windows_{split}_{symbols[i]}.npy')
    all_labels = np.load(path_save + f'all_labels_{split}_{symbols[i]}.npy')
    print(f'Type_{symbols[i]} have {len(all_labels)} sample')
    if data is None:
        data = all_windows
        labels = all_labels
    else:
        if type_beat == 0:
            # Select 4000 random indices from all_windows
            print(f'Random {number_type_N} samples from Type_{symbols[i]}')
            random_indices = np.random.choice(all_windows.shape[0], number_type_N, replace=False)
            data = np.concatenate((data, all_windows[random_indices]))
            labels = np.concatenate((labels, all_labels[random_indices]))
        else:
            data = np.concatenate((data, all_windows))
            labels = np.concatenate((labels, all_labels))

# Generate a permutation of indices
indices = np.random.permutation(data.shape[0])
# Shuffle data and labels using the generated indices
data = data[indices]
labels = labels[indices]

def get_batch_ecg(split):

    data_batch = train_data  if split == 'train' else test_data
    labels_batch  = train_labels  if split == 'train' else test_labels
    ix = torch.randint(data_batch.shape[0], (batch_size,))
    x = torch.stack([torch.tensor(data_batch[i], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(labels_batch[i], dtype=torch.long) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
    # for split in ['train']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_ecg(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones((block_size,block_size)))) #buffer means not updated by optimiser
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #start = time.time()
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # square root headsize # (B, T, C) @ (B, C, T) = B, T, T
        # for every batch, we will now have a T by T matrix giving us the affinities of each token
        # wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))

        # the tril signifies a decoder block, future tokens cannot communicate with the past
        wei = F.softmax(wei, dim=-1)# all attention weights sum to 1 for updating a single token
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        #end = time.time()
        #print(start-end)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        # creating a list of head objects (turned into modules) resulting in a number of head modules
        # then assigns the list of modules to self.heads - these run in parellel
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) #projection generally matches sizes for adding in residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #concatenate the output of the different attention heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), #multiplication performed in attention is all you need paper
            # expands and contracts back down to projection
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # communication
        self.sa = MultiHeadAttention(n_head, head_size)
        # computation
        self.ffwd = FeedForward(n_embd)
        # layer norm
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


criterion=nn.CrossEntropyLoss()
# create heart GPT class
class HeartGPTModel(nn.Module):

    def __init__(self):
        super().__init__()
        # table needs to be vocab size by vocab size, to look up probability of next token given this token
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        # self.lm_head = nn.Linear(n_embd, vocab_size)
        self.lm_head = nn.Linear(n_embd, num_classes)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        #idx is batch, targets is time
        tok_emb = self.token_embedding_table(idx) #(B, T, vocab_size) which is batch, time, channel
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C (integers from 0 to T-1)
        x = tok_emb + pos_emb # B, T, C
        x = self.blocks(x) # B, T, C
        x = self.ln_f(x) # B, T, C

        logits = self.lm_head(x)
        #channel is vocab size, so in this case 65

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # C = 4  from lm_head
            # print("B, T, C = ", B, T, C)
            # logits = logits.view(B*T, C)
            """
            targets = targets.view(B*T)
              ^^^^^^^^^^^^^^^^^
            RuntimeError: shape '[32000]' is invalid for input of size 64
            """
            # targets = targets.view(-1)
            # targets = targets.view(-1, 1)
            # targets_one_hot = F.one_hot(targets, num_classes=4)

            # loss = F.cross_entropy(logits, targets)
            logits = logits.mean(dim=1)  # Shape now becomes (B, C)
            loss = criterion(logits, targets)

        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx (context) to the last block_size tokens because positional embeddings only has up to block size
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


if __name__ == '__main__':
    # model_path = "/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Model/Heatbeat_pretrained_64_8_8_500_1000_500_train_101.pth"
    model = HeartGPTModel()
    # model.load_state_dict(torch.load(model_path))
    m = model.to(device)
    # random loss at this point would be -log(1/65)

    # AdamW
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # counter the number of model parameters to be trained
    num_parameters = count_parameters(model)
    print(f"The model has {num_parameters} trainable parameters.")

    # Initialize KFold with 5 splits
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    loss_train_max = 10
    loss_test_max = 10

    # Iterate through each fold
    fold = 1
    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        print("Training on fold: ", fold)
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                # print(f"step {iter}: train loss {losses['train']:.4f}")

            # if iter % save_interval == 0 or iter == max_iters-1:
            # if iter == max_iters-1:
            # model_path for checkpointing
            if losses['val'] < loss_test_max and losses['train'] < loss_train_max:
                # Delete the previous model
                model_path = f"{path_model}Model_beat_classify_study_data_{n_embd}_{n_head}_{n_layer}_{block_size}_{max_iters}.pth"
                if os.path.exists(model_path):
                    os.remove(model_path)
                torch.save(model.state_dict(), model_path)
                loss_train_max = losses['train']
                loss_test_max = losses['val']

            # get batch
            x_batch, y_batch = get_batch_ecg('train')

            # loss evaluation
            logits, loss = m(x_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        fold += 1






