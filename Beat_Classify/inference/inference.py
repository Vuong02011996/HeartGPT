import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# The following code is adapted from a tutorial by Andrej Kapathy, available at https://github.com/karpathy/nanoGPT
# The explaination behind this code and the model files can be found in the paper "Interpretable Pre-Trained Transformers for Heart Time-Series Data"
# available at https://arxiv.org/abs/2407.20775

model_config = 'ECG_PT' #switch between 'ECG_PT' and 'PPG_PT'

block_size = 500 # this is context length
n_embd = 64
n_head = 8
n_layer = 8

# n_embd = 128
# n_head = 16
# n_layer = 16

dropout = 0.2
num_classes = 4

model_path = "/Model/Heatbeat_pretrained_64_8_8_500_100000_99999_train.pth"
# model_path = "/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Model/Heatbeat_pretrained_128_16_16_500_100_99_train_222.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


vocab_size = 101 # (0 - 100)


def tokenize_biosignal(data):

    # Get the shape of the data
    shape = data.shape

    # If the data is a column vector, reshape it to a row vector
    if len(shape) > 1 and shape[0] > shape[1]:
        data = data.T

    # If there are more than 500 data points, select the last 500
    if data.shape[1] > 500:
        data = data[:, -500:]

    # Scale the values between 0 and 1
    data_min = np.min(data)
    data_max = np.max(data)
    data_scaled = (data - data_min) / (data_max - data_min)

    # Multiply by 100
    data_scaled *= 100

    # Round to the nearest integer
    data_rounded = np.round(data_scaled)

    return data_rounded

#model definition
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones((block_size,block_size)))) #buffer means not updated by optimiser
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention weights
        wei = q @ k.transpose(-2, -1) * C**-0.5 # square root headsize # (B, T, C) @ (B, C, T) = B, T, T
        # for every batch, we will now have a T by T matrix giving us the affinities of each token
        # wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        # the tril signifies a decoder block, future tokens cannot communicate with the past
        wei = F.softmax(wei, dim=-1)# weights corresponding to the update of each token sum to 1

        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
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


# define the main heart_GPT model class
class Heart_GPT_Model(nn.Module):

    def __init__(self):
        super().__init__()

        # table needs to be vocab size by vocab size, to look up probability of next token given this token
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
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
        #channel is vocab size, so in this case 102 or 101

        if targets is None:
            loss = None
        # else:
        #     B, T, C = logits.shape
        #     logits = logits.view(B*T, C)
        #     targets = targets.view(B*T)
        #     loss = F.cross_entropy(logits, targets)

        else:
            B, T, C = logits.shape # C = 4 from lm_head
            # print("B, T, C = ", B, T, C)
            # logits = logits.view(B*T, C)
            """
            targets = targets.view(B*T)
              ^^^^^^^^^^^^^^^^^
            RuntimeError: shape '[32000]' is invalid for input of size 64
            """
            # targets = targets.view(B*T)
            targets_one_hot = F.one_hot(targets, num_classes=4)
            loss = F.cross_entropy(logits, targets_one_hot)
        
        return logits, loss

    def classified(self, idx):
        # idx is (B, T) array of indices in the current context
        # crop idx (context) to the last block_size tokens because positional embeddings only has up to block size
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits, loss = self(idx_cond)
        logits = logits.cpu().detach().numpy()
        # focus only on the last time step
        # logits = logits[:, -1, :] # (B, T, C)becomes (B, C)
        # middle_index = logits.shape[1] // 2 - 1
        # logits = logits[:, middle_index, :] # (B, T, C) becomes (B, C)

        logits = np.mean(logits, axis=1) # (B, T, C) becomes (B, C)
        # apply softmax to get probabilities
        # probs = F.softmax(logits, dim=-1) # (B, C)
        # probs_np = probs.cpu().detach().numpy()
        # Take argmax along the last axis
        argmax_output = np.argmax(logits, axis=-1)

        # sample from the distribution
        # idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # # append sampled index to the running sequence
        # idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return argmax_output

model = Heart_GPT_Model()

model.load_state_dict(torch.load(model_path))
model.eval()
m = model.to(device)

if __name__ == '__main__':
    path_model = '/Model/'
    path_save = '/Data/Data_ECG/'
    split = 'train_V'
    data = np.load(path_save + f'all_windows_{split}.npy')
    # data = data[:10, :]
    all_labels = np.load(path_save + f'all_labels_{split}.npy')
    # indices = np.where(all_labels != 0)[0]
    #  # Take data with these indices
    # data_with_indices = data[indices]
    # all_labels_indices = all_labels[indices]
    # Check data test
    # for i in range(len(data_with_indices)):
    #     print("all_labels[i]: ", all_labels_indices[i])
    #     plot_signal(data_with_indices[i])
    batch_size_infer = 20
    i = 0
    while i < len(all_labels) - batch_size_infer:
        data_tokenised = torch.tensor(data[i:i + batch_size_infer, :], dtype=torch.long, device=device)
        # data_tokenised = torch.tensor(data_with_indices, dtype=torch.long, device=device)
        # argmax_output = m.classified(data_tokenised)[0].tolist()
        argmax_output = m.classified(data_tokenised)
        print(argmax_output)
        i += batch_size_infer



"""
"""