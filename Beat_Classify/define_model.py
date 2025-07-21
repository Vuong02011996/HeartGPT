import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from Beat_Classify.define import num_classes,  block_size, vocab_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'


learning_rate = 3e-04
n_embd = 64 # 384 / 6 means every head is 64 dimensional
n_head = 8
n_layer = 8

# n_embd = 128
# n_head = 16
# n_layer = 16

dropout = 0.2


# GPU is necessary. Training of 8 head, 8 layer model and 500 context length was possible with 12GB VRAM



class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Using mask
        # self.register_buffer('tril', torch.tril(torch.ones((block_size,block_size)))) #buffer means not updated by optimiser
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # start = time.time()
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # square root headsize # (B, T, C) @ (B, C, T) = B, T, T
        # for every batch, we will now have a T by T matrix giving us the affinities of each token
        # Using mask
        # wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))

        # the tril signifies a decoder block, future tokens cannot communicate with the past
        wei = F.softmax(wei, dim=-1)  # all attention weights sum to 1 for updating a single token
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        # end = time.time()
        # print(start-end)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        # creating a list of head objects (turned into modules) resulting in a number of head modules
        # then assigns the list of modules to self.heads - these run in parellel
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # projection generally matches sizes for adding in residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate the output of the different attention heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # multiplication performed in attention is all you need paper
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


# # Number of samples for each class
# num_samples = np.array([1278294, 202285, 109746])

# # Calculate class weights
# class_weights = 1. / num_samples
# class_weights = class_weights / class_weights.sum() * len(num_samples)

# # Convert class weights to a tensor
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# # Define the loss function with class weights
# criterion = nn.CrossEntropyLoss(weight=class_weights)

criterion = nn.CrossEntropyLoss()


# criterion=nn.BCELoss()
# create heart GPT class
class HeartGPTModel(nn.Module):

    def __init__(self):
        super().__init__()
        # table needs to be vocab size by vocab size, to look up probability of next token given this token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        # self.lm_head = nn.Linear(n_embd, vocab_size)
        self.lm_head = nn.Linear(n_embd, num_classes)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx is batch, targets is time
        tok_emb = self.token_embedding_table(idx)  # (B, T, vocab_size) which is batch, time, channel
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # T, C (integers from 0 to T-1)
        x = tok_emb + pos_emb  # B, T, C
        x = self.blocks(x)  # B, T, C
        x = self.ln_f(x)  # B, T, C

        logits = self.lm_head(x)
        # channel is vocab size, so in this case 65

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # C = 4  from lm_head
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
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

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

if __name__ == '__main__':
    pass