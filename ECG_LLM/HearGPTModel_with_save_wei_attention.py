import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from ECG_LLM.config_model import (batch_size,
                                  device,
                                  eval_iters,
                                  n_embd,
                                  dropout,
                                  vocab_size,
                                  block_size,
                                  n_layer,
                                  n_head,
                                  num_classes,
                                  learning_rate,
                                  max_iters,
                                  eval_interval)


class Head(nn.Module):
    def __init__(self, head_size, weights_matrices):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.weights_matrices = weights_matrices  # Shared list to store attention weights

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v

        # Save attention weights to the shared list
        self.weights_matrices.append(wei.detach().cpu().numpy())

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, weights_matrices):
        super().__init__()
        # creating a list of head objects (turned into modules) resulting in a number of head modules
        # then assigns the list of modules to self.heads - these run in parellel
        self.heads = nn.ModuleList([Head(head_size, weights_matrices) for _ in range(num_heads)])
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

    def __init__(self, n_embd, n_head, weights_matrices):
        super().__init__()
        head_size = n_embd // n_head
        # communication
        self.sa = MultiHeadAttention(n_head, head_size, weights_matrices)
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

    def __init__(self, weights_matrices):
        super().__init__()
        # table needs to be vocab size by vocab size, to look up probability of next token given this token
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head, weights_matrices=weights_matrices) for _ in range(n_layer)])
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
        # channel is vocab size, so in this case 65

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

        # if targets is None:
        #     loss = None
        # else:
        #     B, T, C = logits.shape # C = 4  from lm_head
        #     # print("B, T, C = ", B, T, C)
        #     # logits = logits.view(B*T, C)
        #     """
        #     targets = targets.view(B*T)
        #       ^^^^^^^^^^^^^^^^^
        #     RuntimeError: shape '[32000]' is invalid for input of size 64
        #     """
        #     # targets = targets.view(-1)
        #     # targets = targets.view(-1, 1)
        #     # targets_one_hot = F.one_hot(targets, num_classes=4)
        #
        #     # loss = F.cross_entropy(logits, targets)
        #     logits = logits.mean(dim=1)  # Shape now becomes (B, C)
        #     loss = criterion(logits, targets)

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

    def classified(self, idx):
        # idx is (B, T) array of indices in the current context
        # crop idx (context) to the last block_size tokens because positional embeddings only has up to block size
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits, loss = self(idx_cond)
        # focus only on the last time step
        # logits = logits[:, -1, :] # becomes (B, C)
        # # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        probs = probs.cpu().detach().numpy()
        # # sample from the distribution
        # idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # # append sampled index to the running sequence
        # idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        argmax_output = np.argmax(probs, axis=-1)

        return argmax_output

    def get_logits(self, idx):
        # idx is (B, T) array of indices in the current context
        # crop idx (context) to the last block_size tokens because positional embeddings only has up to block size
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits, loss = self(idx_cond)
        # focus only on the last time step
        # logits = logits[:, -1, :] # becomes (B, C)
        # # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)
        # probs = probs.cpu().detach().numpy()
        # # sample from the distribution
        # idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # # append sampled index to the running sequence
        # idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        # argmax_output = np.argmax(probs, axis=-1)

        return probs






