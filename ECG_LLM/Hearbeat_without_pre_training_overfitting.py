import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.model_selection import KFold
import os

from ECG_LLM.dataset_processing.dataset import load_data_bcty, load_data_bcty_all
from ECG_LLM.define import path_model, model_path
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


os.makedirs(path_model, exist_ok=True)



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




def train_overfitting():
    """
        Removing KFold and validation
        Using a small dataset (50 samples)
        Setting dropout to 0
        Training on full training data only
        Reporting loss and training accuracy
        :return:
        Overfitting means your model performs very well on training data but generalizes poorly.
        Regularization methods like dropout, weight decay, or data augmentation prevent overfitting, so you should disable them:
    """
    # Load all data and reduce to small subset for overfitting
    train_data_np, train_label_np = load_data_bcty_all()

    # Use only first 50 samples to force overfitting
    train_data = train_data_np
    train_labels = train_label_np

    # Move model to device
    model = HeartGPTModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def get_batch_overfit():
        ix = torch.randint(len(train_data), (batch_size,))
        x = torch.stack([torch.tensor(train_data[i], dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(train_labels[i], dtype=torch.long) for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    print("Training model to overfit on small dataset...")
    loss_train_max = 10

    for iter in range(max_iters):
        model.train()
        x_batch, y_batch = get_batch_overfit()

        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 50 == 0 or iter == max_iters - 1:
            print(f"Iter {iter}: Train loss = {loss.item():.4f}")
        if loss.item() < loss_train_max:
            # Delete the previous model
            # model_path = f"{path_model}Model_overfitting_n_embd_{n_embd}_n_head_{n_head}_n_layer_{n_layer}_block_size_{block_size}_token_{vocab_size}.pth"
            if os.path.exists(model_path):
                os.remove(model_path)
            torch.save(model.state_dict(), model_path)
            loss_train_max = loss.item()

    # ------------------------
    # Evaluate on training set
    # ------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(train_data)):
            x = torch.tensor(train_data[i], dtype=torch.long).unsqueeze(0).to(device)
            y = train_labels[i]
            pred = model.classified(x)[0]
            if all(pred) == all(y):
                correct += 1
            total += 1

    print(f"\nOverfit training accuracy: {correct}/{total} = {correct / total:.4f}")

def training_with_k_fold():
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

    def get_batch_ecg(split):

        data_batch = train_data if split == 'train' else test_data
        labels_batch = train_labels if split == 'train' else test_labels
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


    # Initialize KFold with 5 splits
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf = KFold(n_splits=5, shuffle=True)
    loss_train_max = 1
    loss_test_max = 1
    # Iterate through each fold
    fold = 1

    # train_data_np, train_label_np, eval_data_np, eval_label_np = load_data_bcty()
    train_data_np, train_label_np = load_data_bcty_all()
    data = train_data_np
    labels = train_label_np

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
                model_path = f"{path_model}Model_beat_classify_study_data_n_embd_{n_embd}_n_head_{n_head}_n_layer_{n_layer}_block_size_{block_size}_token_{vocab_size}.pth"
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


if __name__ == '__main__':
    train_overfitting()


