import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import wfdb as wf
from sklearn.preprocessing import minmax_scale  # for rescaling
from wfdb.processing import resample_sig
import os
from glob import glob
from Beat_Classify.inference.ec57_command import run_bxb, run_sumstats

from Beat_Classify.dataset.data_from_study import plot_signal
from Beat_Classify.dataset.data_from_study import butter_bandpass_filter

# The following code is adapted from a tutorial by Andrej Kapathy, available at https://github.com/karpathy/nanoGPT
# The explaination behind this code and the model files can be found in the paper "Interpretable Pre-Trained Transformers for Heart Time-Series Data"
# available at https://arxiv.org/abs/2407.20775

block_size = 500 # this is context length
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2
num_classes = 3
vocab_size = 1001 # (0 - 100)

model_path = "/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Model/Model_beat_classify_study_data_64_8_8_500_500000.pth"
# model_path = "/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Model/Heatbeat_pretrained_128_16_16_500_100_99_train_222.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def inference_one_file_v1(file_name, batch_size_infer):
    sampling_rate = 100
    # Read data
    signal = wf.rdrecord(file_name, channels=[0]).p_signal[:, 0]
    annotation = wf.rdann(file_name, extension="atr")
    header = wf.rdheader(file_name)
    fs_origin = header.fs
    r_peaks, labels = annotation.sample, np.array(annotation.symbol)

    # remove non-beat labels
    invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']
    indices = [i for i, label in enumerate(labels) if label not in invalid_labels]
    r_peaks, labels = r_peaks[indices], labels[indices]

    # align r-peaks
    # filtering uses a 200-ms width median filter and 600-ms width median filter
    # baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
    # filtered_signal = signal - baseline
    filtered_signal = signal

    # for correct R-peak location
    tol = 0.05
    newR = []
    for r_peak in r_peaks:
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(filtered_signal))
        newR.append(r_left + np.argmax(filtered_signal[r_left:r_right]))
    r_peaks = np.array(newR, dtype="int")

    # remove inter-patient variation
    normalized_signal = filtered_signal / np.mean(filtered_signal[r_peaks])

    # AAMI categories
    AAMI = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # N
        "A": 1, "a": 1, "S": 1, "J": 1,  # SVEB
        "V": 2, "E": 2,  # VEB
        "F": 3,  # F
        "/": 4, "f": 4, "Q": 4  # Q
    }
    categories = [AAMI[label] for label in labels]
    # data = {
    #     # "record": record,
    #     "signal": normalized_signal, "r_peaks": r_peaks, "categories": categories, "fs_origin": fs_origin
    # }

    # heartbeat segmentation interval
    before, after = 250, 250
    # Resample to sampling_rate
    signal, _ = resample_sig(signal, fs_origin, sampling_rate)
    r_peaks = (r_peaks * sampling_rate) // fs_origin

    # scale 0 -> 100
    signal = np.round(100 * minmax_scale(signal), 0)

    signals = []
    labels = []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            continue

        if categories[i] == 4:  # remove AAMI Q class
            continue
        window_peak = signal[max(0, r_peaks[i] - before): min(r_peaks[i], len(signal)) + after]
        # if categories[i] == 1:
        #     plot_signal(window_peak)
        if len(window_peak) != before + after:
            continue
        signals.append(window_peak)
        labels.append(categories[i])
    signals = np.asarray(signals)
    labels = np.asarray(labels)

    y_pred = None

    i = 0
    while i < len(labels):
        if i > len(labels):
            data_tokenised = torch.tensor(signals[i:len(labels), :], dtype=torch.long, device=device)
        else:
            data_tokenised = torch.tensor(signals[i:i + batch_size_infer, :], dtype=torch.long, device=device)
        # data_tokenised = torch.tensor(data_with_indices, dtype=torch.long, device=device)
        # argmax_output = m.classified(data_tokenised)[0].tolist()
        argmax_output = m.classified(data_tokenised)
        if y_pred is None:
            y_pred = argmax_output
        else:
            y_pred = np.concatenate((y_pred, argmax_output))
        print(argmax_output)
        i += batch_size_infer

    label_map = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}
    total_symbol = [label_map[num] for num in y_pred]

    return r_peaks, total_symbol, sampling_rate


def inference_one_file_v2(file_name, model_training, batch_size_infer):
    sampling_rate = 100
    # Read data
    signal = wf.rdrecord(file_name, channels=[0]).p_signal[:, 0]
    annotation = wf.rdann(file_name, extension="atr")
    header = wf.rdheader(file_name)
    fs_origin = header.fs
    r_peaks, labels = annotation.sample, np.array(annotation.symbol)

    # remove non-beat labels
    invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']
    indices = [i for i, label in enumerate(labels) if label not in invalid_labels]
    r_peaks, labels = r_peaks[indices], labels[indices]

    # align r-peaks
    # filtering uses a 200-ms width median filter and 600-ms width median filter
    # baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
    # filtered_signal = signal - baseline
    filtered_signal = signal

    # for correct R-peak location
    tol = 0.05
    newR = []
    for r_peak in r_peaks:
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(filtered_signal))
        newR.append(r_left + np.argmax(filtered_signal[r_left:r_right]))
    r_peaks = np.array(newR, dtype="int")

    # remove inter-patient variation
    normalized_signal = filtered_signal / np.mean(filtered_signal[r_peaks])

    # AAMI categories
    AAMI = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # N
        "A": 1, "a": 1, "S": 1, "J": 1,  # SVEB
        "V": 2, "E": 2,  # VEB
        "F": 3,  # F
        "/": 4, "f": 4, "Q": 4  # Q
    }
    categories = [AAMI[label] for label in labels]

    # heartbeat segmentation interval
    before, after = 250, 250
    # Resample to sampling_rate
    signal = butter_bandpass_filter(signal, 1, 40, 250)
    signal, _ = resample_sig(signal, fs_origin, sampling_rate)
    r_peaks = (r_peaks * sampling_rate) // fs_origin

    # scale 0 -> 100
    signal = np.round((vocab_size - 1) * minmax_scale(signal), 0)
    # plot_signal(signal)

    signals = []
    labels = []
    index_remove_r_peaks = []

    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            index_remove_r_peaks.append(i)
            continue

        if categories[i] == 4:  # remove AAMI Q class
            index_remove_r_peaks.append(i)
            continue
        window_peak = signal[max(0, r_peaks[i] - before): min(r_peaks[i], len(signal)) + after]
        # if categories[i] == 1:
        #     plot_signal(window_peak)
        if len(window_peak) != before + after:
            index_remove_r_peaks.append(i)
            continue

        # plot_signal(window_peak)

        signals.append(window_peak)
        labels.append(categories[i])
    signals = np.asarray(signals)
    labels = np.asarray(labels)
    r_peaks = np.delete(r_peaks, index_remove_r_peaks)

    y_pred = None

    i = 0
    while i < len(labels):
        if i > len(labels):
            data_tokenised = torch.tensor(signals[i:len(labels), :], dtype=torch.long, device=device)
        else:
            data_tokenised = torch.tensor(signals[i:i + batch_size_infer, :], dtype=torch.long, device=device)
        # data_tokenised = torch.tensor(data_with_indices, dtype=torch.long, device=device)
        # argmax_output = m.classified(data_tokenised)[0].tolist()
        argmax_output = model_training.classified(data_tokenised)
        if y_pred is None:
            y_pred = argmax_output
        else:
            y_pred = np.concatenate((y_pred, argmax_output))
        # print(argmax_output)
        i += batch_size_infer

    label_map = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}
    total_symbol = [label_map[num] for num in y_pred]

    # save_path = file_name.split('/')[-1]
    save_path = os.path.dirname(file_name)
    eval_bxb = True
    if eval_bxb:
        wf.wrann(
            record_name=str(file_name.split('/')[-1]),
            extension='ai',
            sample=np.asarray(r_peaks, dtype=int),
            symbol=np.asarray(total_symbol),
            # fs=fs_origin,
            fs=sampling_rate,
            write_dir=save_path
        )

    return r_peaks, total_symbol, sampling_rate


def eval_ec57(model_path_ec57):
    model.load_state_dict(torch.load(model_path_ec57))
    model.eval()
    model_training = model.to(device)
    print(f"Eval model name: {model_path_ec57}")

    # Remove result if exist
    save_log_path = '/home/server2/Desktop/Vuong/Data/PhysionetData/'
    list_files = [i for i in glob(save_log_path + "/*")]
    for file in list_files:
        if "_report_line.out" in file:
            os.remove(file)
        if "_sd.out" in file:
            os.remove(file)
        if "_report_standard.out" in file:
            os.remove(file)

    ref_ext = 'ai'
    report_line_file = os.path.join(save_log_path, f"{ref_ext}_report_line.out")
    sd_file = os.path.join(save_log_path, f"{ref_ext}_sd.out")
    report_standard_file = os.path.join(save_log_path, f"{ref_ext}_report_standard.out")

    path2db = '/home/server2/Desktop/Vuong/Data/PhysionetData/mitdb'
    file_names = glob(path2db + '/*.dat')
    # Get rid of the extension
    # file_names = [p[:-4] for p in file_names
    #               if os.path.basename(p)[:-4] not in ['104', '102', '107', '217', 'bw', 'em', 'ma', '2202', '8205']
    #               # if basename(p)[:-4] in ['100', '101']
    #               if '_200hz' not in os.path.basename(p)[:-4]]
    file_names = sorted(file_names)
    for record in file_names[:]:
        print(f"Process in record {record}")
        if os.path.basename(record)[:-4] in ['104', '102', '107', '217', 'bw', 'em', 'ma', '2202', '8205']:
            continue
        if '_200hz' in os.path.basename(record)[:-4]:
            continue
        record = record[:-4]
        inference_one_file_v2(record, model_training, batch_size_infer=20)

        info_bxb = {
            "report_line_file": report_line_file,
            "sd_file": sd_file,
            "report_standard_file": report_standard_file,
            "save_path": path2db,
            "filename": str(record.split('/')[-1]),
            "ref_ext": ref_ext
        }
        # Run bxb for each event
        run_bxb(info_bxb)

    sumstats_output = run_sumstats(report_line_file)
    if sumstats_output:
        with open(report_line_file, 'w') as f:
            f.write(sumstats_output)

    gross_values = []

    with open(report_line_file, 'r') as file:
        for line in file:
            if line.startswith("Gross"):
                # Split the line and replace '-' with '0'
                parts = line.split()
                gross_values = [float(value.replace('-', '0')) for value in parts[1:]]
                break

    print(gross_values)
    return gross_values
if __name__ == '__main__':
    eval_ec57()

"""
"""