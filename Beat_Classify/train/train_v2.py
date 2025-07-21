import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

from Beat_Classify.inference.transformer_infer_one_file_ecg57 import eval_ec57
from Beat_Classify.define import path_model, path_save_data, path_model_ec57, block_size, batch_size
from Beat_Classify.define_model import n_layer, n_embd, n_head, learning_rate, HeartGPTModel

# Harry Davies 19_09_2024
# The following code is adapted from a tutorial by Andrej Kapathy, available at https://github.com/karpathy/ng-video-lecture


device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_interval = 200 # 2000, sau bao nhieu epoch ites, thi danh gia loss
# save_interval = 10000 # 20000 #how often the model is checkpointed
eval_iters = 20  # 200 so lan data lap de danh gia loss
max_iters = 10000# 1000000


types_beat = [0, 1]
symbols = ['N','V']
split = 'train'
number_type_N = 4000
data = None
labels = None
for i, type_beat in enumerate(types_beat):
    all_windows = np.load(path_save_data + f'all_windows_{split}_{symbols[i]}.npy')
    all_labels = np.load(path_save_data + f'all_labels_{split}_{symbols[i]}.npy')
    print(f'Type_{symbols[i]} have {len(all_labels)} sample')
    if data is None:
        data = all_windows
        labels = all_labels
    else:
        if type_beat == 0:
            np.random.seed(42)
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

def load_data_bcty():
    # if config.label_dir is not None:
    #     data_path = config.label_dir
    # else:
    #     data_path = config.run_dir

    train_data_np = []
    train_label_np = []
    eval_data_np = []
    eval_label_np = []

    SAMPLING_RATE = 128
    subseq_size = int(10 * SAMPLING_RATE)
    data_type = "labseq"
    data_path = "/media/server2/MegaDataset/Vuong_Data/ECG_LLM_DATA/ecg_norm_1_data_label/"

    if data_type == "labseq":
        all_train_data = glob(data_path + '/train/sub*.npy')
        all_eval_data = glob(data_path + '/eval/sub*.npy')
        for train_file in tqdm(all_train_data, leave=True, desc="Load Training Dataset Progress"):
            data = np.load(train_file)
            data = np.reshape(data, (-1, subseq_size, data.shape[-1]))
            label = np.load(train_file.replace("sub_", "lab_"))
            label = np.reshape(label, (-1, subseq_size))

            train_label_np.append(label)
            train_data_np.append(data)

        train_data_np = np.concatenate(train_data_np, axis=0)
        train_label_np = np.concatenate(train_label_np, axis=0)
        for eval_file in tqdm(all_eval_data, leave=True, desc="Load Testing Dataset Progress"):
            data = np.load(eval_file)
            data = np.reshape(data, (-1, subseq_size, data.shape[-1]))
            label = np.load(eval_file.replace("sub_", "lab_"))
            label = np.reshape(label, (-1, subseq_size))

            eval_data_np.append(data)
            eval_label_np.append(label)

        eval_data_np = np.concatenate(eval_data_np, axis=0)
        eval_label_np = np.concatenate(eval_label_np, axis=0)


    # config.set_inputdims(train_data_np.shape[-1])
    return train_data_np, train_label_np, eval_data_np, eval_label_np
train_data_np, train_label_np, eval_data_np, eval_label_np = load_data_bcty()


def get_batch_ecg(split):
    data_batch = train_data if split == 'train' else test_data
    labels_batch = train_labels if split == 'train' else test_labels
    ix = torch.randint(data_batch.shape[0], (batch_size,))
    x = torch.stack([torch.tensor(data_batch[i], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(labels_batch[i], dtype=torch.long) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
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


if __name__ == '__main__':
    # model_path = "/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Model_EC57/Model_beat_classify_study_data_64_8_8_30_100000_19200.pth"
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
    V_se = 0
    V_p = 0

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    # Iterate through each fold
    fold = 1
    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        print("Training on fold: ", fold)
        for iter in range(max_iters):
            if iter % eval_interval == 0 and iter > 0:
                losses = estimate_loss(model)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                # print(f"step {iter}: train loss {losses['train']:.4f}")

                # Append losses to lists
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])


                # model_path for checkpointing
                if losses['val'] < loss_test_max and losses['train'] < loss_train_max:
                    # Delete the previous model
                    model_path = f"{path_model}Model_beat_classify_study_data_{n_embd}_{n_head}_{n_layer}_{block_size}_{max_iters}.pth"
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    torch.save(model.state_dict(), model_path)
                    loss_train_max = losses['train']
                    loss_test_max = losses['val']


                model_path_ec57 = f"{path_model_ec57}Model_beat_classify_study_data_{n_embd}_{n_head}_{n_layer}_{block_size}_{max_iters}_{iter}.pth"
                torch.save(model.state_dict(), model_path_ec57)
                gross_values = eval_ec57(model_path_ec57)
                if int(gross_values[2]) > V_se and int(gross_values[3]) > V_p:
                    #No previous model_path_ec57 maybe not iter - 1 so I will delete all model_path_ec57 except current model_path_ec57
                    for file in os.listdir(path_model_ec57):
                        print("current model_path_ec57: ", model_path_ec57)
                        if os.path.join(path_model_ec57, file) != model_path_ec57:
                            print("previous model_path_ec57: ", file)
                            os.remove(os.path.join(path_model_ec57, file))
                            print(f"{file} has been deleted.")
                    V_se = int(gross_values[2])
                    V_p = int(gross_values[3])
                else:
                    os.remove(model_path_ec57)
                    print(f"Vse={V_se}; V_p={V_p} => {model_path_ec57} has been deleted.")

            # get batch
            x_batch, y_batch = get_batch_ecg('train')
            # loss evaluation
            logits, loss = m(x_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        fold += 1

        # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Evaluation Interval')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    # Save the figure
    plt.savefig('training_validation_loss.png')

    # Optionally, also display the figure
    plt.show()
    print("Done")






