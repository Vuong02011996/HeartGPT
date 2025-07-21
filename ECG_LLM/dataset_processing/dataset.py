import os
import numpy as np
import torch
import random
# from REBAR.experiments.configs.base_expconfig import Base_ExpConfig
from datetime import datetime
from tqdm import tqdm
import scipy.io
from glob import glob
import sys
from ECG_LLM.define import data_path




def load_data_bcty():
    train_data_np = []
    train_label_np = []
    eval_data_np = []
    eval_label_np = []

    SAMPLING_RATE = 128
    subseq_size = int(10 * SAMPLING_RATE)
    data_type = "labseq"


    if data_type == "labseq":
        all_train_data = glob(data_path + '/train/sub*.npy')
        all_eval_data = glob(data_path + '/eval/sub*.npy')
        for train_file in tqdm(all_train_data, leave=True, desc="Load Training Dataset Progress"):
            data = np.load(train_file)
            # data = np.reshape(data, (-1, subseq_size, data.shape[-1]))
            data = np.reshape(data, (-1, subseq_size))
            label = np.load(train_file.replace("sub_", "lab_"))
            label = np.reshape(label, (-1, subseq_size))

            train_label_np.append(label)
            train_data_np.append(data)

        train_data_np = np.concatenate(train_data_np, axis=0)
        train_label_np = np.concatenate(train_label_np, axis=0)
        for eval_file in tqdm(all_eval_data, leave=True, desc="Load Testing Dataset Progress"):
            data = np.load(eval_file)
            # data = np.reshape(data, (-1, subseq_size, data.shape[-1]))
            data = np.reshape(data, (-1, subseq_size))
            label = np.load(eval_file.replace("sub_", "lab_"))
            label = np.reshape(label, (-1, subseq_size))

            eval_data_np.append(data)
            eval_label_np.append(label)

        eval_data_np = np.concatenate(eval_data_np, axis=0)
        eval_label_np = np.concatenate(eval_label_np, axis=0)


    # config.set_inputdims(train_data_np.shape[-1])
    return train_data_np, train_label_np, eval_data_np, eval_label_np



def load_data_bcty_all():
    train_data_np = []
    train_label_np = []

    SAMPLING_RATE = 128
    subseq_size = int(10 * SAMPLING_RATE)
    data_type = "labseq"


    if data_type == "labseq":
        all_train_data = glob(data_path + '/train/sub*.npy')
        all_eval_data = glob(data_path + '/eval/sub*.npy')
        for train_file in tqdm(all_train_data, leave=True, desc="Load Training Dataset Progress"):
            data = np.load(train_file)
            # data = np.reshape(data, (-1, subseq_size, data.shape[-1]))
            data = np.reshape(data, (-1, subseq_size))
            label = np.load(train_file.replace("sub_", "lab_"))
            label = np.reshape(label, (-1, subseq_size))

            train_label_np.append(label)
            train_data_np.append(data)



        for eval_file in tqdm(all_eval_data, leave=True, desc="Load Testing Dataset Progress"):
            data = np.load(eval_file)
            # data = np.reshape(data, (-1, subseq_size, data.shape[-1]))
            data = np.reshape(data, (-1, subseq_size))
            label = np.load(eval_file.replace("sub_", "lab_"))
            label = np.reshape(label, (-1, subseq_size))

            train_data_np.append(data)
            train_label_np.append(label)

        train_data_np = np.concatenate(train_data_np, axis=0)
        train_label_np = np.concatenate(train_label_np, axis=0)


    # config.set_inputdims(train_data_np.shape[-1])
    return train_data_np, train_label_np



if __name__ == '__main__':
    # train_data_np, train_label_np, eval_data_np, eval_label_np = load_data_bcty()
    a = 0