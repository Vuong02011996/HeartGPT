import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from ECG_LLM.define import model_path
from ECG_LLM.HearGPTModel_with_save_wei_attention import HeartGPTModel
from ECG_LLM.utils import get_peaks
import matplotlib.pyplot as plt


weights_matrices = []
model = HeartGPTModel(weights_matrices=weights_matrices)
model.load_state_dict(torch.load(model_path))


def print_all_attention_weights_shape(model):
    for layer_idx, block in enumerate(model.blocks):
        print(f"\n--- Layer {layer_idx} ---")
        for head_idx, head in enumerate(block.sa.heads):
            q_weight_shape = tuple(head.query.weight.shape)
            k_weight_shape = tuple(head.key.weight.shape)
            v_weight_shape = tuple(head.value.weight.shape)
            print(f"  Head {head_idx}:")
            print(f"    Query weight shape: {q_weight_shape}")
            print(f"    Key weight shape:   {k_weight_shape}")
            print(f"    Value weight shape: {v_weight_shape}")

if __name__ == '__main__':
    print_all_attention_weights_shape(model)