import torch
import matplotlib.pyplot as plt
import numpy as np
from ECG_LLM.define import model_path
from ECG_LLM.HearGPTModel_with_save_wei_attention import HeartGPTModel
from ECG_LLM.dataset_processing.dataset import load_data_bcty

device = 'cuda' if torch.cuda.is_available() else 'cpu'




def plot_numpy_data(frame_ecg: np.ndarray, frame_label: np.ndarray) -> None:
    """
    Plots the ECG signal and corresponding labels in two subplots.

    Parameters:
    - frame_ecg (np.ndarray): A 1D NumPy array representing the ECG signal data.
                              Values should be scaled appropriately for visualization.
    - frame_label (np.ndarray): A 1D NumPy array representing the labels corresponding to the ECG signal.
                                Labels indicate specific events or annotations.

    Returns:
    - None: Displays the plot directly using matplotlib.
    """
    # Create subplots for ECG signal and labels
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot the ECG signal
    axs[0].plot(frame_ecg, label="ECG Signal")
    axs[0].set_title("ECG Signal")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    # Plot the labels
    axs[1].plot(frame_label, label="Label", color="orange")
    axs[1].set_title("Label")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("Label Value")
    axs[1].legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def plot_numpy_data_v2(frame_ecg: np.ndarray, frame_label: np.ndarray, origin_label: np.ndarray, attention_weights: np.ndarray) -> None:
    """
    Plots the ECG signal, frame labels, origin labels, and attention weights in four subplots.

    Parameters:
    - frame_ecg (np.ndarray): A 1D NumPy array representing the ECG signal data.
                              Values should be scaled appropriately for visualization.
    - frame_label (np.ndarray): A 1D NumPy array representing the labels corresponding to the ECG signal.
                                Labels indicate specific events or annotations.
    - origin_label (np.ndarray): A 1D NumPy array representing the original labels.
    - attention_weights (np.ndarray): A 2D NumPy array representing the attention weights.

    Returns:
    - None: Displays the plot directly using matplotlib.
    """
    # Create subplots for ECG signal, frame labels, origin labels, and attention weights
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))

    # Plot the ECG signal
    axs[0].plot(frame_ecg, label="ECG Signal")
    axs[0].set_title("ECG Signal")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    # Plot the frame labels
    axs[1].plot(frame_label, label="Frame Label", color="orange")
    axs[1].set_title("Frame Label (Model Output)")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("Label Value")
    axs[1].legend()

    # Plot the origin labels
    axs[2].plot(origin_label, label="Origin Label", color="green")
    axs[2].set_title("Origin Label (Ground Truth)")
    axs[2].set_xlabel("Sample Index")
    axs[2].set_ylabel("Label Value")
    axs[2].legend()

    # # Plot the attention weights
    # axs[3].imshow(attention_weights, aspect='auto', cmap='viridis')
    # sum(attention_weights[:, 0]) = 0.14 # 1 , sum(attention_weights[0,]) = 1 => plot follow row is exactly
    axs[3].plot(attention_weights[0], label="wei token in row", color="green")
    axs[3].set_title("Attention Weights")
    axs[3].set_xlabel("Key Index")
    axs[3].set_ylabel("Query Index")
    axs[3].legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()




def infer_data_training():
    train_data_np, train_label_np, eval_data_np, eval_label_np = load_data_bcty()
    batch_size_infer = 2
    i = 100
    while i < len(train_label_np) - batch_size_infer:
        batch_data = train_data_np[i:i + batch_size_infer, :]
        batch_label = train_label_np[i:i + batch_size_infer, :]
        data_tokenised = torch.tensor(batch_data, dtype=torch.long, device=device)
        
        # Perform inference and collect attention weights
        argmax_output = m.classified(data_tokenised)
        attention_weights = weights_matrices[-1]  # Get the latest attention weights
        
        print(argmax_output)
        for idx, label_strip in enumerate(argmax_output):
            plot_numpy_data_v2(batch_data[idx], label_strip, batch_label[idx], attention_weights[idx])
        i += batch_size_infer


if __name__ == '__main__':
    # Initialize a shared list to store attention weights
    weights_matrices = []

    # Pass the shared list to the model
    model = HeartGPTModel(weights_matrices=weights_matrices)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    m = model.to(device)

    # Run inference
    infer_data_training()

    # # Save the collected attention weights to a file
    # np.save('attention_weights.npy', weights_matrices)
    # print("Attention weights saved to attention_weights.npy")