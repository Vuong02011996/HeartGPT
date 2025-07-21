import matplotlib.pyplot as plt
import numpy as np

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


def ecg_to_tokens(ecg_signal: np.ndarray,
                  n_bins: int = 512,
                  clip_value: float = 1.0) -> np.ndarray:
    """
    Chuyển tín hiệu ECG đã normalize thành chuỗi token rời rạc
    bằng cách dùng delta encoding + quantization.

    Args:
        ecg_signal (np.ndarray): Tín hiệu ECG đã normalize (1D array).
        n_bins (int): Số lượng token (mặc định 512).
        clip_value (float): Giá trị delta lớn nhất được giữ (dùng clip để giới hạn outlier).

    Returns:
        np.ndarray: Mảng các token nguyên (1D array).
    """
    # Bước 1: Tính hiệu delta giữa các điểm liên tiếp
    delta = np.diff(ecg_signal)

    # Bước 2: Giới hạn delta để tránh outlier
    delta_clipped = np.clip(delta, -clip_value, clip_value)

    # Bước 3: Chuẩn hóa delta về [0, 1]
    delta_normalized = (delta_clipped + clip_value) / (2 * clip_value)

    # Bước 4: Quantize delta thành token
    tokens = (delta_normalized * (n_bins - 1)).astype(int)

    # thêm token đầu để đảm bảo độ dài khớp.
    first_token = np.array([tokens[0]], dtype=int)  # hoặc token riêng biệt như [0]
    tokens = np.concatenate([first_token, tokens])

    return tokens


def _bbox_color(symbol):
    """Helper function to define the color of the annotation box based on the symbol."""
    colors = {
        'N': 'lightgreen',
        'V': 'lightcoral',
        'S': 'lightskyblue',
        'F': 'gold',
        'Q': 'lightgray'
    }
    return dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor=colors.get(symbol, "white"))


def plot_ecg_with_annotations(ecg_signal, beats, symbols):
    """
    Plot the ECG signal and annotate the predicted beats and symbols.

    Parameters:
    - ecg_signal (np.ndarray): The ECG signal data.
    - beats (list or np.ndarray): The indices of the predicted beats.
    - symbols (list): The symbols corresponding to the predicted beats.

    Returns:
    - None: Displays the plot directly.
    """
    fig, axis = plt.subplots(figsize=(12, 6))

    # Plot the ECG signal
    axis.plot(ecg_signal, label="ECG Signal")
    axis.set_title("ECG Signal with Predicted Beats and Symbols")
    axis.set_xlabel("Sample Index")
    axis.set_ylabel("Amplitude")
    axis.legend()

    # Annotate the beats and symbols
    y_max = ecg_signal.max()
    [
        axis.annotate(
            x,
            xy=(beats[j], y_max - (y_max / 8)),
            xycoords='data',
            textcoords='data',
            bbox=_bbox_color(x)
        )
        for j, x in enumerate(symbols)
    ]

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage (replace with actual data for testing)
    example_ecg = np.random.rand(1000)  # Simulated ECG signal
    example_label = np.random.randint(0, 2, 1000)  # Simulated binary labels
    plot_numpy_data(example_ecg, example_label)