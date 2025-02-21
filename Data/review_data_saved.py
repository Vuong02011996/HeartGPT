import numpy as np
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def plot_signal(signal, symbol):
    plt.figure()
    plt.plot(signal, label='Signal')

    # Calculate the middle index
    middle_index = len(signal) // 2

    # Plot the red point at the middle index
    plt.plot(middle_index, signal[middle_index], 'ro', label='Middle Point')
    # plt.plot(peak, signal[peak], 'bo', label='Middle Point')
    plt.suptitle(str(symbol))
    plt.legend()
    plt.show()

def plot_windows_grid(windows, grid_size=(10, 10)):
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(windows):
            ax.plot(windows[i])
            # how to plot middle point in each window

            middle_index = len(windows[i]) // 2
            # Plot the red point at the middle index
            ax.plot(middle_index, windows[i][middle_index], 'ro')
            # ax.set_title(f'Window {i+1}')
        ax.axis('off')  # Hide axes for better visualization
    
    plt.tight_layout()
    plt.show()


types_beat = [0, 1, 2]
symbols = ['N', 'S', 'V']
split = 'train'
path_save ='/home/server2/Desktop/Vuong/Reference_Project/HeartGPT/Data/Data_Study/'
# Define the path to the saved file
for i, type_beat in enumerate(types_beat):
    if symbols[i] == 'V':
        file_path = path_save + f'all_windows_{split}_{symbols[i]}.npy'
        # Load the data
        all_windows_loaded = np.load(file_path)
        plot_windows_grid(all_windows_loaded[:100])
        j = 0
        while j < (len(all_windows_loaded)):
            plot_windows_grid(all_windows_loaded[j: max(j + 100, len(all_windows_loaded))])
            j += 100