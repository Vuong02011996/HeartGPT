import numpy as np
import os
import wfdb
import matplotlib.pyplot as plt
from ECG_LLM.define import SAMPLING_RATE, USED_NORM, NUM_CLASS
from scipy.ndimage import label, find_objects, maximum
from ECG_LLM.define import (SAMPLING_RATE,
                                       NUM_CLASS,
                                       MIN_RR_LEN,
                                       OFFSET_LABEL,
                                       GROUP_LABEL_LEN,
                                       HEARTBEAT_TYPE,
                                       FEATURE_LEN,
                                       PHYSIONET_DATA,
                                       CHANNEL_DEFAULT,
                                       USED_NORM,
                                       )

# ======================== PLOT FUNCTIONS =========================================

def plot_ecg_predictions(filtered_waveform, val_predictions, val_labels, record_id, sample_rate=250, num_seconds=10):
    """
    Plots filtered ECG waveform, predictions, and labels.
 
    """
    num_samples = num_seconds * sample_rate
    time_axis = np.linspace(0, num_seconds, num_samples)

    # Get the first channel only
    ecg_signal = filtered_waveform[:num_samples, 0]
    predictions = val_predictions[:num_samples]
    labels = val_labels[:num_samples]

    labels_peaks = get_peaks(ecg_signal, labels)
    predictions_peaks = get_peaks(ecg_signal, predictions)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot ECG waveform
    axs[0].plot(time_axis, ecg_signal, color='black', label="Filtered ECG")
    axs[0].scatter(np.array(labels_peaks) / sample_rate, ecg_signal[labels_peaks], color='red', marker='o', label="Label Peaks")
    axs[0].scatter(np.array(predictions_peaks) / sample_rate, ecg_signal[predictions_peaks], color='green', marker='x', label="Predicted Peaks")
    axs[0].set_ylabel("ECG Signal")
    axs[0].legend()

    # Plot Ground Truth Labels
    axs[1].plot(time_axis, labels, color='blue', linestyle='dashed', label="True Labels")
    axs[1].scatter(np.array(labels_peaks) / sample_rate, np.ones(len(labels_peaks)), color='red', marker='o', label="Label Peaks")
    axs[1].set_ylabel("True Labels (0/1)")
    axs[1].legend()

    # Plot Model Predictions
    axs[2].plot(time_axis, predictions, color='red', linestyle='dotted', label="Predictions")
    axs[2].scatter(np.array(predictions_peaks) / sample_rate, np.ones(len(predictions_peaks)), color='green', marker='x', label="Predicted Peaks")
    axs[2].set_ylabel("Predictions (0/1)")
    axs[2].set_xlabel("Time (seconds)")
    axs[2].legend()
    plt.suptitle("ECG Signal Before and After Filtering with record ID: " + str(record_id))
    plt.tight_layout()
    plt.show()


def plot_ecg_predictions_random(filtered_waveform, val_predictions, val_labels, record_id, sample_rate=250, num_seconds=10):
    """
    Plots filtered ECG waveform, predictions, and labels, with random time range
    """
    total_samples = min(len(filtered_waveform), len(val_predictions), len(val_labels))
    num_samples = num_seconds * sample_rate

   
    if total_samples <= num_samples:
        start_idx = 0  
    else:
        start_idx = np.random.randint(0, total_samples - num_samples)

    end_idx = start_idx + num_samples
    time_axis = np.linspace(start_idx / sample_rate, end_idx / sample_rate, num_samples)

    # Extract the first channel 
    ecg_signal = filtered_waveform[start_idx:end_idx, 0]
    predictions = val_predictions[start_idx:end_idx]
    labels = val_labels[start_idx:end_idx]


    labels_peaks = get_peaks(ecg_signal, labels)
    predictions_peaks = get_peaks(ecg_signal, predictions)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot ECG waveform
    axs[0].plot(time_axis, ecg_signal, color='black', label="Filtered ECG")
    axs[0].scatter(np.array(labels_peaks) / sample_rate + start_idx / sample_rate, ecg_signal[labels_peaks], color='red', marker='o', label="Label Peaks")
    axs[0].scatter(np.array(predictions_peaks) / sample_rate + start_idx / sample_rate, ecg_signal[predictions_peaks], color='green', marker='x', label="Predicted Peaks")
    axs[0].set_ylabel("ECG Signal")
    axs[0].legend()

    # Plot Ground Truth Labels
    axs[1].plot(time_axis, labels, color='blue', label="True Labels")
    axs[1].scatter(np.array(labels_peaks) / sample_rate + start_idx / sample_rate, np.ones(len(labels_peaks)), color='red', marker='o', label="Label Peaks")
    axs[1].set_ylabel("True Labels (0/1)")
    axs[1].legend()

    # Plot Model Predictions
    axs[2].plot(time_axis, predictions, color='brown', label="Predictions")
    axs[2].set_ylabel("Predictions (0/1)")
    axs[2].scatter(np.array(predictions_peaks) / sample_rate + start_idx / sample_rate, np.ones(len(predictions_peaks)), color='green', marker='x', label="Predicted Peaks")
    axs[2].set_xlabel("Time (seconds)")
    axs[2].legend()
    
    fig.suptitle(f"ECG Signal, Ground Truth, and Predictions (Record ID: {record_id})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()






# ============================= HELPER FUNCTIONS ===============================================
def reconstruct(val_predictions):
    """
    Reconstruct labels and prediction to (patents, min(waveform))
    """
    # Reshape to (patent, 14400)
    all_val_prediction = val_predictions.reshape(len(DS_EVAL), -1)

    reconstructed_labels = []
    reconstructed_predictions = []

    # Reconstruct
    for i in range(all_val_prediction.shape[0]):
        predictions_per_patent = []
        for j in range(all_val_prediction.shape[1]):
            predictions_per_patent.extend([all_val_prediction[i][j] for _ in range(NUM_SAMPLES_PER_FRAME)])
        reconstructed_predictions.append(predictions_per_patent)
    reconstructed_predictions = np.array(reconstructed_predictions)

    return reconstructed_predictions


def get_ith_labels_and_predictions(record_id, val_labels, val_predictions, DS_EVAL):
    """
    Get i-th labels and predictions
    """
    search_index = DS_EVAL.index(int(record_id))
    print("Find labels at", search_index, record_id)
    return val_predictions[search_index, :], val_labels[search_index, :]



def rescale_peaks(peaks, original_fs, new_fs):
    """Rescales peak indices from new_fs back to original_fs."""
    return np.round(np.array(peaks) * (original_fs / new_fs)).astype(int)


def op_process_peaks(group_beat_candidate, offset_label=OFFSET_LABEL, min_rr_len=MIN_RR_LEN):
    """
    Processes peak candidates using vectorized operations without loops or globals.

    Refines peak labels based on duration and spacing using label-based analysis.

    Args:
        group_beat_candidate (np.ndarray): Input array with peak candidate values.
                                           Values > 0 indicate potential peaks.
        offset_label (int): Minimum width (duration) for a peak label to be kept.
                            Pulses shorter than this will be removed.
        min_rr_len (int): Minimum space between peak labels. Gaps shorter
                          than this might be merged if adjacent labels have
                          the same assigned maximum value.

    Returns:
        np.ndarray: Processed array where peak labels are refined.
    """
    if group_beat_candidate is None or len(group_beat_candidate) == 0:
        return np.array([], dtype=group_beat_candidate.dtype if group_beat_candidate is not None else float)

    # --- Step 0: Initial Binary Mask & Labeling ---
    # Create a binary version (1 where candidate > 0, else 0)
    binary_candidates = group_beat_candidate > 0

    # Label contiguous regions of 'True' in the binary mask
    # `labeled_array` will have 0 for background, and integers 1, 2, ... for each peak region
    # `num_labels` is the total count of detected peak regions
    labeled_array, num_labels = label(binary_candidates)

    # If no peaks found, return an array of zeros
    if num_labels == 0:
        return group_beat_candidate

    # --- Step 1: Assign Max Value within Each Labeled Segment ---
    # Find the slices corresponding to each label (peak region)
    # `slices` is a list where slices[i] corresponds to label i+1
    slices = find_objects(labeled_array)

    # Calculate the maximum value from the *original* `group_beat_candidate`
    # within each labeled region. `indices` specifies which labels (1 to num_labels)
    # to calculate the maximum for.
    max_values_per_label = maximum(group_beat_candidate,
                                   labels=labeled_array,
                                   index=np.arange(1, num_labels + 1))

    # Create the initial output array by mapping each label index to its max value
    # Prepend 0 for the background label (label 0)
    output_values_map = np.concatenate(([0], max_values_per_label))
    # Use the labeled_array to index into this map, effectively broadcasting the max value
    new_group_beat_candidate_2 = output_values_map[labeled_array]

    # --- Step 2: Remove Short Pulses ---
    # Extract start and (exclusive) stop indices from slices
    # These indices correspond to labels 1 through num_labels
    indx_start_split = np.array([s[0].start for s in slices], dtype=int)
    indx_stop_split = np.array([s[0].stop for s in slices], dtype=int)  # Exclusive stop

    # Calculate widths of labels (pulses)
    pulse_widths = indx_stop_split - indx_start_split

    # Find the *indices* (0 to num_labels-1) of labels that are too short
    short_pulse_indices = np.flatnonzero(pulse_widths < offset_label)

    if len(short_pulse_indices) > 0:
        # Get the actual *label numbers* (1 to num_labels) to remove
        labels_to_remove = short_pulse_indices + 1

        # Create a boolean mask for elements belonging to short pulses
        mask_short_pulses = np.isin(labeled_array, labels_to_remove)

        # Set the values in the output array corresponding to short pulses to 0
        new_group_beat_candidate_2[mask_short_pulses] = 0

        # --- Update label info for the next step ---
        # Keep track of which original labels (and their indices/slices) remain
        keep_mask = np.ones(num_labels, dtype=bool)
        keep_mask[short_pulse_indices] = False

        # Filter start/stop indices and corresponding labels
        remaining_labels = np.arange(1, num_labels + 1)[keep_mask]
        indx_start_split = indx_start_split[keep_mask]
        indx_stop_split = indx_stop_split[keep_mask]
    else:
        # If no short pulses, all original labels remain
        remaining_labels = np.arange(1, num_labels + 1)
        # indx_start_split and indx_stop_split are already correct

    # --- Step 3: Merge Labels Separated by Short Spaces ---
    # Proceed only if more than one label remains after filtering short ones
    if len(remaining_labels) > 1:
        # Calculate spaces between consecutive *remaining* pulses
        # Space = start of next pulse - stop of current pulse
        spaces = indx_start_split[1:] - indx_stop_split[:-1]

        # Find the *indices* (relative to the remaining labels array) where gaps are short
        short_space_indices = np.flatnonzero(spaces < min_rr_len)

        if len(short_space_indices) > 0:
            # Identify the pairs of *original label numbers* surrounding the short gaps
            labels_before_gap = remaining_labels[short_space_indices]
            labels_after_gap = remaining_labels[short_space_indices + 1]

            # Get the assigned maximum values for these label pairs (using the map created in Step 1)
            values_before_gap = output_values_map[labels_before_gap]
            values_after_gap = output_values_map[labels_after_gap]

            # Find which gaps should be merged: condition is same non-zero max value assigned
            merge_mask = (values_before_gap == values_after_gap) & (values_before_gap > 0)

            # Get the indices (relative to remaining labels) of gaps that satisfy the merge condition
            gaps_to_merge_indices = short_space_indices[merge_mask]

            if len(gaps_to_merge_indices) > 0:
                # Get the start/stop indices defining the *gaps* to be filled
                # Start of gap = stop of the preceding pulse
                merge_gap_starts = indx_stop_split[gaps_to_merge_indices]
                # End of gap = start of the succeeding pulse (exclusive)
                merge_gap_ends = indx_start_split[gaps_to_merge_indices + 1]
                # Get the value to fill the gap with
                merge_values = values_before_gap[merge_mask]  # Or values_after_gap

                # --- Fill the gaps (Vectorized Indexing Approach) ---
                # This part is tricky to vectorize without *any* form of iteration.
                # We create arrays of all indices within the gaps and their corresponding values.
                # This can be memory-intensive if gaps are huge, but avoids Python loops.

                # Calculate the number of indices in each gap to fill
                gap_lengths = merge_gap_ends - merge_gap_starts
                valid_gaps_mask = gap_lengths > 0  # Only consider actual gaps

                if np.any(valid_gaps_mask):
                    merge_gap_starts = merge_gap_starts[valid_gaps_mask]
                    merge_gap_ends = merge_gap_ends[valid_gaps_mask]
                    merge_values = merge_values[valid_gaps_mask]
                    gap_lengths = gap_lengths[valid_gaps_mask]

                    # Create an array indicating the start index of each gap's indices in the final flat array
                    index_starts = np.concatenate(([0], np.cumsum(gap_lengths[:-1])))

                    # Total number of indices to fill
                    total_indices = np.sum(gap_lengths)

                    # Create the flat array of indices to fill
                    all_gap_indices = np.empty(total_indices, dtype=int)
                    # Create the flat array of values to fill with
                    all_gap_values = np.empty(total_indices, dtype=new_group_beat_candidate_2.dtype)

                    # Use np.repeat and np.arange cleverly to generate indices without a Python loop
                    # Create increments for each gap's range
                    increments = np.arange(total_indices) - np.repeat(index_starts, gap_lengths)
                    # Add the starting index of each gap
                    all_gap_indices = increments + np.repeat(merge_gap_starts, gap_lengths)

                    # Repeat the merge values for each index in the gap
                    all_gap_values = np.repeat(merge_values, gap_lengths)

                    # --- Perform the vectorized assignment ---
                    # Ensure indices are within the bounds of the output array
                    valid_assignment_mask = (all_gap_indices >= 0) & (all_gap_indices < len(new_group_beat_candidate_2))

                    new_group_beat_candidate_2[all_gap_indices[valid_assignment_mask]] = all_gap_values[
                        valid_assignment_mask]

    return new_group_beat_candidate_2


def get_peaks(ecg_signal, group_beat_candidate, reprocess=True):
    """
    Get the position of peak or prediction or labels
    Args:
    - ecg_signal: 1D array of ecg signal
    - preferences: 1D array (list) of labels or predictions
    """
    peaks = []
    symbol = []
    in_peak = False
    beat_inv = {i: k for i, k in enumerate(HEARTBEAT_TYPE.keys())}
    if reprocess:
        preferences = op_process_peaks(group_beat_candidate)
    else:
        preferences = group_beat_candidate.copy()

    for i in range(len(preferences)):
        if preferences[i] > 0:
            if not in_peak:
                start = i  # Start of a detected beat
                in_peak = True
            # Find max within detected 1s
            if i == len(preferences) - 1 or preferences[i + 1] == 0:
                peak_idx = start + np.argmax(np.abs(ecg_signal[start:i+1]))     # Peak index --> With max absolute value
                peaks.append(peak_idx)
                symbol.append(beat_inv[preferences[start]])
                in_peak = False

    return np.asarray(peaks, dtype=np.int64), np.asarray(symbol)

def get_peaks3(ecg_signal, group_beat_candidate, prob_max_masked, reprocess=True):
    """
    Get the position of peak or prediction or labels
    Args:
    - ecg_signal: 1D array of ecg signal
    - preferences: 1D array (list) of labels or predictions
    """
    peaks = []
    symbol = []
    in_peak = False
    beat_inv = {i: k for i, k in enumerate(HEARTBEAT_TYPE.keys())}
    prob_max_masked_filler = np.where(prob_max_masked < 0.6, 0, prob_max_masked)
    if reprocess:
        preferences = op_process_peaks(group_beat_candidate)
    else:
        preferences = group_beat_candidate.copy()

    for i in range(len(preferences)):
        if preferences[i] > 0:
            if not in_peak:
                start = i  # Start of a detected beat
                in_peak = True
            # Find max within detected 1s
            if i == len(preferences) - 1 or preferences[i + 1] == 0:
                in_peak = False
                peak_idx = start + np.argmax(np.abs(ecg_signal[start:i + 1]))  # Peak index --> With max absolute value
                peaks.append(peak_idx)

                idx_max_prob = np.argmax(prob_max_masked[start:i + 1])
                idx_max = np.argmax(preferences[start:i + 1])
                type_symbol = np.max(preferences[start:i + 1])
                if type_symbol == 3:
                    # type_symbol = 3
                    if prob_max_masked[idx_max_prob + start] > 0.7 and preferences[idx_max_prob + start] == 1:
                        type_symbol = preferences[idx_max_prob + start]
                    if prob_max_masked[idx_max_prob + start] > 0.6 and preferences[idx_max_prob + start] != 1:
                        type_symbol = preferences[idx_max_prob + start]

                    # if prob_max_masked[idx_max_prob + start] > 0.7:
                    #     type_symbol = preferences[idx_max_prob + start]

                elif type_symbol == 2:
                    prob_chose_V = prob_max_masked[idx_max + start]
                    for j in range(start, i + 1):
                        if preferences[j] == 2 and prob_max_masked[j] > prob_max_masked[idx_max + start]:
                            prob_chose_V = prob_max_masked[j]

                    if prob_chose_V < 0.5:
                        type_symbol = preferences[idx_max_prob + start]

                symbol.append(beat_inv[type_symbol])

    return np.asarray(peaks, dtype=np.int64), np.asarray(symbol)


def filter_peaks(peaks, min_distance=45):
    """
    Filter out peaks that are too close after get_peaks
    Args: 
    - peaks: list of peaks position from get_peaks()
    """
    if not peaks:
        return []
    
    filtered_peaks = [peaks[0]]
    
    for i in range(1, len(peaks)):
        if peaks[i] - filtered_peaks[-1] >= min_distance:
            filtered_peaks.append(peaks[i])
        else:
            filtered_peaks[-1] = peaks[i]
    
    return filtered_peaks


def filter_large_ones(arr, min_size=20):
    """
    Filter, keep the series of 1s with the size larger than min_size --> To filter out 1's noise series from prediction
    """  
    arr = np.array(arr)
    
    # Find all sequences of consecutive 1s
    one_segments = []
    start = None
    
    for i in range(len(arr)):
        if arr[i] == 1 and start is None:
            start = i
        elif arr[i] == 0 and start is not None:
            one_segments.append((start, i - 1))
            start = None
    
    if start is not None:
        one_segments.append((start, len(arr) - 1))
    
    # Create output array
    output = np.zeros_like(arr)
    
    # Keep only segments larger than min_size
    for seg_start, seg_end in one_segments:
        if (seg_end - seg_start + 1) >= min_size:
            output[seg_start:seg_end + 1] = 1
    
    return output.tolist()


def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)
