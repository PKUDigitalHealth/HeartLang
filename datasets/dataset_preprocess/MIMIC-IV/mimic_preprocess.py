import argparse
import gc
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import wfdb
from torch.utils.data import Dataset
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

mimic_leads = [
    "I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6",
]

standard_leads = [
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
]

def reorder_leads(data, current_order, target_order):
    """
    Reorder the leads of ECG data according to a target order.

    Parameters:
    - data: numpy array of shape (samples, leads, data_points)
    - current_order: list of leads in the order they are currently in the data
    - target_order: list of leads in the desired order

    Returns:
    - reordered_data: numpy array with leads reordered according to target_order
    """
    index_map = {lead: idx for idx, lead in enumerate(current_order)}
    new_indices = [index_map[lead] for lead in target_order]
    return data[:, new_indices, :]

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to the data.

    Parameters:
    - data: numpy array of shape (samples, leads, data_points)
    - lowcut: Low cutoff frequency (Hz)
    - highcut: High cutoff frequency (Hz)
    - fs: Sampling rate (Hz)
    - order: Order of the filter (default is 5)

    Returns:
    - filtered_data: numpy array of the same shape as data
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            filtered_data[i, j, :] = filtfilt(b, a, data[i, j, :])

    return filtered_data

def concatenate_leads_end_to_end(data):
    """
    Concatenate multi-lead ECG data end-to-end for each sample.

    Parameters:
    - data: numpy array of shape (samples, leads, data_points)

    Returns:
    - concatenated_data: numpy array where each sample is reshaped by concatenating leads end-to-end
    """
    return np.hstack(data.transpose(0, 2, 1))

def normalize_data(train_data, val_data):
    """
    Normalize training and validation data using StandardScaler.

    Parameters:
    - train_data: Concatenated training data in 2D array (samples, features)
    - val_data: Concatenated validation data in 2D array (samples, features)

    Returns:
    - normalized_train_data: Normalized training data
    - normalized_val_data: Normalized validation data
    - scaler: Fitted StandardScaler
    """
    scaler = StandardScaler()
    normalized_train_data = scaler.fit_transform(train_data)
    normalized_val_data = scaler.transform(val_data)
    return normalized_train_data, normalized_val_data, scaler

def restore_shape(normalized_data, original_shape):
    """
    Restore the normalized data back to its original multi-lead ECG shape.

    Parameters:
    - normalized_data: Normalized data in 2D array (samples, concatenated_leads_data)
    - original_shape: The original shape of the data before concatenation (samples, leads, data_points)

    Returns:
    - restored_data: Data restored to original shape
    """
    num_leads, lead_length = original_shape[1], original_shape[2]
    split_indices = [(i + 1) * lead_length for i in range(num_leads - 1)]
    return np.array([np.split(sample, split_indices) for sample in normalized_data])

def replace_outliers(data):
    """
    Replace NaN or inf values in each data point of each lead with the average of neighboring data points.

    Parameters:
    - data: numpy array of shape (leads, data_points)

    Returns:
    - data: numpy array with NaN and inf replaced
    """
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]) or np.isinf(data[i, j]):
                start_index, end_index = max(0, j - 6), min(data.shape[1], j + 7)
                valid_data = data[i, start_index:end_index]
                valid_data = valid_data[~np.isnan(valid_data) & ~np.isinf(valid_data)]
                data[i, j] = np.mean(valid_data) if valid_data.size > 0 else 0
    return data

def resample_signals(signals, original_fs, target_fs):
    """
    Resample the signals from original_fs to target_fs.

    Parameters:
    - signals: numpy array of shape (batch_size, leads, data_points)
    - original_fs: Original sampling rate
    - target_fs: Target sampling rate

    Returns:
    - resampled_signals: numpy array of the resampled signals
    """
    batch_size, num_leads = signals.shape[0], signals.shape[1]
    num_target_samples = int(signals.shape[2] * (target_fs / original_fs))
    resampled_signals = np.zeros((batch_size, num_leads, num_target_samples))

    for b in range(batch_size):
        for i in range(num_leads):
            resampled_signals[b, i, :] = signal.resample(signals[b, i, :], num_target_samples)

    return resampled_signals

def replace_nan_with_window_average(data, window_size=6):
    """
    Replace NaN values in the data array with the average of a window of surrounding data points.

    Parameters:
    - data: numpy array of shape (samples, leads, data_points)
    - window_size: Number of elements to consider on each side of NaN

    Returns:
    - data: numpy array with NaN replaced
    """
    for sample_index in tqdm(range(data.shape[0]), desc="Processing NaN Samples"):
        for lead_index in range(data.shape[1]):
            nan_indices = np.where(np.isnan(data[sample_index, lead_index]))[0]
            for idx in nan_indices:
                start_idx, end_idx = max(0, idx - window_size), min(data.shape[2], idx + window_size + 1)
                window = data[sample_index, lead_index, start_idx:end_idx]
                valid_window = window[~np.isnan(window)]
                data[sample_index, lead_index, idx] = np.mean(valid_window) if valid_window.size > 0 else 0
    return data

def process_batch(batch_data, batch_index, save_path, original_fs, target_fs, is_norm=False):
    """
    Process a single batch of ECG data, resample it, filter it, and save it.

    Parameters:
    - batch_data: List of ECG data arrays
    - batch_index: Index of the current batch
    - save_path: Path to save the processed data
    - original_fs: Original sampling frequency
    - target_fs: Target sampling frequency
    - is_norm: Whether to normalize the data
    """
    batch_save_path = os.path.join(save_path, f"batch_{batch_index}/")
    os.makedirs(batch_save_path, exist_ok=True)

    batch_data = np.array(batch_data)
    batch_data = replace_nan_with_window_average(batch_data)
    all_data_resampled = resample_signals(batch_data, original_fs, target_fs)
    reordered_data = reorder_leads(all_data_resampled, mimic_leads, standard_leads)
    filtered_data = bandpass_filter(reordered_data, lowcut=0.67, highcut=40, fs=target_fs)

    print(f"Shape of the collected data array: {filtered_data.shape}")

    X_train, X_val = train_test_split(filtered_data, test_size=0.1, random_state=0)
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")

    if is_norm:
        original_shape_train, original_shape_val = X_train.shape, X_val.shape
        concatenated_train_data = concatenate_leads_end_to_end(X_train)
        concatenated_val_data = concatenate_leads_end_to_end(X_val)

        normalized_train_data, normalized_val_data, scaler = normalize_data(
            concatenated_train_data, concatenated_val_data
        )

        X_train = restore_shape(normalized_train_data, original_shape_train)
        X_val = restore_shape(normalized_val_data, original_shape_val)

        scaler_file = os.path.join(batch_save_path, "standard_scaler_lead.pkl")
        with open(scaler_file, "wb") as ss_file:
            pickle.dump(scaler, ss_file)

        print(f"Normalized training data shape: {X_train.shape}")
        print(f"Normalized validation data shape: {X_val.shape}")

    np.save(os.path.join(batch_save_path, "train_data.npy"), X_train)
    np.save(os.path.join(batch_save_path, "val_data.npy"), X_val)

    print("Data saved successfully!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the processed data")
    args = parser.parse_args()

    mimic_path = args.dataset_path
    labels_path = os.path.join(mimic_path, "record_list.csv")
    save_path = args.output_path
    os.makedirs(save_path, exist_ok=True)

    labels_df = pd.read_csv(labels_path)
    hash_file_names = labels_df.iloc[:, 4]

    original_fs, target_fs = 500, 100
    batch_size = 100100
    batch_data = []
    batch_index = 0

    for file_index, file_name in enumerate(tqdm(hash_file_names, desc="Reading ECG Data")):
        full_path = os.path.join(mimic_path, file_name)
        try:
            signals, _ = wfdb.rdsamp(full_path)
            signals = signals.transpose(1, 0)
            batch_data.append(signals)

            if len(batch_data) >= batch_size or file_index == len(hash_file_names) - 1:
                process_batch(
                    batch_data, batch_index, save_path, original_fs, target_fs, is_norm=False
                )
                batch_data = []
                gc.collect()
                batch_index += 1

        except FileNotFoundError:
            print(f"File not found: {full_path}")

if __name__ == "__main__":
    main()