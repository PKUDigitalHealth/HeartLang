import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from tqdm import tqdm
import pyemd

import numpy as np
from scipy.signal import firwin, filtfilt


def apply_fir_filter_to_leads(ecg_data, fs, cutoff=0.5, numtaps=21):
    nyquist = 0.5 * fs
    fir_coeff = firwin(numtaps, cutoff / nyquist)
    filtered_ecg_data = np.zeros_like(ecg_data)
    for i in range(ecg_data.shape[0]):  
        filtered_ecg_data[i] = filtfilt(fir_coeff, 1.0, ecg_data[i])
    return filtered_ecg_data


def remove_baseline_mean_per_lead(ecg_signals):
    corrected_signals = np.zeros_like(ecg_signals)
    for i in range(ecg_signals.shape[0]):
        baseline = np.mean(ecg_signals[i])
        corrected_signals[i] = ecg_signals[i] - baseline
    return corrected_signals


def resample_signals(signals, original_fs, target_fs):
    """
    Resample the signals from original_fs to target_fs and replace outliers.

    Parameters:
    - signals: numpy array of shape (batch_size, leads, data_points)
    - original_fs: Original sampling rate
    - target_fs: Target sampling rate

    Returns:
    - resampled_signals: numpy array of the resampled signals
    """
    batch_size, num_leads, _ = signals.shape
    num_original_samples = signals.shape[2]
    num_target_samples = int(num_original_samples * (target_fs / original_fs))
    resampled_signals = np.zeros((batch_size, num_leads, num_target_samples))
    for b in range(batch_size):
        for i in range(num_leads):
            resampled_signals[b, i, :] = signal.resample(
                signals[b, i, :], num_target_samples
            )
    return resampled_signals


def normalize_data(data, new_min=0, new_max=1):
    min_val = np.min(data)
    max_val = np.max(data)

    normalized_data = (new_max - new_min) * (data - min_val) / (
        max_val - min_val
    ) + new_min

    return normalized_data


if __name__ == "__main__":
    data_types = ["train", "val", "test"]
    meta_path = "./datasets/ecg_datasets/CPSC2018"
    dataset_path = "/hot_data/jinjiarui/datasets/cpsc2018"
    for data_type in data_types:

        file_path = os.path.join(os.getcwd(), f"icbeb_{data_type}.csv")
        df = pd.read_csv(file_path)

        stacked_ecg_data = []
        stacked_label_data = []
        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Processing {data_type} data"
        ):
            data_path = os.path.join(dataset_path, row["filename"] + ".mat")
            ecg_data = loadmat(data_path)["ECG"][0][0][2]

            # we only keep the first 2500 points as METS and MERL did
            ecg_data = ecg_data[:, 0:2500]

            # padding to 5000 to match the pre-trained data length
            ecg_data = np.pad(
                ecg_data, ((0, 0), (0, 2500)), "constant", constant_values=0
            )

            # normalzie the ecg data to -1 1
            normalzied_ecg = normalize_data(ecg_data, new_min=-3, new_max=3)

            label = row[7:].values
            stacked_ecg_data.append(normalzied_ecg)
            stacked_label_data.append(label)
        stacked_ecg_data = np.array(stacked_ecg_data)
        stacked_label_data = np.array(stacked_label_data)

        resample_ecg_data = resample_signals(stacked_ecg_data, 500, 100)

        print(resample_ecg_data.shape)
        print(stacked_label_data.shape)

        np.save(
            os.path.join(meta_path, f"data/{data_type}_data.npy"), resample_ecg_data
        )
        np.save(
            os.path.join(meta_path, f"data/{data_type}_labels.npy"), stacked_label_data
        )
