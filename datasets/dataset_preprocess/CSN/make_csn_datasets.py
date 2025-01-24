import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from tqdm import tqdm

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            filtered_data[i, j, :] = filtfilt(b, a, data[i, j, :])

    return filtered_data

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
    meta_path = "./datasets/ecg_datasets/CSN/"
    dataset_path = "/hot_data/jinjiarui/datasets/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
    for data_type in data_types:

        file_path = os.path.join(os.getcwd(), f"chapman_{data_type}.csv")
        df = pd.read_csv(file_path)

        stacked_ecg_data = []
        stacked_label_data = []
        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Processing {data_type} data"
        ):
            row["ecg_path"] = row["ecg_path"].replace("/chapman", "")
            # data_path = dataset_path + row["ecg_path"]
            data_path = row["ecg_path"]
            ecg_data = loadmat(data_path)["val"].astype(np.float32)

            # normalzie the ecg data to -3 3
            normalzied_ecg = normalize_data(ecg_data, new_min=-3, new_max=3)

            label = row[3:].values
            stacked_ecg_data.append(normalzied_ecg)
            stacked_label_data.append(label)
        stacked_ecg_data = np.array(stacked_ecg_data)
        stacked_label_data = np.array(stacked_label_data)

        resample_ecg_data = resample_signals(stacked_ecg_data, 500, 100)
        bandpass_ecg_data = bandpass_filter(resample_ecg_data, 0.67, 40, 100)

        print(bandpass_ecg_data.shape)
        print(stacked_label_data.shape)

        np.save(
            os.path.join(meta_path, f"data/{data_type}_data.npy"), bandpass_ecg_data
        )
        np.save(
            os.path.join(meta_path, f"data/{data_type}_labels.npy"), stacked_label_data
        )
