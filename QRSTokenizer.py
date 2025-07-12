# --------------------------------------------------------
# Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model
# By Jiarui Jin and Haoyu Wang
# ---------------------------------------------------------
from wfdb import processing
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse

class QRSTokenizer(nn.Module):

    def __init__(
        self, fs, max_len, token_len, save_path, stage, used_channels
    ):
        super(QRSTokenizer, self).__init__()
        self.fs = fs
        self.max_len = max_len
        self.token_len = token_len
        self.save_path = save_path
        self.device = None
        self.stage = stage
        self.used_channels = used_channels

    def qrs_detection(self, ecg_signal):
        channels, _ = ecg_signal.shape

        lead_signal = ecg_signal[0, :]
        qrs_inds = processing.xqrs_detect(sig=lead_signal, fs=self.fs, verbose=False)

        return qrs_inds

    def extract_qrs_segments(self, ecg_signal, qrs_inds):
        channels, _ = ecg_signal.shape
        channel_qrs_segments = []

        for channel_index in range(channels):
            qrs_segments = []
            for i in range(len(qrs_inds)):
                if i == 0:
                    center = qrs_inds[i]
                    start = max(center - self.token_len // 2, 0)
                    if (i + 1) < len(qrs_inds):
                        end = (qrs_inds[i] + qrs_inds[i + 1]) // 2
                    else:
                        end = min(
                            start + self.token_len, len(ecg_signal[channel_index])
                        )
                elif i == len(qrs_inds) - 1:
                    center = qrs_inds[i]
                    end = min(
                        center + self.token_len // 2, len(ecg_signal[channel_index])
                    )
                    start = (qrs_inds[i] + qrs_inds[i - 1]) // 2
                else:
                    center = qrs_inds[i]
                    start = (qrs_inds[i] + qrs_inds[i - 1]) // 2
                    end = (qrs_inds[i] + qrs_inds[i + 1]) // 2

                start = max(start, 0)
                end = min(end, len(ecg_signal[channel_index]))
                actual_len = end - start

                if actual_len > self.token_len:
                    center = qrs_inds[i]
                    start = max(center - self.token_len // 2, 0)
                    end = min(start + self.token_len, len(ecg_signal[channel_index]))

                segment = np.zeros(self.token_len)
                segment_start = max(self.token_len // 2 - (center - start), 0)
                segment_end = segment_start + (end - start)

                if segment_end > self.token_len:
                    end -= segment_end - self.token_len
                    segment_end = self.token_len


                segment[segment_start:segment_end] = ecg_signal[channel_index][
                    start:end
                ]

                qrs_segments.append(segment)

            channel_qrs_segments.append(qrs_segments)

        return channel_qrs_segments

    def assign_time_blocks(self, qrs_inds, interval_length=100):
        in_time = [(ind // interval_length) + 1 for ind in qrs_inds]
        return in_time

    def qrs_to_sequence(self, channel_qrs_segments, qrs_inds):
        qrs_sequence = []
        in_chans = []
        in_times = []

        for channal_index, channel in enumerate(channel_qrs_segments):
            in_times.extend(self.assign_time_blocks(qrs_inds))
            for segments in channel:
                qrs_sequence.append(segments)
                in_chans.append(self.used_channels[channal_index] + 1)

        current_patch_size = len(qrs_sequence)
        if current_patch_size < self.max_len:
            padding_needed = self.max_len - current_patch_size
            for _ in range(padding_needed):
                qrs_sequence.append(np.zeros(self.token_len))
                in_chans.append(0)
                in_times.append(0)

        elif current_patch_size > self.max_len:
            qrs_sequence = qrs_sequence[: self.max_len]
            in_chans = in_chans[: self.max_len]
            in_times = in_times[: self.max_len]

        return np.stack(qrs_sequence), np.array(in_chans), np.array(in_times)

    def plot_qrs_segments(self, qrs_segments, data_path, index):

        save_path = os.path.dirname(data_path) + f"/figs/QRS_Segments_{index}.png"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        channel = len(qrs_segments)

        if channel == 1:
            fig, axs = plt.subplots(
                channel, figsize=(12, 2 * channel), constrained_layout=True
            )
            axs = [axs]
        else:
            fig, axs = plt.subplots(
                channel, figsize=(12, 2 * channel), constrained_layout=True
            )

        for i, segments in enumerate(qrs_segments):
            offset = 0
            for segment in segments:
                segment_length = len(segment)
                axs[i].plot(
                    range(offset, offset + segment_length), segment, color="blue"
                )
                offset += segment_length

            axs[i].set_title(f"Channel {i + 1}")
            axs[i].set_xlabel("Sample Points")
            axs[i].set_ylabel("Amplitude")

        plt.suptitle("ECG Channels with Sequentially Plotted QRS Segments")
        plt.savefig(save_path)
        plt.close()

    def forward(self, x, plot):
        x = x[:, self.used_channels, :]
        bs, c, l = x.shape
        batch_qrs_seq = []
        batch_in_chans = []
        batch_in_times = []

        indexs = np.random.choice(range(bs), size=5, replace=False)

        for batch in tqdm(range(bs), desc="Processing batches"):
            ecg_signal = x[batch]
            qrs_inds = self.qrs_detection(ecg_signal)
            channel_qrs_segments = self.extract_qrs_segments(ecg_signal, qrs_inds)

            if batch in indexs and plot:
                self.plot_qrs_segments(
                    channel_qrs_segments, self.save_path, index=batch
                )

            qrs_sequence, in_chans, in_times = self.qrs_to_sequence(
                channel_qrs_segments, qrs_inds
            )
            batch_qrs_seq.append(qrs_sequence)
            batch_in_chans.append(in_chans)
            batch_in_times.append(in_times)

        batch_qrs_seq = np.array(batch_qrs_seq).astype(np.float32)
        batch_in_chans = np.array(batch_in_chans).astype(np.int32)
        batch_in_times = np.array(batch_in_times).astype(np.int32)

        np.save(os.path.join(self.save_path, f"{self.stage}_data.npy"), batch_qrs_seq)
        np.save(
            os.path.join(self.save_path, f"{self.stage}_data_in_chans.npy"),
            batch_in_chans,
        )
        np.save(
            os.path.join(self.save_path, f"{self.stage}_data_in_times.npy"),
            batch_in_times,
        )


def select_dataset(dataset_name):
    if dataset_name == "PTBXL":
        data_category = [
            "all",
            "diagnostic",
            "form",
            "rhythm",
            "subdiagnostic",
            "superdiagnostic",
        ]
        data_type = ["train", "val", "test"]

    elif dataset_name == "MIMIC-IV":
        data_category = [
            "batch_0",
            "batch_1",
            "batch_2",
            "batch_3",
            "batch_4",
            "batch_5",
            "batch_6",
            "batch_7",
        ]
        data_type = ["train", "val"]

    elif dataset_name == 'CSN':
        data_category = ['data']
        data_type = ["train", "val", "test"]

    elif dataset_name == "CPSC2018":
        data_category = ["data"]
        data_type = ["train", "val", "test"]
    
    elif dataset_name == "PhysioNet2021":
        data_category = ["data"]
        data_type = ["train", "val", "test"]

    else:
        raise ValueError("No such dataset available.")

    return data_category, data_type

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., PTBXL, MIMIC-IV, CSN, CPSC2018).",
    )
    parser.add_argument(
        "--used_channels",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        help="List of channel indices to use (e.g., 0 1 2 for leads I, II, III).Default corresponds to the order: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'].",
    )
    # In Exp we use the following channels:
    # used_channels = [0]   # ["I"]
    # used_channels = [0,1]   # ["I","II"]
    # used_channels = [0,1,7]   # ["I","II","V2"]
    # used_channels = [0,1,2,3,4,5] # ["I","II","III","aVR","aVL","aVF"]
    # used_channels = [0,1,2,3,4,5,6,7,8,9,10,11] # ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    args = parser.parse_args()
    dataset_name = args.dataset_name
    used_channels = args.used_channels

    data_category, data_type = select_dataset(dataset_name)

    for category in data_category:
        for type_ in data_type:
            print(f"Processing data for {category} {type_}...")

            data_path = (
                f"./datasets/ecg_datasets/{dataset_name}/{category}/{type_}_data.npy"
            )
            save_path = f"./datasets/ecg_datasets/{dataset_name}_QRS/{category}/"

            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            data = np.load(data_path)
            print(data.shape)

            if data.shape[1] > 12:
                data = data.transpose(0, 2, 1)

            Tokenizer = QRSTokenizer(
                fs=100,
                max_len=256,
                token_len=96,
                save_path=save_path,
                stage=type_,
                used_channels=used_channels,
            )

            Tokenizer(data, plot=True)