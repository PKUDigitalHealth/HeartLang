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

    def __init__(self, fs, max_len, token_len, save_path, stage, used_chans=None):
        super(QRSTokenizer, self).__init__()
        self.fs = fs
        self.max_len = max_len
        self.token_len = token_len
        self.save_path = save_path
        self.stage = stage
        self.used_chans = used_chans

    def qrs_detection(self, ecg_signal):
        channels, _ = ecg_signal.shape
        all_qrs_inds = []

        for channel_index in range(channels):
            lead_signal = ecg_signal[channel_index, :]
            qrs_inds = processing.xqrs_detect(
                sig=lead_signal, fs=self.fs, verbose=False
            )
            all_qrs_inds.append(qrs_inds)

        # lead_signal = ecg_signal[0, :]
        # qrs_inds = processing.xqrs_detect(sig=lead_signal, fs=self.fs, verbose=False)

        return all_qrs_inds

    def extract_qrs_segments(self, ecg_signal, qrs_inds):
        channels, _ = ecg_signal.shape
        channel_qrs_segments = []

        for channel_index in range(channels):
            qrs_segments = []
            for i in range(len(qrs_inds[channel_index])):
                if i == 0:
                    center = qrs_inds[channel_index][i]
                    start = max(center - self.token_len // 2, 0)
                    if (i + 1) < len(qrs_inds[channel_index]):
                        end = end = (
                            qrs_inds[channel_index][i] + qrs_inds[channel_index][i + 1]
                        ) // 2
                    else:
                        end = min(
                            start + self.token_len, len(ecg_signal[channel_index])
                        )
                elif i == len(qrs_inds[channel_index]) - 1:
                    center = qrs_inds[channel_index][i]
                    end = min(
                        center + self.token_len // 2, len(ecg_signal[channel_index])
                    )
                    start = (
                        qrs_inds[channel_index][i] + qrs_inds[channel_index][i - 1]
                    ) // 2
                else:
                    center = qrs_inds[channel_index][i]
                    start = (
                        qrs_inds[channel_index][i] + qrs_inds[channel_index][i - 1]
                    ) // 2
                    end = (
                        qrs_inds[channel_index][i] + qrs_inds[channel_index][i + 1]
                    ) // 2

                start = max(start, 0)
                end = min(end, len(ecg_signal[channel_index]))
                actual_len = end - start

                if actual_len > self.token_len:
                    center = qrs_inds[channel_index][i]
                    start = max(center - self.token_len // 2, 0)
                    end = min(start + self.token_len, len(ecg_signal[channel_index]))

                segment = np.zeros(self.token_len)
                segment_start = max(self.token_len // 2 - (center - start), 0)
                segment_end = segment_start + (end - start)

                if segment_end > self.token_len:
                    end -= segment_end - self.token_len
                    segment_end = self.token_len

                # print(f"Segment start: {segment_start}, Segment end: {segment_end}")

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
            in_times.extend(self.assign_time_blocks(qrs_inds[channal_index]))
            for segments in channel:
                qrs_sequence.append(segments)
                # in_chans.append(channal_index + 1)
                in_chans.append(self.used_chans[channal_index] + 1)

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

    def plot_original_signal(self, ecg_signal, index):
        num_channels = ecg_signal.shape[0]
        fig, axs = plt.subplots(
            num_channels, 1, figsize=(12, 2 * num_channels), constrained_layout=True
        )

        if num_channels == 1:
            axs = [axs]

        for i in range(num_channels):
            time_points = np.linspace(
                0, len(ecg_signal[i]) / self.fs, num=len(ecg_signal[i])
            )
            axs[i].plot(time_points, ecg_signal[i], label=f"Lead {i + 1}")
            axs[i].set_title(f"Channel {i + 1}")
            axs[i].set_xlabel("Time (seconds)")
            axs[i].set_ylabel("Amplitude")
            axs[i].legend()
            axs[i].grid(True)

        plt.suptitle("Original ECG Signals")
        save_path = os.path.join(
            self.save_path, f"figs/QRS_Segments_{index}_origin.png"
        )
        plt.savefig(save_path)
        plt.close()

    def plot_sequence(self, qrs_sequence, index):
        fig, axs = plt.subplots(16, 16, figsize=(16, 16), constrained_layout=True)

        for i in range(self.max_len):
            row = i // 16
            col = i % 16

            axs[row, col].plot(qrs_sequence[i], color="blue")
            axs[row, col].set_title(f"Token {i+1}")
            axs[row, col].axis("off")

        plt.suptitle(f"QRS Token for Batch {index}", fontsize=16)
        save_path = os.path.join(self.save_path, f"figs/QRS_Token_{index}.png")
        plt.savefig(save_path)
        plt.close()

    def forward(self, x, plot=False):
        x = x[:, self.used_chans, :]
        bs, c, l = x.shape
        batch_qrs_seq = []
        batch_in_chans = []
        batch_in_times = []

        indexs = np.random.choice(range(bs), size=5, replace=False)
        # print(f"Selected indices: {indexs}")
        indexs = [
            40,
            460,
            928,
            1128,
            1202,
            1226,
            1231,
            1300,
            2409,
            3844,
            4174,
            6381,
            18795,
            22469,
            22775,
        ]

        for batch in tqdm(range(bs), desc=f"Detect Batch QRS {self.stage}"):
            ecg_signal = x[batch]
            qrs_inds = self.qrs_detection(ecg_signal)
            channel_qrs_segments = self.extract_qrs_segments(ecg_signal, qrs_inds)

            if batch in indexs and plot:
                self.plot_qrs_segments(
                    channel_qrs_segments, self.save_path, index=batch
                )
                self.plot_original_signal(ecg_signal, batch)

            qrs_sequence, in_chans, in_times = self.qrs_to_sequence(
                channel_qrs_segments, qrs_inds
            )

            if batch in indexs and plot:
                self.plot_sequence(qrs_sequence, batch)

            batch_qrs_seq.append(qrs_sequence)
            batch_in_chans.append(in_chans)
            batch_in_times.append(in_times)

        batch_qrs_seq = np.array(batch_qrs_seq).astype(np.float32)
        batch_in_chans = np.array(batch_in_chans).astype(np.int64)
        batch_in_times = np.array(batch_in_times).astype(np.int64)

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
    """
    Select dataset categories and types based on the dataset name.
    """
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

    elif dataset_name == "CSN":
        data_category = ["data"]
        data_type = ["train", "val", "test"]

    elif dataset_name == "CPSC2018":
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
    # used_channels = [0]   #["I"]
    # used_channels = [0,1]   #["I","II"]
    # used_channels = [0,1,7]   #["I","II","V2"]
    # used_channels = [0,1,2,3,4,5] #["I","II","III","aVR","aVL","aVF"]
    # used_channels = [0,1,2,3,4,5,6,7,8,9,10,11] #["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

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
            save_path = f"./datasets/ecg_datasets/{dataset_name}_QRS_{len(used_channels)}Leads/{category}/"

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
                used_chans=used_channels,
            )

            Tokenizer(data, plot=False)
