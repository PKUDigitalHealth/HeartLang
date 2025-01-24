import bisect
import random

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
import os

folder_path = List[Path]


def build_ECGpretrain_dataset(datasets_2dlist: list, stage):
    dataset_list = []
    for dataset_1dlist in datasets_2dlist:
        dataset = ECGDataset(dataset_1dlist, stage)
        dataset_list.append(dataset)
    return dataset_list


class ECGSingleDataset(Dataset):
    def __init__(self, folder_path: Path, stage: str):
        self.__folder_path = folder_path
        self.__stage = stage
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.data = np.load(
            os.path.join(str(self.__folder_path), self.__stage + "_data.npy")
        )
        self.in_time_matrix = np.load(
            os.path.join(str(self.__folder_path), self.__stage + "_data_in_times.npy")
        )
        self.in_chan_matrix = np.load(
            os.path.join(str(self.__folder_path), self.__stage + "_data_in_chans.npy")
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        return self.data[idx], self.in_chan_matrix[idx], self.in_time_matrix[idx]


class ECGDataset(Dataset):
    def __init__(self, folder_paths: folder_path, stage):
        self.__folder_paths = folder_paths
        self.__datasets = []
        self.__dataset_start_idxes = []
        self.__stage = stage

        self.__length = None
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [
            ECGSingleDataset(folder_path, self.__stage)
            for folder_path in self.__folder_paths
        ]

        sample_num = 0
        for dataset in self.__datasets:
            self.__dataset_start_idxes.append(sample_num)
            sample_num += len(dataset)
        self.__length = sample_num

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_start_idxes, idx) - 1
        item_idx = idx - self.__dataset_start_idxes[dataset_idx]
        return self.__datasets[dataset_idx][item_idx]


class ECGDatasetFinetune(Dataset):

    def __init__(
        self, data_folder_path, stage, split_ratio=1.0, sampling_method="random"
    ):
        self.stage = stage
        self.split_ratio = split_ratio
        self.sampling_method = sampling_method
        self.data_path = os.path.join(data_folder_path, stage + "_data.npy")
        self.label_path = os.path.join(data_folder_path, stage + "_labels.npy")
        self.chan_matrix_path = os.path.join(
            data_folder_path, stage + "_data_in_chans.npy"
        )
        self.time_matrix_path = os.path.join(
            data_folder_path, stage + "_data_in_times.npy"
        )
        self.data, self.labels, self.in_chan_matrix, self.in_time_matrix = (
            self.load_data()
        )
        self.prepare_data()
        self.print_class_distribution()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        in_chan_matrix = self.in_chan_matrix[idx]
        in_time_matrix = self.in_time_matrix[idx]
        return sample, label, in_chan_matrix, in_time_matrix

    def random_sampling(self, data, labels, in_chan_matrix, in_time_matrix):
        sample_indices = random.sample(
            range(len(data)), int(len(data) * self.split_ratio)
        )
        data = data[sample_indices]
        labels = labels[sample_indices]
        in_chan_matrix = in_chan_matrix[sample_indices]
        in_time_matrix = in_time_matrix[sample_indices]
        return data, labels, in_chan_matrix, in_time_matrix


    def stratified_sampling(self, data, labels, in_chan_matrix, in_time_matrix):
        y = np.argmax(labels.numpy(), axis=1)
        unique_classes = np.unique(y)

        # Initialize lists to collect the selected indices
        train_indices = []

        for cls in unique_classes:
            class_indices = np.where(y == cls)[0]
            np.random.shuffle(class_indices)

            # Calculate the number of samples to select from this class
            n_samples = max(1, int(len(class_indices) * self.split_ratio))

            # Add selected indices to the train_indices list
            train_indices.extend(class_indices[:n_samples])

        # Convert train_indices to a numpy array and shuffle it
        train_indices = np.array(train_indices)
        np.random.shuffle(train_indices)

        # Subset the data, labels, in_chan_matrix, and in_time_matrix
        data = data[train_indices]
        labels = labels[train_indices]
        in_chan_matrix = in_chan_matrix[train_indices]
        in_time_matrix = in_time_matrix[train_indices]

        return data, labels, in_chan_matrix, in_time_matrix

    def load_data(self):
        data = torch.from_numpy(np.load(self.data_path,allow_pickle=True)).float()
        labels = torch.from_numpy(np.load(self.label_path,allow_pickle=True).astype(np.int64))
        in_chan_matrix = torch.from_numpy(np.load(self.chan_matrix_path,allow_pickle=True))
        in_time_matrix = torch.from_numpy(np.load(self.time_matrix_path,allow_pickle=True))

        return data, labels, in_chan_matrix, in_time_matrix

    def print_class_distribution(self):
        label_counts = torch.sum(self.labels, dim=0)
        print(f"==================== stage {self.stage}'s class distribution ====================")
        for class_index, count in enumerate(label_counts):
            print(f"Class {class_index}: {int(count.item())} samples")

    def prepare_data(self):
        if self.stage == "train" and self.split_ratio < 1.0:
            if self.sampling_method == "random":
                self.data, self.labels, self.in_chan_matrix, self.in_time_matrix = (
                    self.random_sampling(
                        self.data, self.labels, self.in_chan_matrix, self.in_time_matrix
                    )
                )
            elif self.sampling_method == "stratified":
                self.data, self.labels, self.in_chan_matrix, self.in_time_matrix = (
                    self.stratified_sampling(
                        self.data, self.labels, self.in_chan_matrix, self.in_time_matrix
                    )
                )

        # print(self.labels)


def prepare_finetune_dataset(data_folder_path, split_ratio, sampling_method):
    # seed = 0
    # np.random.seed(seed)
    train_dataset = ECGDatasetFinetune(
        data_folder_path=data_folder_path,
        stage="train",
        split_ratio=split_ratio,
        sampling_method=sampling_method,
    )
    val_dataset = ECGDatasetFinetune(data_folder_path=data_folder_path, stage="val")
    test_dataset = ECGDatasetFinetune(data_folder_path=data_folder_path, stage="test")
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # datasets_train = [[Path("../../Datasets/PTBXL_QRS/all/data")]]
    # dataset_train_list = build_ECGpretrain_dataset(datasets_train, "train")
    # dataloader = torch.utils.data.DataLoader(dataset_train_list[0], batch_size=2, shuffle=True)
    # for step, (batch) in enumerate(dataloader):
    #     print(batch[0].shape)
    #     print(batch[1].shape)
    #     print(batch[2].shape)

    # dataset = ECGDatasetFinetune(
    #     data_folder_path="./Datasets/PTBXL_QRS/rhythm", stage="train"
    # )
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    # for batch in dataloader:
    #     print(batch[0].shape)
    #     print(batch[1].shape)
    #     print(batch[2].shape)
    #     print(batch[3].shape)
    #     break

    train_dataset, test_dataset, val_dataset = prepare_finetune_dataset("./datasets/ecg_datasets/PTBXL_QRS/rhythm",0.01,"random")
    data_loader_val = torch.utils.data.DataLoader(val_dataset,batch_size=int(1.5 * 2),drop_last=False,)
    for batch in data_loader_val:
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        break
    print(len(data_loader_val))
    train_dataset, test_dataset, val_dataset = prepare_finetune_dataset("./datasets/ecg_datasets/PTBXL_QRS/rhythm",0.1,"random")
    data_loader_val = torch.utils.data.DataLoader(val_dataset,batch_size=int(1.5 * 2),drop_last=False,)
    for batch in data_loader_val:
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        break
    print(len(data_loader_val))
