import numpy as np
import matplotlib.pyplot as plt
import os

def visulization_ecg(ecg_data_path, index=0, random=True):
    ecg_data = np.load(ecg_data_path)
    if random:
        sample_index = np.random.randint(ecg_data.shape[0])
    else:
        sample_index = index

    sample = ecg_data[sample_index, :, :]
    length, channel = sample.shape

    fig, axs = plt.subplots(
        channel, 1, figsize=(12, 2 * channel), constrained_layout=True
    )

    for i in range(channel):
        axs[i].plot(sample[:, i], label=f"Channel {i+1}")
        axs[i].legend(loc="upper right")
        axs[i].set_xlim([0, length])
        axs[i].set_ylim([sample[:,i].min(), sample[:,i].max()])


    save_path = os.path.join(os.path.dirname(os.path.dirname(ecg_data_path)), "figs")
    print(f"save_path: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"ecg_{sample_index}_visualization.png"))
