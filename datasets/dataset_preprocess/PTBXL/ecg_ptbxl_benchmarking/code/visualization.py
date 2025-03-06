from utils.visualization_ecg import visulization_ecg

if __name__ == "__main__":
    data_path = "/home/jinjiarui/run/ECG-Learning/PTB_XL/ecg_ptbxl_benchmarking/output/exp_diagnostic/data/test_data.npy"
    visulization_ecg(data_path, 0, True)
