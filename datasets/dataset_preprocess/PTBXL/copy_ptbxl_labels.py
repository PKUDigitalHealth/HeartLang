import os
import shutil

# 定义源和目标目录的根路径
source_root = "/hot_data/jinjiarui/run/HeartLang/datasets/ecg_datasets/PTBXL"
target_root = "/hot_data/jinjiarui/run/HeartLang/datasets/ecg_datasets/PTBXL_QRS"

# 定义所有可能的子目录
subdirs = ["all", "diagnostic", "form", "rhythm", "subdiagnostic", "superdiagnostic"]

# 定义所有可能的文件前缀
file_prefixes = ["train", "val", "test"]

# 遍历所有子目录和文件前缀
for subdir in subdirs:
    for prefix in file_prefixes:
        # 构造源文件路径
        source_file = os.path.join(source_root, subdir, f"{prefix}_labels.npy")

        # 构造目标文件路径
        target_dir = os.path.join(target_root, subdir)
        target_file = os.path.join(target_dir, f"{prefix}_labels.npy")

        # 创建目标目录（如果不存在）
        os.makedirs(target_dir, exist_ok=True)

        # 复制文件
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            print(f"Copied {source_file} to {target_file}")
        else:
            print(f"{source_file} does not exist")

print("All files have been copied successfully.")
