import os
import shutil

# 定义要处理的根目录
root_dir = "/hot_data/jinjiarui/run/HeartLang/datasets/ecg_datasets/PTBXL"

# 定义需要处理的子目录列表
subdirs = ["all", "diagnostic", "form", "rhythm", "subdiagnostic", "superdiagnostic"]

for subdir in subdirs:
    data_dir = os.path.join(root_dir, subdir, "data")

    if os.path.exists(data_dir):
        # 列出data目录下的所有文件
        files = os.listdir(data_dir)

        for file_name in files:
            # 构造源文件路径和目标文件路径
            source_file = os.path.join(data_dir, file_name)
            target_file = os.path.join(root_dir, subdir, file_name)

            # 移动文件
            shutil.move(source_file, target_file)
            print(f"Moved {source_file} to {target_file}")

        # 删除data目录
        os.rmdir(data_dir)
        print(f"Deleted directory {data_dir}")
    else:
        print(f"{data_dir} does not exist")

print("All files have been moved and data directories have been deleted successfully.")
