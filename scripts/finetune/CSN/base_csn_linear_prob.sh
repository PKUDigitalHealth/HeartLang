#!/bin/bash

# Define the split_ratios array to use  
split_ratios=(0.01 0.1 1)

sampling_methods=("random")

export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=1

generate_random_port() {
    while :; do
        PORT=$(( ( RANDOM % 63000 )  + 2000 ))
        ss -lpn | grep -q ":$PORT " || break
    done
    echo $PORT
}

# Train
for ratio in "${split_ratios[@]}"
do
    for method in "${sampling_methods[@]}"
    do
        PORT=$(generate_random_port)
        echo "Running training with split_ratio: $ratio and sampling_method: $method on port: $PORT"
        torchrun --nnodes=1 --master_port=$PORT --nproc_per_node=1 run_class_finetuning.py \
            --dataset_dir datasets/ecg_datasets/CSN_QRS/data \
            --output_dir checkpoints/finetune/csn/finetune_csn_base_linear_${ratio}_${method}/ \
            --log_dir log/finetune/finetune_csn_base_linear_${ratio}_${method} \
            --model HeartLang_finetune_base \
            --finetune checkpoints/heartlang_base/checkpoint-200.pth \
            --trainable linear \
            --split_ratio $ratio \
            --sampling_method $method \
            --weight_decay 0.05 \
            --batch_size 256 \
            --lr 5e-3 \
            --update_freq 1 \
            --warmup_epochs 10 \
            --epochs 100 \
            --layer_decay 0.9 \
            --dist_eval \
            --save_ckpt_freq 100 \
            --seed 0 \
            --is_binary \
            --nb_classes 38 \
            --world_size 1
    done
done

# Test

for ratio in "${split_ratios[@]}"
do
    for method in "${sampling_methods[@]}"
    do
        PORT=$(generate_random_port)
        echo "Running testing with split_ratio: $ratio and sampling_method: $method on port: $PORT"
        torchrun --nnodes=1 --master_port=$PORT --nproc_per_node=1 run_class_finetuning.py \
            --dataset_dir datasets/ecg_datasets/CSN_QRS/data \
            --output_dir checkpoints/finetune/csn/finetune_csn_base_linear_${ratio}_${method}/ \
            --log_dir log/finetune_test/finetune_csn_base_linear_${ratio}_${method} \
            --model HeartLang_finetune_base \
            --eval \
            --trainable linear \
            --split_ratio $ratio \
            --sampling_method $method \
            --batch_size 256 \
            --dist_eval \
            --seed 0 \
            --is_binary \
            --nb_classes 38 \
            --world_size 1
    done
done