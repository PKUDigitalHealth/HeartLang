#!/bin/bash

export OMP_NUM_THREADS=48

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes=1 --nproc_per_node=8 run_vqhbr_training.py \
    --output_dir checkpoints/vqhbr/MIMIC-IV/ \
    --log_dir log/vqhbr/MIMIC-IV/ \
    --model vqhbr \
    --codebook_n_emd 8192 \
    --codebook_emd_dim 128 \
    --quantize_kmeans_init \
    --batch_size 64 \
    --opt adamw \
    --opt_betas 0.9 0.99 \
    --weight_decay 1e-4 \
    --warmup_epochs 10 \
    --epochs 100 \
    --save_ckpt_freq 10 \
    --world_size 8 \
    --val_freq 10 \
    --lr 5e-5 \
    --min_lr 1e-5

echo "Training script has completed execution."
