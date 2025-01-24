OMP_NUM_THREADS=32 

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes=1 --master_port 49207 --nproc_per_node=8  run_heartlang_pretraining.py\
        --output_dir  checkpoints/pretrain/MIMIC-IV \
        --log_dir  log/pretrain/MIMIC-IV \
        --model  HeartLang \
        --tokenizer_model vqhbr \
        --tokenizer_weight  checkpoints/vqhbr/MIMIC-IV/checkpoint-100.pth \
        --batch_size 64 \
        --lr 5e-4 \
        --warmup_epochs 5 \
        --clip_grad 3.0 \
        --layer_scale_init_value 0.1 \
        --opt_betas 0.9 0.98 \
        --opt_eps 1e-8  \
        --epochs 200 \
        --save_ckpt_freq 10 \
        --codebook_size 8192 \
        --codebook_dim 128 \
        --gradient_accumulation_steps 1 \
        --world_size 8
        