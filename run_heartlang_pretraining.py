# --------------------------------------------------------
# Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model
# By Jiarui Jin and Haoyu Wang
# Based on BEiT-v2, timm, DeiT, DINO and LaBraM code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/935963004/LaBraM
# ---------------------------------------------------------

import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from timm.models import create_model
from utils.optim_factory import create_optimizer
from utils.QRSDataset import build_ECGpretrain_dataset
from engine_for_pretraining import train_one_epoch
import utils.utils as utils
from utils.utils import NativeScalerWithGradNormCount as NativeScaler, set_seed
import modeling_pretrain as modeling_pretrain
import modeling_vqhbr as modeling_vqhbr


def get_args():
    parser = argparse.ArgumentParser("HeartLang pre-training script", add_help=False)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--save_ckpt_freq", default=20, type=int)

    # tokenizer settings
    parser.add_argument("--tokenizer_weight", type=str)
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        default="vqhbr",
        help="Name of tokenizer to pretrain",
    )
    # Tokenizer parameters
    parser.add_argument(
        "--codebook_size", default=8192, type=int, help="size of codebook"
    )
    parser.add_argument("--codebook_dim", default=128, type=int, help="code dim")

    # Model parameters
    parser.add_argument(
        "--model",
        default="HeartLang",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--layer_scale_init_value",
        default=0.1,
        type=float,
        help="0.1 for base. set 0 to disable layer scale",
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument(
        "--start_epoch", default=1, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)

    return parser.parse_args()


# pretrain model
def get_model(args):
    print(f"Pretrain model: {args.model}")
    model = create_model(args.model, pretrained=False, vocab_size=args.codebook_size)
    return model


# tokenizer
def get_tokenizer(args):
    print(f"Tokenizer model: {args.tokenizer_model}")
    model = create_model(
        args.tokenizer_model,
        pretrained=True,
        pretrained_weight=args.tokenizer_weight,  # path of the tokenizer's weight
        as_tokenzer=True,
        n_code=args.codebook_size,
        code_dim=args.codebook_dim,
    ).eval()
    return model


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    set_seed(args.seed)

    model = get_model(args)
    datasets_train = [
        [
            Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_0"),
            Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_2"),
            Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_3"),
            Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_4"),
            Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_5"),
            Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_6"),
            Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_7"),
        ]
    ]
    dataset_train_list = build_ECGpretrain_dataset(datasets_train, "train")

    # prepare tokenizer
    vqhbr = get_tokenizer(args).to(device)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = (
            sum([len(dataset) for dataset in dataset_train_list])
            // args.batch_size
            // num_tasks
        )

        sampler_train_list = []
        for dataset in dataset_train_list:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            )
            sampler_train_list.append(sampler_train)
            print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train_list = []
    for dataset, sampler in zip(dataset_train_list, sampler_train_list):
        data_loader_train = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_train_list.append(data_loader_train)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params:", n_parameters)

    print("Tokenizer = %s" % str(vqhbr))
    total_batch_size = (
        args.batch_size * utils.get_world_size() * args.gradient_accumulation_steps
    )
    print("LR = %.8f" % args.lr)
    print("Total batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print(
        "Number of training examples per epoch = %d"
        % (total_batch_size * num_training_steps_per_epoch)
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )
    print(
        "Max WD = %.7f, Min WD = %.7f"
        % (max(wd_schedule_values), min(wd_schedule_values))
    )

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch - 1)
        if log_writer is not None:
            log_writer.set_step((epoch - 1) * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model,
            vqhbr,
            data_loader_train_list,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=(epoch - 1) * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            args=args,
        )
        if args.output_dir:
            utils.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                save_ckpt_freq=args.save_ckpt_freq,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
