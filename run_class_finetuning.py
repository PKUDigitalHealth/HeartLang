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
from pyexpat import model
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.utils import ModelEma
from utils.optim_factory import (
    create_optimizer,
    get_parameter_groups,
    LayerDecayValueAssigner,
)
from engine_for_finetuning import train_one_epoch, evaluate
from utils.utils import (
    NativeScalerWithGradNormCount as NativeScaler,
    freeze_except_prefix,
    freeze_except_prefixes,
    freeze_specific_layers,
    get_trainable_layers,
    save_results_as_csv,
    set_seed,
)
import utils.utils as utils
from scipy import interpolate
import modeling_finetune as modeling_finetune
from utils.QRSDataset import prepare_finetune_dataset
import torch.distributed as dist


def get_args():
    parser = argparse.ArgumentParser(
        "HeartLang fine-tuning script for ECG classification",
        add_help=False,
    )
    parser.add_argument("--is_binary", action="store_true", default=True)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_freq", default=5, type=int)

    # robust evaluation
    parser.add_argument(
        "--robust_test", default=None, type=str, help="robust evaluation dataset"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="HeartLang_finetune_base",
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

    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )

    parser.add_argument(
        "--disable_eval_during_finetuning", action="store_true", default=False
    )

    parser.add_argument("--model_ema", action="store_true", default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999, help="")
    parser.add_argument(
        "--model_ema_force_cpu", action="store_true", default=False, help=""
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
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument("--layer_decay", type=float, default=0.9)

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
        help="num of steps to warmup LR, will overload warmup_epochs if set > 0",
    )

    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Finetuning params
    parser.add_argument("--finetune", default=False, help="finetune from checkpoint")
    parser.add_argument("--trainable", default="all", help="trainable layers")
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)
    parser.add_argument("--model_filter_name", default="gzp", type=str)
    parser.add_argument("--init_scale", default=0.001, type=float)
    parser.add_argument("--use_mean_pooling", action="store_true")
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument("--use_cls", action="store_false", dest="use_mean_pooling")

    # Dataset parameters
    parser.add_argument(
        "--sampling_method", default="random", help="train dataset sampling method"
    )
    parser.add_argument(
        "--split_ratio", default=1.0, type=float, help="train dataset split ratio"
    )
    parser.add_argument("--dataset_dir", default="", help="dataset path")
    parser.add_argument(
        "--nb_classes", default=0, type=int, help="number of the classification types"
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

    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt")
    parser.set_defaults(save_ckpt=True)

    parser.add_argument(
        "--start_epoch", default=1, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
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

    parser.add_argument("--enable_deepspeed", action="store_true", default=False)

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig

            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def get_models(args):
    model = create_model(args.model, pretrained=False, num_classes=args.nb_classes)

    return model


def get_dataset(args):
    train_dataset, val_dataset, test_dataset = prepare_finetune_dataset(
        args.dataset_dir, args.split_ratio, args.sampling_method
    )
    metrics = [
        "accuracy",
        "f1",
        "recall",
        "precision",
    ]
    return train_dataset, val_dataset, test_dataset, metrics


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    set_seed(args.seed)

    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
    dataset_train, dataset_val, dataset_test, metrics = get_dataset(args)

    val_invalid_columns = utils.check_dataset_labels(dataset_val)
    print("val dataset has invalid columns: ", val_invalid_columns)
    test_invalid_columns = utils.check_dataset_labels(dataset_test)
    print("test dataset has invalid columns: ", test_invalid_columns)

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
            if type(dataset_test) == list:
                sampler_test = [
                    torch.utils.data.DistributedSampler(
                        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
                    )
                    for dataset in dataset_test
                ]
            else:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test,
                    num_replicas=num_tasks,
                    rank=global_rank,
                    shuffle=False,
                )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    num_gpus = utils.get_world_size()
    total_batch_size = args.batch_size * args.update_freq * num_gpus

    while total_batch_size > len(dataset_train) and args.batch_size > 1:
        args.batch_size //= 2
        total_batch_size = args.batch_size * args.update_freq * num_gpus
    args.batch_size = max(args.batch_size, 1)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            # batch_size=int(2 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        if type(dataset_test) == list:
            data_loader_test = [
                torch.utils.data.DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size=int(1.5 * args.batch_size),
                    # batch_size=int(2 * args.batch_size),
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=False,
                )
                for dataset, sampler in zip(dataset_test, sampler_test)
            ]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test,
                sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
    else:
        data_loader_val = None
        data_loader_test = None

    model = get_models(args)

    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.finetune, map_location="cpu")

        print("checkpoint.keys():", checkpoint.keys())

        print("=" * 25 + "Load ckpt from %s" % args.finetune + "=" * 25)

        checkpoint_model = None
        for model_key in args.model_key.split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    else:
        model.randomly_initialize_weights()
        print("=" * 25 + "Random Init" + "=" * 25)
        pass

    model.to(device)

    if args.trainable == "linear":
        freeze_except_prefix(model, "mlp_head")
    if args.trainable == "adapter":
        freeze_specific_layers(model, ["adapter", "mlp_head"])
    elif args.trainable == "all":
        pass

    trainable_layers = get_trainable_layers(model)
    print("Trainable layers:", trainable_layers)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print("number of params:", n_parameters)

    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print(
        "Batch size on one GPU = %d, %d GPUs total batch size = %d"
        % (args.batch_size, args.world_size, total_batch_size)
    )
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(
                args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
            )
        )
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model,
            args.weight_decay,
            skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None,
        )
        model, optimizer, _, _ = ds_init(
            args=args,
            model=model,
            model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print(
            "model.gradient_accumulation_steps() = %d"
            % model.gradient_accumulation_steps()
        )
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True
            )
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args,
            model_without_ddp,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
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

    if args.is_binary == True:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_ema,
    )

    if args.eval:
        print("Test batch size = %d" % data_loader_test.batch_size)
        print("Number of test examples = %d" % len(dataset_test))
        print("Number of testing steps in one epoch = %d" % len(data_loader_test))
        test_stats = evaluate(
            data_loader_test,
            model,
            device,
            header="Test:",
            metrics=metrics,
            is_binary=args.is_binary,
            dataset_dir=args.dataset_dir,
        )

        roc_auc = test_stats["roc_auc"]
        print(f"====== {args.log_dir} ======")
        print(f"====== AUC: {roc_auc} ======")

        if dist.get_rank() == 0:
            results_save_path = "results/"
            os.makedirs(results_save_path, exist_ok=True)
            save_results_as_csv(roc_auc, results_save_path, "finetune", args)

        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_auc = 0.0
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch - 1)
        if log_writer is not None:
            log_writer.set_step(
                (epoch - 1) * num_training_steps_per_epoch * args.update_freq
            )
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            log_writer=log_writer,
            start_steps=(epoch - 1) * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            is_binary=args.is_binary,
        )

        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                model_ema=model_ema,
                save_ckpt_freq=args.save_ckpt_freq,
            )

        # dist.barrier()
        if data_loader_val is not None:
            print("Eval batch size = %d" % data_loader_val.batch_size)
            print("Number of eval examples = %d" % len(dataset_val))
            print("Number of eval steps in one epoch = %d" % len(data_loader_val))
            val_stats = evaluate(
                data_loader_val,
                model,
                device,
                header="Val:",
                metrics=metrics,
                is_binary=args.is_binary,
                dataset_dir=args.dataset_dir,
            )
            print(
                f"AUC of the network on the {len(dataset_val)} val ECG: {val_stats['roc_auc']*100:.2f}%"
            )

            if max_auc < val_stats["roc_auc"]:
                max_auc = val_stats["roc_auc"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                        model_ema=model_ema,
                    )

            print(f"Max auc val: {max_auc*100:.2f}%")
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == "accuracy":
                        log_writer.update(accuracy=value, head="val", step=epoch)
                    elif key == "balanced_accuracy":
                        log_writer.update(
                            balanced_accuracy=value, head="val", step=epoch
                        )
                    elif key == "f1_weighted":
                        log_writer.update(f1_weighted=value, head="val", step=epoch)
                    elif key == "pr_auc":
                        log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == "roc_auc":
                        log_writer.update(roc_auc=value, head="val", step=epoch)
                    elif key == "cohen_kappa":
                        log_writer.update(cohen_kappa=value, head="val", step=epoch)
                    elif key == "loss":
                        log_writer.update(loss=value, head="val", step=epoch)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        else:
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
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
