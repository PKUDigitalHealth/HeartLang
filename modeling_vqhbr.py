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

import os
import pickle
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from sklearn.manifold import TSNE
from backbone_vqhbr import VqhbrBackbone
from utils.norm_ema_quantizer import NormEMAVectorQuantizer
from matplotlib.colors import ListedColormap
from collections import Counter
from matplotlib.patches import Circle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib import colors


class VQHBR(nn.Module):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        n_embed=8192,
        codebook_embed_dim=128,
        decay=0.99,
        quantize_kmeans_init=True,
        decoder_out_dim=96,
        smooth_l1_loss=False,
        **kwargs,
    ):
        super().__init__()

        print("Final encoder config", encoder_config)
        self.encoder = VqhbrBackbone(**encoder_config)
        print("Final decoder config", decoder_config)
        self.decoder = VqhbrBackbone(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed,
            embedding_dim=codebook_embed_dim,
            beta=1.0,
            kmeans_init=quantize_kmeans_init,
            decay=decay,
        )

        self.decoder_out_dim = decoder_out_dim

        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config["embed_dim"], encoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Linear(encoder_config["embed_dim"], codebook_embed_dim),  # for quantize
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config["embed_dim"], decoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Linear(decoder_config["embed_dim"], self.decoder_out_dim),
        )

        self.kwargs = kwargs

        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

        self.step = 0
        self.plot_once = True
        self.plot_save_path = f"figs/{self.kwargs['model']}/"
        os.makedirs(self.plot_save_path, exist_ok=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            # "quantize.embedding.weight",
            # "decoder.cls_token",
            # "decoder.pos_embed",
            # "decoder.time_embed",
            # "encoder.cls_token",
            # "encoder.pos_embed",
            # "encoder.time_embed",
        }

    @property
    def device(self):
        return self.decoder.cls_token.device

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, in_chan_matrix=None, in_time_matrix=None, **kwargs):
        quantize, embed_ind, loss = self.encode(
            data, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
        )
        output = {}
        output["token"] = embed_ind.view(data.shape[0], -1)
        output["input_img"] = data
        output["quantize"] = quantize

        return output

    def encode(self, x, in_chan_matrix=None, in_time_matrix=None):
        bs, seq_len, time_window = x.shape
        encoder_features = self.encoder(
            x,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
            return_qrs_tokens=True,
        )
        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(
                encoder_features.type_as(self.encode_task_layer[-1].weight)
            )
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)
        return quantize, embed_ind, loss

    def decode(self, quantize, in_chan_matrix=None, in_time_matrix=None, **kwargs):
        decoder_features = self.decoder(
            quantize,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
            return_qrs_tokens=True,
        )
        rec = self.decode_task_layer(decoder_features)
        return rec

    def get_codebook_indices(
        self, x, in_chan_matrix=None, in_time_matrix=None, **kwargs
    ):
        # for HeartLang pre-training
        return self.get_tokens(
            x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix, **kwargs
        )["token"]

    def get_codebook_quantize(
        self, x, in_chan_matrix=None, in_time_matrix=None, **kwargs
    ):
        # for HeartLang pre-training
        return self.get_tokens(
            x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix, **kwargs
        )["quantize"]

    def visualize_codebook(self, data, codes, indexs, target_indexs):
        for target_idx in target_indexs:
            # get all positions of idx
            positions = np.where(indexs == target_idx)[0]

            folder_path = os.path.join(
                # self.plot_save_path, "codebook", f"idx_{target_idx}"
                self.plot_save_path,
                "codebook",
                f"all_samples_{target_idx}",
            )
            os.makedirs(folder_path, exist_ok=True)
            
            random_indices = np.random.choice(positions, size=100, replace=False)
            corresponding_data = data[random_indices]
            np.save(
                os.path.join(
                    self.plot_save_path,
                    "codebook",
                    f"corresponding_data_{target_idx}.npy",
                ),
                corresponding_data,
            )
            num_signals = corresponding_data.shape[0] 
            rows = cols = int(np.ceil(np.sqrt(num_signals)))

            plt.figure(figsize=(cols * 3, rows * 3))

            for i, signal in enumerate(corresponding_data):
                plt.subplot(rows, cols, i + 1)
                plt.plot(signal) 
                plt.title(f"Sample {i}", fontsize=8)
                plt.axis("off")

            plt.tight_layout()

            output_file = os.path.join(folder_path, f"all_samples_{target_idx}.png")
            plt.savefig(output_file)
            plt.close()

            for i, signal in enumerate(corresponding_data):

                plt.figure(dpi=300)
                plt.plot(signal,linewidth=2)
                plt.gca().set_xticks([]) 
                plt.gca().set_yticks([])  

                # plt.title(f"Idx {target_idx} - Sample {i}")


                output_file = os.path.join(folder_path, f"sample_{i}.png")
                plt.savefig(output_file, dpi=300)
                plt.close()

            print(f"Visualization for idx {target_idx} saved in {folder_path}")

    def calculate_rec_loss(self, rec, target):
        # target = rearrange(target, "b n a c -> b (n a) c")
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / std
        return x

    def plot_ecg(self, x, xrec):
        beats, length = x.shape
        cols = int(np.ceil(np.sqrt(beats)))
        rows = int(np.ceil(beats / cols))

        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 3, rows * 3), constrained_layout=True
        )

        axs = axs.ravel()

        for i in range(beats):
            axs[i].plot(x[i], color="blue", label=f"Beat {i + 1} Original")
            axs[i].plot(xrec[i], color="red", label=f"Beat {i + 1} Reconstructed")
            axs[i].legend(loc="upper right")
            axs[i].set_xlim([0, length])

            if np.all(x[i] == 0):
                axs[i].set_ylim([-3, 3])
            else:
                axs[i].set_ylim(
                    [min(x[i].min(), xrec[i].min()), max(x[i].max(), xrec[i].max())]
                )

        for i in range(beats, rows * cols):
            axs[i].axis("off")

        plt.savefig(
            self.plot_save_path
            + f"x_recx_{self.step}_{'train' if self.training else 'val'}.png"
        )
        plt.close(fig)

    def plot_ecg_beats(self, x):
        beats, length = x.shape

        cols = int(np.ceil(np.sqrt(beats))) 
        rows = int(np.ceil(beats / cols)) 

        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 3, rows * 3), constrained_layout=True
        )

        axs = axs.ravel() 

        for i in range(beats):
            axs[i].plot(x[i], color="blue")
            axs[i].set_xlim([0, length])
            # axs[i].axis("off")

            if np.all(x[i] == 0):
                axs[i].set_ylim([-3, 3])
            else:
                axs[i].set_ylim([x[i].min(), x[i].max()])

        for i in range(beats, rows * cols):
            axs[i].axis("off")

        plt.savefig(
            self.plot_save_path
            + f"x_{self.step}_{'train' if self.training else 'val'}_beats.png"
        )
        plt.close(fig)

    def forward(self, x, in_chan_matrix=None, in_time_matrix=None, **kwargs):
        """
        x: shape [bs, seq_len, time_window]
        """
        quantize, embed_ind, emb_loss = self.encode(x, in_chan_matrix, in_time_matrix)
        xrec = self.decode(quantize, in_chan_matrix, in_time_matrix)

        rec_loss = self.calculate_rec_loss(xrec, x)
        loss = emb_loss + rec_loss

        self.plot_ecg_beats(x[0].cpu().detach().numpy())

        if self.plot_once and not self.training:  # and not self.training
            print("=" * 10 + "Ploting..." + "=" * 10)
            x_plot, xrec_plot = (
                x[0].cpu().detach().numpy(),
                xrec[0].cpu().detach().numpy(),
            )
            self.plot_ecg(x_plot, xrec_plot)
            self.plot_once = False

        if self.step % 1000 == 0 and self.training:
            self.plot_once = True
        if self.training:
            self.step += 1

        log = {}
        split = "train" if self.training else "val"
        log[f"{split}/quant_loss"] = emb_loss.detach().mean()
        log[f"{split}/rec_loss"] = rec_loss.detach().mean()
        log[f"{split}/total_loss"] = loss.detach().mean()

        return loss, log


def get_model_default_params():
    return dict(
        seq_len=256,
        time_window=96,
        embed_dim=768,
        depth=4,
        heads=2,
        mlp_dim=2048,
        dropout=0.0,
        emb_dropout=0.0,
    )


@register_model
def vqhbr(
    pretrained=False,
    pretrained_weight=None,
    as_tokenzer=False,
    n_code=8192,
    code_dim=128,
    **kwargs,
):
    encoder_config, decoder_config = (
        get_model_default_params(),
        get_model_default_params(),
    )

    # encoder settings
    encoder_config["depth"] = 4
    encoder_config["heads"] = 2
    encoder_config["Encoder"] = True
    # decoder settings
    decoder_config["code_dim"] = code_dim
    decoder_config["depth"] = 2
    decoder_config["heads"] = 2
    decoder_config["Encoder"] = False
    decoder_out_dim = 96

    kwargs["model"] = "vqhbr"

    model = VQHBR(
        encoder_config,
        decoder_config,
        n_code,
        code_dim,
        decoder_out_dim=decoder_out_dim,
        **kwargs,
    )

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None

        if pretrained_weight.startswith("https"):
            weights = torch.hub.load_state_dict_from_url(
                pretrained_weight, map_location="cpu", check_hash=True
            )
        else:
            weights = torch.load(pretrained_weight, map_location="cpu")

        if "model" in weights:
            weights = weights["model"]
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())

        for k in keys:
            if (
                k.startswith("loss")
                or k.startswith("teacher")
                or k.startswith("scaling")
            ):
                del weights[k]
        model.load_state_dict(weights)
    return model


if __name__ == "__main__":
    x = torch.randn(4, 256, 96)
    model = vqhbr()
    out = model(x)
    print(f"output: {out}")
