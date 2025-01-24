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

import math
import torch
from timm.models import register_model
from torch import nn
from modeling_pretrain import ST_ECGFormer
class ST_ECGFormerClassifier(nn.Module):
    def __init__(
        self,
        seq_len,
        time_window,
        depth,
        embed_dim,
        heads,
        mlp_dim,
        dim_head=64,
        num_classes=None,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.backbone = ST_ECGFormer(
            seq_len=seq_len,
            time_window=time_window,
            depth=depth,
            embed_dim=embed_dim,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        self.depth = depth
        self.mlp_head = nn.Linear(embed_dim, num_classes)
        # for i in range(self.depth):
        #     self.backbone.transformer.layers[i][0].adapter = FFTAdapter(768)

    def forward(
        self, x, in_chan_matrix=None, in_time_matrix=None, return_all_tokens=True
    ):
        cls_token = self.backbone(
            x,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
            return_all_tokens=return_all_tokens,
        )[:, 0]
        output = self.mlp_head(cls_token)
        return output

    def get_num_layers(self):
        return self.depth

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {"pos_embedding", "cls_token", "token_embed"}
        return {}

    def randomly_initialize_weights(self):
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.1, b=0.1)


def get_model_default_params():
    return dict(
        seq_len=256,
        time_window=96,
        num_classes=1000,
        embed_dim=768,
        depth=8,
        heads=4,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )


@register_model
def HeartLang_finetune_base(pretrained=False, **kwargs):
    config = get_model_default_params()
    config["num_classes"] = kwargs["num_classes"]
    config["depth"] = 12
    config["heads"] = 8

    config["mlp_dim"] = 1024
    model = ST_ECGFormerClassifier(**config)

    return model



if __name__ == "__main__":
    model = HeartLang_finetune_base(num_classes=12)
    print(model)
