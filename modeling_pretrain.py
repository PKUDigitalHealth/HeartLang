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

from typing import Tuple
import math
import torch
from timm import create_model
from timm.models import register_model
from torch import nn
from einops import rearrange, repeat, pack
import modeling_vqhbr as modeling_vqhbr


def random_masking(x, mask_ratio):
    """
    intput x: [bs, seq_len, embed_dim] ; mask_ratio: float
    output mask matrix: (bool)[bs, seq_len]
    """
    bs, seq_len, dim = x.shape
    len_keep = int(seq_len * (1 - mask_ratio))
    noise = torch.rand(bs, seq_len, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.ones([bs, seq_len], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return mask.to(torch.bool)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, heads=8, head_dim=64, dropout=0.0):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == embed_dim)

        self.heads = heads
        self.scale = head_dim**-0.5

        self.norm = nn.LayerNorm(embed_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            embed_dim=embed_dim,
                            heads=heads,
                            head_dim=dim_head,
                            dropout=dropout,
                        ),
                        FeedForward(embed_dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class SpatialEmbedding(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(SpatialEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, x, in_chan_matrix):
        spatial_embeddings = self.embed(
            in_chan_matrix
        )  # [batch_size, seq_len, embed_dim]
        return x + spatial_embeddings


class TemporalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(TemporalEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, x, time_index_matrix):
        temporal_embeddings = self.embed(
            time_index_matrix
        )  # [batch_size, seq_len, embed_dim]
        return x + temporal_embeddings


class ST_ECGFormer(nn.Module):
    def __init__(
        self,
        seq_len,
        time_window,
        depth,
        embed_dim,
        heads,
        mlp_dim,
        dim_head=64,
        vocab_size=1024,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super().__init__()
        self.token_embed = TokenEmbedding(c_in=time_window, d_model=embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(embed_dim))

        # learnable position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))

        self.spa_embed = SpatialEmbedding(num_embeddings=16, embed_dim=embed_dim)
        self.tem_embed = TemporalEmbedding(num_embeddings=16, embed_dim=embed_dim)

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.norm_layer = nn.LayerNorm(embed_dim)

        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward_feature(
        self, x, mask_bool_matrix=None, in_chan_matrix=None, in_time_matrix=None
    ):
        x = self.token_embed(x)
        b, seq_len, embed_dim = x.shape

        if mask_bool_matrix is None:
            mask_bool_matrix = torch.zeros((b, seq_len), dtype=torch.bool).to(x.device)
        mask_tokens = self.mask_token.expand(b, seq_len, -1)
        w = mask_bool_matrix.unsqueeze(-1).type_as(mask_tokens)
        x = x * (1 - w) + mask_tokens * w

        if in_chan_matrix is not None:
            x = self.spa_embed(x, in_chan_matrix)

        if in_time_matrix is not None:
            x = self.tem_embed(x, in_time_matrix)

        cls_tokens = repeat(self.cls_token, "d -> b d", b=b)
        x, ps = pack([cls_tokens, x], "b * d")
        x += self.pos_embed

        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm_layer(x)

        return x

    def forward(
        self,
        x,
        mask_bool_matrix=None,
        in_chan_matrix=None,
        in_time_matrix=None,
        return_all_tokens=False,
        return_qrs_tokens=False,
    ):
        x = self.forward_feature(x, mask_bool_matrix, in_chan_matrix, in_time_matrix)
        if return_all_tokens:
            return x
        x = x[:, 1:]
        if return_qrs_tokens:
            return x
        else:
            return self.lm_head(x[mask_bool_matrix])


def get_model_default_params():
    return dict(
        seq_len=256,
        time_window=96,
        embed_dim=768,
        depth=8,
        heads=4,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )


@register_model
def HeartLang(pretrained=False, **kwargs):
    config = get_model_default_params()
    if "vocab_size" in kwargs:
        config["vocab_size"] = kwargs["vocab_size"]
        _ = kwargs.pop("vocab_size")
    else:
        config["vocab_size"] = 8192

    config["depth"] = 12
    config["heads"] = 8

    config["mlp_dim"] = 1024

    model = ST_ECGFormer(**config)
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == "__main__":
    sample = torch.randn(4, 256, 96)
    mask_bool_matrix = random_masking(sample, 0.5)
    in_chan_matrix = torch.randint(0, 13, (4, 256))
    in_time_matrix = torch.randint(0, 11, (4, 256))
    print("mask_bool_matrix: ", mask_bool_matrix.shape)

    model = create_model("HeartLang", pretrained=False)
    out = model(sample, mask_bool_matrix, in_chan_matrix, in_time_matrix)
    print("output: ", out.shape)

    vqhbr = create_model("vqhbr", pretrained=False, n_code=1024)
    input_ids = vqhbr.get_codebook_indices(sample, in_chan_matrix, in_time_matrix)
    labels = input_ids[mask_bool_matrix]
    print(labels)
    print("vqhbr output shape:", labels.shape)
    loss = nn.CrossEntropyLoss()(out, labels)
    print("Loss:", loss.item())
