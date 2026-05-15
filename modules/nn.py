import math
import numpy as np
import torch as th
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F

class PositionalEncoding(Module):
    def __init__(self, d_model, dropout=0.0, max_len=1000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)

    
    
class TimeEmbedding(Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, 4*dim),
            nn.SiLU(),
            nn.Linear(4*dim, dim)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FeedForwardBlock(Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4*dim, dim)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
    
class MHA(nn.Module):
    def __init__(self, d_model, heads,batch_first=True, dropout = 0.0):
        super().__init__()

        self.heads = heads
        self.head_size = head_size = d_model // heads
        self.d_model = d_model
        self.batch_first = batch_first
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.output_layer = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):

        batch_size = k.size(0)
        heads = self.heads
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = q.view(batch_size, -1, heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, -1, heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, heads, self.head_size).transpose(1, 2)
        q = q / math.sqrt(self.head_size)
        scores = th.matmul(q, k.transpose(2,3))

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        if key_padding_mask is not None:
            scores =scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(1), -th.inf)

        attention = self.softmax(scores)
        attention = self.dropout(attention)
        context = th.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, heads * self.head_size)

        output = self.output_layer(context)

        return output
  
def create_attn_block(attn_type, *args, **kwargs):
    if attn_type == 'mha':
        return MHA(*args, **kwargs)
    else:
        raise ValueError(f'Unknown attention type: {attn_type}')
        

class AdaLNTransformerDecoderBlock(Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        self_attn_dropout: float = 0.0,
        cross_attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        causal: bool = False,
        attn_type: str = 'mha'
    ):
        """
        :param dim: dim for each head
        """
        super().__init__()
        self.causal = causal
        hidden_dim = int(dim * heads)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = create_attn_block(
            attn_type, hidden_dim, heads, 
            batch_first=True
        )
        self.cross_attn = create_attn_block(
            attn_type, hidden_dim, heads,
            batch_first=True
        )
        self.feedforward = FeedForwardBlock(hidden_dim, dropout=ff_dropout)
        self.emb_layers = nn.Sequential(
            nn.Mish(),
            nn.Linear(emb_dim, 9*hidden_dim)
        )
        nn.init.zeros_(self.emb_layers[1].weight)
        nn.init.zeros_(self.emb_layers[1].bias)

        self.text_emb = nn.Linear(768, emb_dim)
        
    def forward(
        self,
        x: Tensor,
        mem: Tensor,
        emb: Tensor,
        padding_mask: Tensor = None,
        mem_padding_mask: Tensor = None
    ):
        (
            a1, a2, a3, 
            b1, b2, b3, 
            g1, g2, g3
        ) = self.emb_layers(emb).chunk(9, dim=-1)
    
        if self.causal:
            T = x.size(1)
            mask = th.triu(th.ones(T, T, device=x.device), diagonal=1).bool()
        else:
            mask = None
        h = self.norm1(x) * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
        h = self.attn(
            h, h, h, 
            attn_mask=mask,
            key_padding_mask=padding_mask
        )
        x = x + a1.unsqueeze(1) * h
        h = self.norm2(x) * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
        h = self.cross_attn(
            h, mem, mem)
        x = x + a2.unsqueeze(1) * h
        h = self.norm3(x) * (1 + g3.unsqueeze(1)) + b3.unsqueeze(1)
        x = x + a3.unsqueeze(1) * self.feedforward(h)
        
        return x
    
    
class  AdaLNTransformerDecoder(Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dim: int,
        heads: int,
        emb_dim: int,
        num_layers: int,
        pe_dropout: float = 0.0,
        self_attn_dropout: float = 0.0,
        cross_attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        causal: bool = False,
        attn_type: str = 'mha'
    ):

        super().__init__()
        hidden_dim = int(dim * heads)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        if attn_type == 'tisa':
            self.pe = nn.Identity()
        else:
            self.pe = PositionalEncoding(hidden_dim, dropout=pe_dropout, batch_first=True)
        self.layers = nn.ModuleList([
            AdaLNTransformerDecoderBlock(
                dim, heads, emb_dim,
                self_attn_dropout=self_attn_dropout, 
                cross_attn_dropout=cross_attn_dropout,
                ff_dropout=ff_dropout,
                causal=causal,
                attn_type=attn_type
            )
            for _ in range(num_layers)
        ])
        self.out_layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(
        self,
        x: Tensor,
        mem: Tensor,
        emb: Tensor,
        padding_mask: Tensor = None,
        mem_padding_mask: Tensor = None
    ):
        x = self.in_proj(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(
                x, mem, emb,
                padding_mask=padding_mask,
                mem_padding_mask=mem_padding_mask
            )
        x = self.out_layers(x)
        return x
    
    
class Block1D(Module):
    def __init__(self, dim, dim_out, groups=16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x):
        output = self.block(x)
        return output
    
    
class ResnetBlock1D(Module):
    def __init__(self, dim, dim_out, emb_dim, groups=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, emb):
        h = self.block1(x)
        h += self.mlp(emb).unsqueeze(-1)
        h = self.block2(h)
        output = h + self.res_conv(x)
        return output
    
    
class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)
    
    
class Upsample1D(nn.Module):

    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs