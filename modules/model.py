import torch as th
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .nn import (
    AdaLNTransformerDecoder, 
    TimeEmbedding,
    Block1D,
    ResnetBlock1D,
    Downsample1D,
    Upsample1D,
    AdaLNTransformerDecoderBlock
)
from .tensor_utils import (
    interpnd, 
    create_padding_mask
)


def init_params(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Estimator(nn.Module):
    def __init__(
        self,
        pose_dim: int,
        dim: int = 64,
        heads: int = 8,
        down_ts: int = 2,
        attn_type: str = 'tisa',
        n_attn_blocks: int = 1,
        num_mid_blocks: int = 6,
        emb_dim_expansion: int = 1,
        disable_mem_mean: bool = True,
        self_attn_dropout: float = 0.0,
        cross_attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.down_factor = 2**down_ts
        self.disable_mem_mean = disable_mem_mean
        
        hidden_dim = dim * heads 
        decoder_in_dim = decoder_out_dim = pose_dim
        emb_dim = emb_dim_expansion * hidden_dim
        
        self.time_emb = TimeEmbedding(emb_dim)
        self.time_emb.apply(init_params)
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        
        if down_ts > 0:
            c_out = pose_dim            
            for i in range(down_ts+1):
                c_in = c_out
                c_out = hidden_dim
                res_block = ResnetBlock1D(c_in, hidden_dim, emb_dim)
                attn_blocks = nn.ModuleList([
                    AdaLNTransformerDecoderBlock(
                        dim, 
                        heads=heads, 
                        emb_dim=emb_dim, 
                        self_attn_dropout=self_attn_dropout,
                        cross_attn_dropout=cross_attn_dropout,
                        ff_dropout=ff_dropout,
                        attn_type=attn_type
                    )
                    for _ in range(n_attn_blocks)
                ])
                down_block = Downsample1D(hidden_dim) if i < down_ts else nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
                self.down_blocks.append(nn.ModuleList([res_block, attn_blocks, down_block]))
            for i in range(down_ts+1):
                res_block = ResnetBlock1D(2*hidden_dim, hidden_dim, emb_dim)
                attn_blocks = nn.ModuleList([
                    AdaLNTransformerDecoderBlock(
                        dim, 
                        heads=heads, 
                        emb_dim=emb_dim,
                        self_attn_dropout=self_attn_dropout,
                        cross_attn_dropout=cross_attn_dropout,
                        ff_dropout=ff_dropout,
                        attn_type=attn_type
                    )
                    for _ in range(n_attn_blocks)
                ])
                up_block = Upsample1D(hidden_dim) if i < down_ts else nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
                self.up_blocks.append(nn.ModuleList([res_block, attn_blocks, up_block]))
                
            self.out_layers = nn.Sequential(
                Block1D(hidden_dim, hidden_dim),
                nn.Conv1d(hidden_dim, pose_dim, 1)
            )
            decoder_in_dim = decoder_out_dim = hidden_dim
        else:
            self.out_layers= nn.Identity()
            
        self.down_blocks.apply(init_params)
        self.up_blocks.apply(init_params)
        self.text_emb = nn.Linear(1024, emb_dim)   
        self.mid_blocks = AdaLNTransformerDecoder(
            decoder_in_dim, 
            decoder_out_dim, 
            dim, 
            heads=heads,
            emb_dim=emb_dim, 
            num_layers=num_mid_blocks,
            self_attn_dropout=self_attn_dropout,
            cross_attn_dropout=cross_attn_dropout,
            ff_dropout=ff_dropout,
            attn_type=attn_type
        )
        self.mid_blocks.apply(init_params)
        
    def forward(
        self, 
        x: Tensor, 
        x_size: Tensor, 
        t: Tensor, 
        text_size: Tensor,
        text_feats: Tensor
    ):
        x_length = x.size(1) 
        text_feats = self.text_emb(text_feats)
        emb = self.time_emb(t)    

        if x_length % self.down_factor != 0: 
            pad = self.down_factor - x_length % self.down_factor
            x = F.pad(x, (0, 0, 0, pad), 'constant', 0)   
        padding_mask = create_padding_mask(x_size, max_length=x.size(1))
        
        mem_in = text_feats.clone()
        skips = [] 
        x = rearrange(x, 'b t d -> b d t')
        for res, attns, down in self.down_blocks:
            x = res(x, emb)
            x = rearrange(x, 'b d t -> b t d')
            for attn in attns:
                x = attn(
                    x, mem_in, emb,
                    padding_mask=padding_mask,
                    mem_padding_mask=padding_mask
                )
            x = rearrange(x, 'b t d -> b d t')
            skips.append(x)
            x = down(x)
            padding_mask = interpnd(padding_mask, x.size(2), mode='nearest')
        
        x = rearrange(x, 'b d t -> b t d')
        x = self.mid_blocks(
            x, mem_in, emb,
            padding_mask=padding_mask,
            mem_padding_mask=padding_mask
        )
        x = rearrange(x, 'b t d -> b d t')
        for res, attns, up in self.up_blocks:
            x = th.cat([x, skips.pop()], dim=1)
            x = res(x, emb)
            x = rearrange(x, 'b d t -> b t d')
            for attn in attns:
                x = attn(
                    x, mem_in, emb,
                    padding_mask=padding_mask,
                    mem_padding_mask=padding_mask
                )
            x = rearrange(x, 'b t d -> b d t')
            x = up(x)
            padding_mask = interpnd(padding_mask, x.size(2), mode='nearest')
        
        out = self.out_layers(x)
        out = rearrange(out, 'b d t -> b t d')
        out = out[:, :x_length]
        return out
