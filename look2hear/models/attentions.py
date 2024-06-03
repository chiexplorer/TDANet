import math
from inspect import isfunction
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
from einops import rearrange, repeat, einsum
import torch.nn.functional as F

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, max_length):
        pe = torch.zeros(max_length, in_channels)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, in_channels, 2, dtype=torch.float)
                * -(math.log(10000.0) / in_channels)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, n_head, dropout=0.0, is_casual=True):
        super().__init__()
        self.pos_enc = PositionalEncoding(in_channels, 10000)
        self.attn_in_norm = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels, n_head, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_channels)
        self.is_casual = is_casual

    def forward(self, x):
        x = x.transpose(1, 2)
        attns = None
        output = self.pos_enc(self.attn_in_norm(x))
        output, _ = self.attn(output, output, output)
        output = self.norm(output + self.dropout(output))
        return output.transpose(1, 2)

class LinearAttention(nn.Module):
    """ 从LDM源码里薅的, 将常规卷积替换为深度卷积 """
    def __init__(self, in_chans, heads=4, bias=True):
        super().__init__()
        dim_head = in_chans // heads
        self.heads = heads
        self.pos_enc = PositionalEncoding(in_chans, 10000)
        self.attn_in_norm = nn.LayerNorm(in_chans)
        self.to_qkv = nn.Sequential(
            nn.Conv1d(in_chans, in_chans, 1, stride=1, padding=0, bias=bias, groups=in_chans),
            nn.Conv1d(in_chans, in_chans*3, 1, stride=1, padding=0, bias=bias)
        )  # 自定义的卷积qkv映射模块
        # self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.norm = nn.LayerNorm(in_chans)
        self.to_out = nn.Conv1d(in_chans, in_chans, 1, groups=in_chans, bias=bias)

    def forward(self, x):
        b, c, l = x.shape
        x_in = self.pos_enc(self.attn_in_norm(x))  # LN并加入位置编码
        qkv = self.to_qkv(x_in)
        q, k, v = rearrange(qkv, 'b (qkv heads c) l -> qkv b heads c l', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c l -> b (heads c) l', heads=self.heads, l=l)
        return self.to_out(out)

class CrossAttention(nn.Module):
    """ 原始的基础注意力模块，支持子注意力或互注意力 """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class MHAConv(nn.Module):
    def __init__(self, in_channels, n_head, dropout, is_casual):
        super().__init__()
        self.pos_enc = PositionalEncoding(in_channels, 10000)
        self.attn_in_norm = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels, n_head, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_channels)
        self.is_casual = is_casual

    def forward(self, x):
        x = x.transpose(1, 2)
        attns = None
        output = self.pos_enc(self.attn_in_norm(x))
        output, _ = self.attn(output, output, output)
        output = self.norm(output + self.dropout(output))
        return output.transpose(1, 2)


if __name__ == '__main__':
    from torchinfo import summary
    from thop import profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # # 卷积注意力
    # module = LinearAttention(128, 4).cuda()
    # x = torch.rand(1, 128, 2010, device=device)
    # y = module(x)
    # print(y.shape)
    # macs, params = profile(module, inputs=(x,))
    # mb = 1024 * 1024
    # print(f"MACs: [{macs / mb / 1024}] Gb \nParams: [{params / mb}] Mb")
    # print("模型参数量详情：")
    # summary(module, input_size=(1, 128, 2010), mode="train")

    # # 常规注意力
    module = MultiHeadAttention(128, 4).cuda()
    x = torch.rand(1, 128, 2010, device=device)
    y = module(x)
    print(y.shape)
    macs, params = profile(module, inputs=(x,))
    mb = 1024 * 1024
    print(f"MACs: [{macs / mb / 1024}] Gb \nParams: [{params / mb}] Mb")
    print("模型参数量详情：")
    summary(module, input_size=(1, 128, 2010), mode="train")