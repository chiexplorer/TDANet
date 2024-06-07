###
# Author: Kai Li
# Date: 2022-05-03 18:11:15
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-08-29 16:44:07
###
from audioop import bias

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from look2hear.models.base_model import BaseModel
from look2hear.models.SeBlock import SEBasicBlock1D

from look2hear.models.attentions import LinearAttention

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob

    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


def GlobLN(nOut):
    return nn.GroupNorm(1, nOut, eps=1e-8)


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    """
    This class defines a normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: number of output channels
        """
        super().__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)

class DilatedConv(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)

class DilatedSeparableConvNorm(nn.Module):
    """
    This class defines the separabal dilated convolution with normalized output.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.dw_conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        self.pw_conv = nn.Conv1d(
            nIn,
            nOut,
            1,
            stride=1,
            dilation=1,
            padding=0,
            groups=1,
        )
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.dw_conv(input)
        output = self.pw_conv(output)
        return self.norm(output)

class SAM1D(nn.Module):
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.PReLU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups = self.dim // ca_num_heads

        if ca_attention == 1:
            # SAM
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.s = nn.Linear(dim, dim, bias=qkv_bias)
            for i in range(self.ca_num_heads):
                local_conv = nn.Conv1d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                                       padding=(1 + i), stride=1, groups=dim // self.ca_num_heads)
                setattr(self, f"local_conv_{i + 1}", local_conv)
            # SAA
            self.proj0 = nn.Conv1d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                                   groups=self.split_groups)
            # self.bn = nn.BatchNorm1d(dim * expand_ratio)  # 原归一化方式
            self.norm = GlobLN(dim*expand_ratio)  # 向TDANet统一
            self.proj1 = nn.Conv1d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        else:
            head_dim = dim // sa_num_heads
            self.scale = qk_scale or head_dim ** -0.5
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, N, C = x.shape
        if self.ca_attention == 1:
            # MHMC
            v = self.v(x)
            s = self.s(x).reshape(B, N, self.ca_num_heads, C // self.ca_num_heads).permute(2, 0, 3, 1)
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                s_i = s[i]
                s_i = local_conv(s_i).reshape(B, self.split_groups, -1, N)
                if i == 0:
                    s_out = s_i
                else:
                    s_out = torch.cat([s_out, s_i], 2)
            # SAA
            s_out = s_out.reshape(B, C, N)
            s_out = self.proj1(self.act(self.norm(self.proj0(s_out))))
            self.modulator = s_out
            s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
            x = s_out * v

        else:
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
                self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)).view(B, C,
                                                                                                          N).transpose(
                    1, 2)

        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1):
        super().__init__()
        self.fc1 = ConvNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv1d(
            hidden_size, hidden_size, 5, 1, 2, bias=True, groups=hidden_size
        )
        self.act = nn.ReLU()
        self.fc2 = ConvNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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
    """ 如何理解将通道和特征维度调序的操作（通道数固定方便实现，且参数量小一些[maybe]）？是否需要两重dropout """
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
        output, _ = self.attn(output, output, output)  # 原代码
        output = self.norm(output + self.dropout(output))  # 原代码 错误的残差连接方式
        # output = self.norm(self.dropout(output))  # 修正(1) 去掉output的自加
        # # 修改代码(2)
        # attn_output, _ = self.attn(output, output, output)
        # attn_output = self.norm(output + attn_output)  # 残差连接

        return output.transpose(1, 2)

class GlobalAttention(nn.Module):
    def __init__(self, in_chan, out_chan, drop_path) -> None:
        super().__init__()
        # dropout+droppath？这表示GA中应用了两重droppout
        self.attn = MultiHeadAttention(out_chan, 8, 0.1, False)
        self.mlp = Mlp(out_chan, out_chan * 2, drop=0.1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))  # MHSA消融实验
        x = x + self.drop_path(self.mlp(x))
        return x


class LA(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)  # 样本再编码
        self.global_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)  # 偏移因子
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)  # 缩放因子
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_l: local features
        x_g: global features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=T, mode="nearest")

        out = local_feat * sig_act + global_feat
        return out

class SAMLA(nn.Module):
    def __init__(self, dim: int, inp: int, oup: int, kernel: int = 1, ca_num_heads: int=4) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)  # 样本再编码
        self.global_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)  # 偏移因子
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)  # 缩放因子
        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        self.ca_num_heads = ca_num_heads
        self.split_groups = dim // ca_num_heads
        for i in range(self.ca_num_heads):
                local_conv = nn.Conv1d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                                       padding=(1 + i), stride=1, groups=dim // self.ca_num_heads)
                setattr(self, f"local_conv_{i + 1}", local_conv)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_l: local features
        x_g: global features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)
        local_feat = local_feat.reshape(B, self.ca_num_heads, N // self.ca_num_heads, T).permute(1, 0, 2, 3)
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = local_feat[i]
            s_i = local_conv(s_i).reshape(B, self.split_groups, -1, T)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 2)
        local_feat = local_feat.reshape(B, N, T)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=T, mode="nearest")

        out = local_feat * sig_act + global_feat
        return out

class AdaLN(nn.Module):
    """
    This class defines the Simplified Adaptive Layer Normalization module.
    @param feat_l: length of input feature
    @param feat_g: length of cond feature
    @param c_out: the number of output channels
    """
    def __init__(self, feat_l, feat_g, c_out):
        super(AdaLN, self).__init__()

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(feat_g, 2 * feat_l, bias=False),
            GlobLN(c_out)
        )
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        gamma, beta = self.adaLN_modulation(x_g).chunk(2, dim=-1)
        gamma = self.act(gamma)
        # gamma = gamma.unsqueeze(-1)
        # beta = beta.unsqueeze(-1)
        return x_l * gamma + beta

class UConvBlock(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, feat_len=None):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(
                in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1
            )
        )
        # 卷积注意力改版
        # self.attn_down = nn.ModuleList()
        # self.attn_down.append(LinearAttention(in_channels, 4))  # 考虑是否要加到第一层中
        # self.attn_up = nn.ModuleList()
        # self.attn_up.append(LinearAttention(in_channels, 4))

        # # 卷积替换avg pooling改版
        self.conv_pool = nn.ModuleList()
        self.conv_pool.append(
            DilatedSeparableConvNorm(
                in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1
            )
        )


        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
                conv_stride = 2 ** i
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=2 * stride + 1,
                    stride=stride,
                    groups=in_channels,
                    d=1,
                )
            )
            # # 卷积注意力改版
            # self.attn_down.append(
            #     LinearAttention(in_channels, 4)
            # )
            # self.attn_up.append(
            #     LinearAttention(in_channels, 4)
            # )
            # 卷积替换avg pooling改版
            self.conv_pool.append(
                DilatedSeparableConvNorm(
                    in_channels,
                    in_channels,
                    kSize=2 * conv_stride + 1,
                    stride=conv_stride,
                    groups=in_channels,
                    d=1,
                )
            )

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        # self.sam_block = SAM1D(dim=in_channels, ca_num_heads=4, ca_attention=1, proj_drop=0.0, sa_num_heads=8, expand_ratio=2)
        self.globalatt = GlobalAttention(
            in_channels * upsampling_depth, in_channels, 0.1
        )
        self.last_layer = nn.ModuleList([])
        # self.sam_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(LA(in_channels, in_channels, 5))
            # # SAM增强LA改版
            # self.last_layer.append(SAMLA(in_channels,in_channels, in_channels, 5))
            # # LA后接SAM改版
            # self.sam_layer.append(SAM1D(dim=in_channels, ca_num_heads=4, ca_attention=1, proj_drop=0.0, sa_num_heads=8, expand_ratio=2))


    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()  # (B, Cout, T)
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)  # (B, Cin, T)
        # UNet最底层特征是未经过下采样的，由conv1d得到
        output = [self.spp_dw[0](output1)]  # (B, Cin, T)
        # output = [self.attn_down[0](self.spp_dw[0](output1))]  # (B, Cin, T)  # 考虑是否加到第一层

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            # out_k = self.attn_down[k-1](self.spp_dw[k](output[-1]))  # 若加到第一层，使用k为idx
            out_k = self.spp_dw[k](output[-1])  # 原实现
            output.append(out_k)

        # 卷积替换avg pooling改版
        conv_output = []
        for k, fea in enumerate(output):
            conv_out_k = self.conv_pool[self.depth - k - 1](fea)
            conv_output.append(conv_out_k)

        # global features
        global_f = []
        for fea in conv_output:
            # global_f.append(F.adaptive_avg_pool1d(
            #     fea, output_size=output[-1].shape[-1]
            # ))  # [B, Cin, T/2^S]
            # 卷积替换avg pooling改版
            global_f.append(fea)  # [B, Cin, T/2^S]
        # global_f = self.sam_block(torch.stack(global_f, dim=1).sum(1))  # SAM替换MHSA
        # global_f = self.globalatt(global_f)
        global_f = self.globalatt(torch.stack(global_f, dim=1).sum(1))  # [B, Cin, T/2^S]  # 原代码

        x_fused = []  # 融合gm后的不同尺度特征
        # Gather them now in reverse order
        for idx in range(self.depth):
            tmp = F.interpolate(global_f, size=output[idx].shape[-1], mode="nearest") + output[idx]
            x_fused.append(tmp)

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)
            # expanded = self.attn_up[i](expanded)  # 卷积注意力改版
            # expanded = self.sam_layer[i](expanded)  # SAM增强LA改版
        return self.res_conv(expanded) + residual

class UConvBlockV1(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, feat_len=None):
        super().__init__()
        assert feat_len is not None, "Fool! You have forggoten to provide the feature length!"
        self.feat_len = feat_len
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(
                in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1
            )
        )

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=2 * stride + 1,
                    stride=stride,
                    groups=in_channels,
                    d=1,
                )
            )

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.globalatt = GlobalAttention(
            in_channels * upsampling_depth, in_channels, 0.1
        )
        # self.globalatt_mult = nn.ModuleList(
        #     [GlobalAttention(in_channels, in_channels, 0.1) for _ in range(self.depth)]
        # )  # 全局特征提取改版
        self.last_layer = nn.ModuleList([])
        # 原始LA Layer
        for i in range(self.depth - 1):
            self.last_layer.append(LA(in_channels, in_channels, 5))

        """
        # adaLN 替换 LA Layer(LA改版)
        feat_len_tmp = feat_len
        feat_lens = [feat_len]
        for i in range(self.depth - 2):
            feat_len_tmp = (feat_len_tmp + 1) // 2
            feat_lens.append(feat_len_tmp)
        for i in range(self.depth - 1):
            if i == self.depth - 2:
                adaLN_modulation = AdaLN(feat_lens[i], feat_lens[i - 1], in_channels)
            else:
                adaLN_modulation = AdaLN(feat_lens[i], feat_lens[i + 1], in_channels)
            self.last_layer.append(adaLN_modulation)
        """
        self.se_block = nn.ModuleList([])
        # SE Block特征改版
        for i in range(self.depth):
            self.se_block.append(SEBasicBlock1D(in_channels, in_channels))


    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()  # (B, Cout, T)
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)  # (B, Cin, T)
        # UNet最上层特征是未经过下采样的，由conv1d得到
        output = [self.spp_dw[0](output1)]  # (B, Cin, T)

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # global features
        global_f = []
        for i, fea in enumerate(output):
            """
            # 全局特征提取改版
            tmp_feat = F.adaptive_avg_pool1d(
                fea, output_size=output[-1].shape[-1]
            )
            global_f.append(self.globalatt_mult[i](tmp_feat))  # [B, Cin, T/2^S]
            """
            fea = self.se_block[i](fea)  # SEBlock增强局部特征
            global_f.append(F.adaptive_avg_pool1d(
                fea, output_size=output[-1].shape[-1]
            ))

        global_f = self.globalatt(torch.stack(global_f, dim=1).sum(1))  # [B, Cin, T/2^S]
        # global_f = torch.stack(global_f, dim=1).sum(1) / float(len(output))  # 全局特征提取改版, 除以长度保持尺度一致

        x_fused = []  # 融合gm后的不同尺度特征
        # Gather them now in reverse order
        for idx in range(self.depth):
            tmp = F.interpolate(global_f, size=output[idx].shape[-1], mode="nearest") + output[idx]
            x_fused.append(tmp)

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)
        return self.res_conv(expanded) + residual

class Recurrent(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, _iter=4, feat_len=None):
        super().__init__()
        self.unet = UConvBlock(out_channels, in_channels, upsampling_depth, feat_len)  # (LA改版)
        self.iter = _iter
        # self.attention = Attention_block(out_channels)
        self.concat_block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )
        # # concat作为scale因子
        # self.concat_block_scale = nn.Sequential(
        #     nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels), nn.Sigmoid()
        # )

    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            # # concat_block消融实验
            # if i == 0:
            #     x = self.unet(x)
            # else:
            #     x = self.unet(mixture + x)
            # # 原代码
            if i == 0:
                x = self.unet(x)
            else:
                x = self.unet(self.concat_block(mixture + x))
            # # 改为掩模形式，更倾向于RNN
            # x = self.concat_block_scale(self.unet(x)) * x

        return x


class TDANet(BaseModel):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        num_sources=2,
        sample_rate=16000,
        feat_len=None
    ):
        super(TDANet, self).__init__(sample_rate=sample_rate)

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size * sample_rate // 1000
        self.enc_num_basis = self.enc_kernel_size // 2 + 1  # 编码器输出通道数
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(
            self.enc_kernel_size // 4 * 4 ** self.upsampling_depth
        ) // math.gcd(self.enc_kernel_size // 4, 4 ** self.upsampling_depth)

        # Front end
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=self.enc_num_basis,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_kernel_size // 4,
            padding=self.enc_kernel_size // 2,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(self.enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=self.enc_num_basis, out_channels=out_channels, kernel_size=1
        )

        # Separation module
        self.sm = Recurrent(out_channels, in_channels, upsampling_depth, num_blocks, feat_len=feat_len)  # (LA改版)

        mask_conv = nn.Conv1d(out_channels, num_sources * self.enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=self.enc_num_basis * num_sources,
            out_channels=num_sources,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_kernel_size // 4,
            padding=self.enc_kernel_size // 2,
            groups=1,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, window - stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    # Forward pass
    def forward(self, input_wav):
        # input shape: (B, T)
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0)
        if input_wav.ndim == 2:
            input_wav = input_wav
        if input_wav.ndim == 3:
            input_wav = input_wav.squeeze(1)

        x, rest = self.pad_input(
            input_wav, self.enc_kernel_size, self.enc_kernel_size // 4
        )
        # Front end
        x = self.encoder(x.unsqueeze(1))  # [B, enc_num_basis, L]

        # Split paths
        s = x.clone()
        # Separation module
        x = self.ln(x)
        x = self.bottleneck(x)  # [B, out_channels, L]
        x = self.sm(x)  # [B, out_channels, L]

        x = self.mask_net(x)  # [B, 2*enc_num_basis, L]
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)  # [B, 2, enc_num_basis, L]
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)  # [B, 2, enc_num_basis, L]
        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))  # [B, 2, T]
        # 截断至对齐原音频的长度
        estimated_waveforms = estimated_waveforms[
            :,
            :,
            self.enc_kernel_size
            - self.enc_kernel_size
            // 4 : -(rest + self.enc_kernel_size - self.enc_kernel_size // 4),
        ].contiguous()
        if was_one_d:
            return estimated_waveforms.squeeze(0)
        return estimated_waveforms

    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args


if __name__ == '__main__':
    from thop import profile
    from torchinfo import summary
    sr = 8000
    model_configs = {
        "out_channels": 128,
        "in_channels": 512,
        "num_blocks": 16,
        "upsampling_depth": 5,
        "enc_kernel_size": 4,
        "num_sources": 2,
        "feat_len": 3010
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TDANet测试
    feat_len = 3010
    model = TDANet(sample_rate=sr, **model_configs).cuda()
    x = torch.randn(1, 24000, dtype=torch.float32, device=device)
    macs, params = profile(model, inputs=(x, ))
    mb = 1024*1024
    print(f"MACs: [{macs/mb/1024}] Gb \nParams: [{params/mb}] Mb")
    print("模型参数量详情：")
    summary(model, input_size=(1, 24000), mode="train")
    y = model(x)
    print(y.shape)

    # # DialateConvNorm——任意shape输入测试
    # in_channels = 512
    # mudule = DilatedConvNorm(
    #     in_channels, in_channels, kSize=5, stride=9, groups=in_channels, d=1
    # )
    # x = torch.rand(2, 512, 2016)
    # y = mudule(x)
    # print(y.shape)

    # # # UConvBlock——参数量测试
    # model = UConvBlock(out_channels=128, in_channels=512, upsampling_depth=5).cuda()
    # x = torch.rand(1, 128, 2010, dtype=torch.float32, device=device)
    # macs, params = profile(model, inputs=(x,))
    # mb = 1024 * 1024
    # print(f"MACs: [{macs / mb / 1024}] Gb \nParams: [{params / mb}] Mb")
    # print("模型参数量详情：")
    # summary(model, input_size=(1, 128, 2010), mode="train")
    # y = model(x)
    # print(y.shape)

    # # AdaLN测试
    # model = AdaLN(512, 256, 128).cuda()
    # x_l = torch.rand(2, 128, 512, device=device)
    # x_g = torch.rand(2, 128, 256, device=device)
    # y = model(x_l, x_g)
    # print("模型参数量详情：")
    # summary(model, input_size=((2, 128, 512), (2, 128, 256)), mode="train")
    # print(y.shape)

    # # UConvBlockV1——正确性测试
    # feat_len = 3010
    # model = UConvBlockV1(out_channels=128, in_channels=512, upsampling_depth=5, feat_len=feat_len).to(device)
    # x = torch.rand(1, 128, feat_len, dtype=torch.float32, device=device)
    # # macs, params = profile(model, inputs=(x,))
    # # mb = 1024 * 1024
    # # print(f"MACs: [{macs / mb / 1024}] Gb \nParams: [{params / mb}] Mb")
    # # print("模型参数量详情：")
    # y = model(x)
    # print(y.shape)
    # summary(model, input_size=(1, 128, feat_len), mode="train")

    # # GlobalAttention——测试
    # feat_len = 377
    # model = GlobalAttention(in_chan=128, out_chan=128, drop_path=0.1).to(device)
    # x = torch.rand(1, 128, feat_len, dtype=torch.float32, device=device)
    # y = model(x)
    # print(y.shape)
    # summary(model, input_size=(1, 128, feat_len), mode="train")