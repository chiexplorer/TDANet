###
# Author: Kai Li
# Date: 2022-05-03 18:11:15
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-08-29 16:44:07
###
from audioop import bias
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from look2hear.models.base_model import BaseModel
from look2hear.models.TransXNet import Attention1D, Mlp1D, DynamicConv1d
from look2hear.models.EMCAD_v1_6 import CAB

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
    This class defines the convolution layer with normalization
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


class GlobalAttention(nn.Module):
    def __init__(self, in_chan, out_chan, drop_path) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(out_chan, 8, 0.1, False)
        self.mlp = Mlp(out_chan, out_chan * 2, drop=0.1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class LA(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=T, mode="nearest")

        out = local_feat * sig_act + global_feat
        return out

class LAOpt2(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.cab = CAB(inp, oup, ratio=32)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=x_l.shape[-1], mode="nearest")

        out = x_l * sig_act
        out = self.cab(out) * out

        return out

class UConvBlock(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DynamicConv1d(
                in_channels,
                kernel_size=5,
                reduction_ratio=4,
                num_groups=2,
                stride=1,
                act_cfg=None,
                bias=True
            )
        )

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(
                DynamicConv1d(
                    in_channels,
                    kernel_size=2 * stride + 1,
                    reduction_ratio=4,
                    num_groups=2,
                    stride=stride,
                    act_cfg=None,
                    bias=True
                )
            )
        # 通道信息融合模块
        # self.pconvs = nn.ModuleList()
        # for _ in range(0, upsampling_depth):
        #     self.pconvs.append(nn.Conv1d(in_channels, in_channels, 1))

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.globalatt = GlobalAttention(
            in_channels * upsampling_depth, in_channels, 0.1
        )
        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(LAOpt2(in_channels, in_channels, 5))

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        time_static = {}
        # start_time = round(time.perf_counter(), 5)
        output1 = self.proj_1x1(x)
        # proj_time = round(time.perf_counter(), 5)
        output = [self.spp_dw[0](output1)]
        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)
        # downsample_time = round(time.perf_counter(), 5)
        # global features
        global_f = []
        for fea in output:
            global_f.append(F.adaptive_avg_pool1d(
                fea, output_size=output[-1].shape[-1]
            ))
        # glo_feat_time = round(time.perf_counter(), 5)
        global_f = self.globalatt(torch.stack(global_f, dim=1).sum(1))  # [B, N, T]
        # glo_attn_time = round(time.perf_counter(), 5)
        x_fused = []
        # Gather them now in reverse order
        for idx in range(self.depth):
            tmp = F.interpolate(global_f, size=output[idx].shape[-1], mode="nearest") + output[idx]
            # x_fused.append(self.pconvs[idx](tmp))  # 通道信息融合
            x_fused.append(tmp)

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)
        # la_time = round(time.perf_counter(), 5)
        # time_static["proj_time"] = (proj_time - start_time) * 1000
        # time_static["downsample_time"] = (downsample_time - proj_time) * 1000
        # time_static["glo_feat_time"] = (glo_feat_time - downsample_time) * 1000
        # time_static["glo_attn_time"] = (glo_attn_time - glo_feat_time) * 1000
        # time_static["la_time"] = (la_time - glo_attn_time) * 1000
        # print("*******UConvBlock推理耗时，depth=1*******\n", time_static)
        return self.res_conv(expanded) + residual


class Recurrent(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, _iter=4):
        super().__init__()
        self.unet = UConvBlock(out_channels, in_channels, upsampling_depth)
        self.iter = _iter
        # self.attention = Attention_block(out_channels)
        self.concat_block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )

    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.unet(x)
            else:
                x = self.unet(self.concat_block(mixture + x))
        return x


class TDANetChannelFusion(BaseModel):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        num_sources=2,
        sample_rate=16000,
        feat_len=3010
    ):
        super(TDANetChannelFusion, self).__init__(sample_rate=sample_rate)

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size * sample_rate // 1000
        self.enc_num_basis = self.enc_kernel_size // 2 + 1
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
        self.sm = Recurrent(out_channels, in_channels, upsampling_depth, num_blocks)

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
        time_stat = {}
        # input shape: (B, T)
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0)
        if input_wav.ndim == 2:
            input_wav = input_wav
        if input_wav.ndim == 3:
            input_wav = input_wav.squeeze(1)
        # start_time = round(time.perf_counter(), 5)
        x, rest = self.pad_input(
            input_wav, self.enc_kernel_size, self.enc_kernel_size // 4
        )
        # pad_time = round(time.perf_counter(), 5)
        # Front end
        x = self.encoder(x.unsqueeze(1))
        # enc_time = round(time.perf_counter(), 5)
        # Split paths
        s = x.clone()
        # Separation module
        x = self.ln(x)
        x = self.bottleneck(x)
        # bottleneck_time = round(time.perf_counter(), 5)
        x = self.sm(x)
        # sm_time = round(time.perf_counter(), 5)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)
        # mask_time = round(time.perf_counter(), 5)
        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        estimated_waveforms = estimated_waveforms[
            :,
            :,
            self.enc_kernel_size
            - self.enc_kernel_size
            // 4 : -(rest + self.enc_kernel_size - self.enc_kernel_size // 4),
        ].contiguous()
        # dec_time = round(time.perf_counter(), 5)
        # time_stat["pad_time"] = (pad_time - start_time) * 1000
        # time_stat["enc_time"] = (enc_time - pad_time) * 1000
        # time_stat["bottleneck_time"] = (bottleneck_time - enc_time) * 1000
        # time_stat["sm_time"] = (sm_time - bottleneck_time) * 1000
        # time_stat["mask_time"] = (mask_time - sm_time) * 1000
        # time_stat["dec_time"] = (dec_time - mask_time) * 1000
        # time_stat["total_time"] = (dec_time - start_time) * 1000
        # print("*******TDANet推理耗时，depth=1*******\n", time_stat)
        if was_one_d:
            return estimated_waveforms.squeeze(0)
        return estimated_waveforms

    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args


if __name__ == '__main__':
    import time
    from thop import profile
    from torchinfo import summary
    from ptflops import get_model_complexity_info

    sr = 16000
    audio_len = 32000
    model_configs = {
        "out_channels": 128,
        "in_channels": 512,
        "num_blocks": 8,
        "upsampling_depth": 5,
        "enc_kernel_size": 4,
        "num_sources": 2,
        "feat_len": 3010
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # TDANet测试
    feat_len = 3010
    model = TDANetChannelFusion(sample_rate=sr, **model_configs).to(device)
    x = torch.randn(1, audio_len, dtype=torch.float32, device=device)
    macs, params = profile(model, inputs=(x, ))
    mb = 1000*1000
    print(f"MACs: [{macs/mb/1000}] Gb \nParams: [{params/mb}] Mb")
    print("模型参数量详情：")
    summary(model, input_size=(1, audio_len), mode="train")
    start_time = time.time()
    y = model(x)
    print("batch耗时：{:.4f}".format(time.time() - start_time), y.shape)
    # # # 详细计算复杂度
    # mb = 1000 * 1000
    # shape = (1, audio_len)
    # macs, params = get_model_complexity_info(
    #     model, shape, as_strings=False, print_per_layer_stat=True, input_constructor=lambda _: {"input_wav": x}
    # )
    # print(f'Computational complexity: {macs/mb}')
    # print(f'Number of parameters: {params/mb/1000}')


    # # # UConvBlock——参数量测试
    # model = UConvBlock(out_channels=128, in_channels=512, upsampling_depth=5).cuda()
    # x = torch.rand(1, 128, 2010, dtype=torch.float32, device=device)
    # macs, params = profile(model, inputs=(x,))
    # mb = 1000*1000
    # print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")
    # print("模型参数量详情：")
    # summary(model, input_size=(1, 128, 2010), mode="train")
    # y = model(x)
    # print(y.shape)

    # # # DropPath测试
    # feat_len = 512
    # droppath = DropPath(drop_prob=0.1)
    # module = Mlp(16, 16, drop=0.1)
    # x = torch.rand(1, 16, feat_len, dtype=torch.float32, device=device)
    # y = droppath(module(x))
    # print(y.shape)


