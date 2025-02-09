from audioop import bias
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import ReformerModel, ReformerConfig, AutoTokenizer, ReformerLayer, ReformerAttention

from thop import profile
from torchinfo import summary
from ptflops import get_model_complexity_info

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def GlobLN(nOut):
    return nn.GroupNorm(1, nOut, eps=1e-8)

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

#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveAvgPool1d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv1d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv1d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)

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

class LAOpt1(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)

        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, T = x_l.shape

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")

        out = x_l * sig_act + x_l
        return out

class LAOpt2(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.cab = CAB(inp, oup, ratio=16)
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

class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.GroupNorm(1, F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.GroupNorm(1, F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

    def forward(self, g, x):
        x_up = F.interpolate(x, size=g.size(-1), mode='nearest')
        g1 = self.W_g(g)
        x1 = self.W_x(x_up)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x_up * psi


if __name__ == '__main__':
    x_l = torch.rand(1, 512, 2010)
    x_g = torch.rand(1, 512, 1005)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # la = LA(512, 512)
    # start_time = time.time()
    # out = la(x_l, x_g)
    # print("处理耗时：{:.4f}".format(time.time() - start_time))
    #
    # # 粗统计模型参数量和计算量
    # macs, params = profile(la, inputs=(x_l, x_g))
    # mb = 1000 * 1000
    # print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")
    # print("模型参数量详情：")
    # summary(la, input_size=[(1, 512, 2010), (1, 512, 1005)], mode="train")

    # # 详细统计模型参数量和计算量
    # shape = (1, 512, 2010)
    # macs, params = get_model_complexity_info(
    #     la, shape, as_strings=False, print_per_layer_stat=True, input_constructor=lambda _: {"x_l": x_l, "x_g": x_g}
    # )
    # print(f'Computational complexity: {macs/mb}')
    # print(f'Number of parameters: {params/mb/1000}')

    config = ReformerConfig(
        attention_head_size=64,  # 头维度
        attn_layers=["lsh"],  # 注意力层
        num_attention_heads=8,  # 头数
        hidden_size=512,  # 隐藏层维度
        num_hidden_layers=1,  # 层数
        feed_forward_size=2048,  # FFN 层维度
        max_position_embeddings=4096,  # 最大序列长度
    )
    # model = ReformerModel(config)
    # print(model)
    #
    # tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # # 创建假输入 (batch_size=1, sequence_length=1024)
    # input_ids = torch.randint(0, 1000, (1, 1024))  # 词 ID
    # y = model(**inputs)
    # print(y)

    # # ReformerLayer demo
    # t0 = torch.randn((1, 11, 256)).to(device)
    # model = ReformerLayer(config).to(device)
    # h = torch.randn((1, 11, 256)).to(device)
    # y = model(t0, h)
    # print(y.hidden_states.shape, y.attn_output)

    # ReformerAttention demo
    # model = ReformerAttention(config).to(device)
    # h = torch.randn((1, 513, 512)).to(device)
    # y = model(h)
    # print(y.hidden_states.shape)
    # macs, params = profile(model, inputs=(h))
    # mb = 1000 * 1000
    # print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")

    # # # LAOpt1 demo
    # la = LAOpt1(512, 512)
    # start_time = time.time()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # out = la(x_l, x_g)
    # print("处理耗时：{:.4f}".format(time.time() - start_time))
    # # 粗统计模型参数量和计算量
    # macs, params = profile(la, inputs=(x_l, x_g))
    # mb = 1000 * 1000
    # print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")

    # # LAOpt2 demo
    la = LAOpt2(512, 512).to(device)
    start_time = time.time()
    out = la(x_l, x_g)
    print("处理耗时：{:.4f}".format(time.time() - start_time))
    # 粗统计模型参数量和计算量
    macs, params = profile(la, inputs=(x_l, x_g))
    mb = 1000 * 1000
    print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")
    print("模型参数量详情：")
    summary(la, input_size=[(1, 512, 2010), (1, 512, 1005)], mode="train")

    # # LGAG复杂度测试
    # module = LGAG(F_g=512, F_l=512, F_int=256, kernel_size=3,
    #                       groups=256, activation='relu').to(device)
    # start_time = time.time()
    # out = module(x_g, x_l)
    # print("处理耗时：{:.4f}".format(time.time() - start_time))
    # macs, params = profile(module, inputs=(x_l, x_g))
    # mb = 1000 * 1000
    # print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")
    # print("模型参数量详情：")
    # summary(module, input_data=(x_l, x_g), mode="train")