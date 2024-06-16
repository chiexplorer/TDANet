import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        # 公式1
        resetgate = F.sigmoid(i_r + h_r)
        # 公式2
        inputgate = F.sigmoid(i_i + h_i)
        # 公式3
        newgate = F.tanh(i_n + (resetgate * h_n))
        # 公式4，不过稍微调整了一下公式形式
        hy = newgate + inputgate * (hidden - newgate)

        return hy


class GRUConvCell(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size=3, padding=1):
        super(GRUConvCell, self).__init__()

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel

        self.gate_conv = nn.Conv1d(gru_input_channel, output_channel * 2, kernel_size=kernel_size, padding=kernel_size//2)  # 公式1&2
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-6, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-6, True)

        # filters used for outputs
        self.output_conv = nn.Conv1d(gru_input_channel, output_channel, kernel_size=kernel_size, padding=kernel_size//2)  # 公式3
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-6, True)

        self.activation = nn.Tanh()

    # 公式1，2
    def gates(self, x, h):
        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        C = f.shape[1]
        r, u = torch.split(f, C // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = torch.sigmoid(rn)
        uns = torch.sigmoid(un)
        return rns, uns

    # 公式3
    def output(self, x, h, r, u):
        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)  # 映射通道数到输出通道数
        on = self.output_norm(o)
        return on

    def forward(self, x, h=None):
        N, C, L = x.shape
        HC = self.output_channel
        if (h is None):
            h = torch.zeros((N, HC, L), dtype=torch.float, device=x.device)
        r, u = self.gates(x, h)
        o = self.output(x, h, r, u)
        y = self.activation(o)

        # 公式4
        return u * h + (1 - u) * y

class GRUDWConvCell(nn.Module):

    def __init__(self, input_channel, kernel_size=3):
        super(GRUDWConvCell, self).__init__()
        # filters used for gates
        # gru_input_channel = input_channel + output_channel
        self.input_channel = input_channel
        self.reset_conv_x = nn.Conv1d(input_channel, input_channel, kernel_size=kernel_size, padding=kernel_size // 2,
                                      groups=input_channel)
        self.reset_conv_h = nn.Conv1d(input_channel, input_channel, kernel_size=kernel_size, padding=kernel_size // 2,
                                      groups=input_channel)
        self.update_conv_x = nn.Conv1d(input_channel, input_channel, kernel_size=kernel_size, padding=kernel_size // 2,
                                       groups=input_channel)
        self.update_conv_h = nn.Conv1d(input_channel, input_channel, kernel_size=kernel_size, padding=kernel_size // 2,
                                       groups=input_channel)
        self.output_conv_x = nn.Conv1d(input_channel, input_channel, kernel_size=kernel_size, padding=kernel_size // 2,
                                       groups=input_channel)
        self.output_conv_h = nn.Conv1d(input_channel, input_channel, kernel_size=kernel_size, padding=kernel_size // 2,
                                       groups=input_channel)

        # self.gate_conv = nn.Conv1d(gru_input_channel, output_channel * 2, kernel_size=kernel_size, padding=kernel_size//2)  # 公式1&2
        self.reset_gate_norm = nn.GroupNorm(1, input_channel, 1e-6, True)
        self.update_gate_norm = nn.GroupNorm(1, input_channel, 1e-6, True)

        # filters used for outputs
        # self.output_conv = nn.Conv1d(gru_input_channel, output_channel, kernel_size=kernel_size, padding=kernel_size//2)  # 公式3
        self.output_norm = nn.GroupNorm(1, input_channel, 1e-6, True)

        self.activation = nn.Tanh()

    # 公式1，2
    def gates(self, x, h):
        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        C = f.shape[1]
        r, u = torch.split(f, C // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = torch.sigmoid(rn)
        uns = torch.sigmoid(un)
        return rns, uns

    # 公示1
    def reset_gate(self, x, h):
        r = self.reset_conv_x(x) + self.reset_conv_h(h)
        rn = self.reset_gate_norm(r)
        rns = torch.sigmoid(rn)
        return rns

    # 公示2
    def update_gate(self, x, h):
        u = self.update_conv_x(x) + self.update_conv_h(h)
        un = self.update_gate_norm(u)
        uns = torch.sigmoid(un)
        return uns

    # 公式3
    def output(self, x, h, r, u):
        o = self.output_conv_x(x) + self.output_conv_h(r * h)
        on = self.output_norm(o)
        return on

    def forward(self, x, h=None):
        N, C, L = x.shape
        HC = self.input_channel
        if (h is None):
            h = torch.zeros((N, HC, L), dtype=torch.float, device=x.device)
        r = self.reset_gate(x, h)
        u = self.update_gate(x, h)
        o = self.output(x, h, r, u)
        y = self.activation(o)

        # 公式4
        return u * h + (1 - u) * y

class GRUNet(nn.Module):

    def __init__(self, in_channel=4, out_channle=None, hidden_size=64):
        super(GRUNet, self).__init__()
        out_channle = in_channel if out_channle is None else out_channle
        # self.gru_1 = GRUConvCell(input_channel=in_channel, output_channel=hidden_size)
        # self.gru_2 = GRUConvCell(input_channel=hidden_size, output_channel=hidden_size)
        # self.gru_3 = GRUConvCell(input_channel=hidden_size, output_channel=hidden_size)
        self.gru_1 = GRUDWConvCell(input_channel=in_channel)
        self.gru_2 = GRUDWConvCell(input_channel=in_channel)
        self.gru_3 = GRUDWConvCell(input_channel=in_channel)

        self.fc = nn.Conv1d(in_channels=hidden_size, out_channels=out_channle, kernel_size=3, padding=1)

    def forward(self, x, h=None):
        if h is None:
            h = [None, None, None]

        h1 = self.gru_1(x, h[0])
        h2 = self.gru_2(h1, h[1])
        h3 = self.gru_3(h2, h[2])

        o = self.fc(h3)

        return o, [h1, h2, h3]


if __name__ == '__main__':
    device = 'cuda'
    from torchinfo import summary
    from thop import profile


    x = torch.rand(1, 128, 3010).to(device)

    grunet = GRUNet(in_channel=128, hidden_size=128)
    grunet = grunet.to(device)
    # grunet.eval()
    macs, params = profile(grunet, inputs=(x, None))
    mb = 1000*1000
    print(f"MACs: [{macs/mb/1000}] G \nParams: [{params/mb}] M")
    print("模型参数量详情：")
    summary(grunet, input_size=((1, 128, 3010)), mode="train")
    h = None
    o, h_n = grunet(x, h)
    print("output:", o.shape)
    for i, h in enumerate(h_n):
        print(f"h_n [{i}]:", h.shape)

