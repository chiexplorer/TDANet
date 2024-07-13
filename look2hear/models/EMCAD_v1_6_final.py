import torch
import torch.nn as nn
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


def get_feat_lens(feat_len, depth):
    feat_len_tmp = feat_len
    feat_lens = [feat_len]
    for i in range(depth - 1):
        feat_len_tmp = (feat_len_tmp + 1) // 2
        feat_lens.append(feat_len_tmp)
    feat_lens.reverse()  # 翻转
    return feat_lens


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GroupNorm) or isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


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


def channel_shuffle(x, groups):
    batchsize, num_channels, length = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, length)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, length)
    return x


#   Multi-scale depth-wise convolution (MSDC)
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.GroupNorm(1, self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        # self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        return outputs


class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB)
    """

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        super(MSCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        # 原版：nn.Conv1d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
        # 轻量化v1: nn.Conv1d(self.in_channels, self.ex_channels, 3, 1, 1, groups=self.in_channels, bias=False),
        # 轻量化v2: nn.Conv1d(self.in_channels, self.ex_channels, 1, 1, 0, groups=self.in_channels//4, bias=False),
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv1d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.GroupNorm(1, self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        # 原版：nn.Conv1d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
        # 轻量化v1: nn.Conv1d(self.combined_channels, self.out_channels, 3, 1, 1, groups=self.combined_channels, bias=False),
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv1d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.GroupNorm(1, self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv1d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        # self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)  # 轻量化v1
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)  # 轻量化v1
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


#   Multi-scale convolution block (MSCB)
def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
              add=True, activation='relu6'):
    """
    create a series of multi-scale convolution blocks.
    """
    convs = []
    mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation)
    convs.append(mscb)
    if n > 1:
        for i in range(1, n):
            mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation)
            convs.append(mscb)
    conv = nn.Sequential(*convs)
    return conv


#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, scale_len, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(size=scale_len),
            nn.Conv1d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.GroupNorm(1, self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )  # 轻量化v1
        # self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)  # 轻量化v1
        return x

class EUCBLight(nn.Module):
    def __init__(self, scale_len, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCBLight, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(size=scale_len),
            nn.Conv1d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                              padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.GroupNorm(1, self.in_channels),
            act_layer(activation, inplace=True)
        )

        # self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        # x = channel_shuffle(x, self.in_channels)
        # x = self.pwc(x)  # 轻量化v1
        return x



#   Large-kernel grouped attention gate (LGAG)
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

        # self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


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

        # self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)

    #   Spatial attention block (SAB)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        # self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


#   Efficient multi-scale convolutional attention decoding (EMCAD)
class EMCADv1_6_Final(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
                 add=True, lgag_ks=3, activation='relu', feat_len=None):
        super(EMCADv1_6_Final, self).__init__()
        eucb_ks = 3  # kernel size for eucb
        self.feat_len = feat_len
        assert feat_len is not None, "Fool! You must provide the feature length"
        self.stage_len_list = get_feat_lens(feat_len, 4)  # stage对应的特征长度
        self.lgag4 = LGAG(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2, kernel_size=lgag_ks,
                          groups=channels[0] // 2, activation=activation)
        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.eucb3 = EUCBLight(self.stage_len_list[0], in_channels=channels[0], out_channels=channels[1],
                          kernel_size=eucb_ks, stride=eucb_ks // 2, activation=activation)
        self.lgag3 = LGAG(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2, kernel_size=lgag_ks,
                          groups=channels[1] // 2, activation=activation)
        # self.mscb3 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes,
        #                        expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
        #                        activation=activation)

        self.eucb2 = EUCB(self.stage_len_list[1], in_channels=channels[1], out_channels=channels[2],
                          kernel_size=eucb_ks, stride=eucb_ks // 2, activation=activation)
        self.lgag2 = LGAG(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, kernel_size=lgag_ks,
                          groups=channels[2] // 2, activation=activation)
        # self.mscb2 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes,
        #                        expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
        #                        activation=activation)

        self.eucb1 = EUCBLight(self.stage_len_list[2], in_channels=channels[2], out_channels=channels[3],
                          kernel_size=eucb_ks, stride=eucb_ks // 2, activation=activation)
        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2), kernel_size=lgag_ks,
                          groups=int(channels[3] / 2), activation=activation)
        # self.mscb1 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes,
        #                        expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
        #                        activation=activation)

        self.eucb0 = EUCB(self.stage_len_list[3], in_channels=channels[3], out_channels=channels[4],
                          kernel_size=eucb_ks, stride=eucb_ks // 2, activation=activation)
        self.lgag0 = LGAG(F_g=channels[4], F_l=channels[4], F_int=int(channels[4] / 2), kernel_size=lgag_ks,
                          groups=int(channels[4] / 2), activation=activation)
        self.mscb0 = MSCBLayer(channels[4], channels[4], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

    def forward(self, x, skips):
        # MSCAM4
        d4 = skips[-1] + self.lgag4(g=x, x=skips[-1])
        d4 = skips[-1] + d4
        d4 = self.mscb4(d4)

        # EUCB3
        d3 = self.eucb3(d4)

        # LGAG3
        x3 = self.lgag3(g=d3, x=skips[-2])

        # Additive aggregation 3
        d3 = d3 + x3

        # MSCAM3
        # d3 = self.mscb3(d3)

        # EUCB2
        d2 = self.eucb2(d3)

        # LGAG2
        x2 = self.lgag2(g=d2, x=skips[-3])

        # Additive aggregation 2
        d2 = d2 + x2

        # MSCAM2
        # d2 = self.mscb2(d2)

        # EUCB1
        d1 = self.eucb1(d2)

        # LGAG1
        x1 = self.lgag1(g=d1, x=skips[-4])

        # Additive aggregation 1
        d1 = d1 + x1

        # MSCAM1
        # d1 = self.mscb1(d1)

        # EUCB0
        d0 = self.eucb0(d1)
        # LGAG0
        x0 = self.lgag0(g=d0, x=skips[-5])
        # Additive aggregation 0
        d0 = d0 + x0
        # MSCAM0
        d0 = self.mscb0(d0)

        return [d4, d3, d2, d1, d0]


if __name__ == '__main__':
    from torchinfo import summary
    from thop import profile
    from ptflops import get_model_complexity_info


    def get_skips(shape, channs, n_layers=4, scale_0=4, scale_n=2, device="cpu"):
        skips = []
        if len(shape) == 3:
            B, C, L = shape

            l = L
            c = channs[0]
            skips.append(torch.rand(B, c, L, device=device))

            for i in range(n_layers - 1):
                l = (l + 1) // scale_n

                skips.append(torch.rand(B, channs[i], l, device=device))
            # skips.reverse()
        elif len(shape) == 4:
            B, C, H, W = shape

            h = H * scale_n
            w = W * scale_n
            c = channs[1]
            skips.append(torch.rand(B, c, h, w, device=device))

            for i in range(1, n_layers - 1):
                h = h * scale_n
                w = w * scale_n

                skips.append(torch.rand(B, channs[i + 1], h, w, device=device))
        return skips

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # EMCAD4TDANet测试
    x_len = 3010
    feat_len = 189

    channs = [512]*5
    model = EMCADv1_6_Final(channels=channs, feat_len=x_len, expansion_factor=1).cuda()
    shape = (1, 512, feat_len)
    l0_shape = (1, 512, x_len)
    x = torch.rand(*shape, device=device)
    skips = get_skips(l0_shape, channs, n_layers=5, device=device)
    mb = 1000 * 1000
    y = model(skips[-1], skips)
    print(y[0].shape, y[1].shape, y[2].shape, y[3].shape, y[4].shape)
    # 计算复杂度
    macs, params = profile(model, inputs=(x, skips))
    print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")
    print("模型参数量详情：")
    summary(model, input_data=(x, skips), mode="train")
    # # 详细计算复杂度
    # input_res = ((1, 512, feat_len), (1, 512, feat_len))
    # macs, params = get_model_complexity_info(
    #     model, shape, as_strings=False, print_per_layer_stat=True, input_constructor=lambda _: {"x": x, "skips": skips}
    # )
    # print(f'Computational complexity: {macs/mb}')
    # print(f'Number of parameters: {params/mb/1000}')


    # # EUCB测试
    # module = EUCB(753, 16, 16, kernel_size=3, stride=1, activation='relu').cuda()
    # x = torch.rand(4, 16, 377, device=device)
    # y = module(x)
    # print(y.shape)

    # # # EMCADF1测试
    # feat_len = 3010
    # channs = [512]*5
    # model = EMCADF1(channels=channs, feat_len=feat_len, expansion_factor=0.5).cuda()
    # shape = (1, 512, feat_len)
    #
    # x = torch.rand(*shape, device=device)
    # skips = get_skips(x.shape, channs, n_layers=5, device=device)
    # y = model(skips[-1], skips)
    # print(y.shape)
    # # macs, params = profile(model, inputs=(x, skips))
    # # mb = 1000 * 1000
    # # print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")
    # print("模型参数量详情：")
    # summary(model, input_data=(x, skips), mode="train")

    # CAB复杂度测试
    # module = CAB(512).cuda()
    # x = torch.rand(1, 512, 3010, device=device)
    # macs, params = profile(module, inputs=(x))
    # mb = 1000 * 1000
    # print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")
    # print("模型参数量详情：")
    # summary(module, input_data=(x), mode="train")

    # # SAB复杂度测试
    # module = SAB().cuda()
    # x = torch.rand(1, 512, 3010, device=device)
    # macs, params = profile(module, inputs=(x, ))
    # mb = 1000 * 1000
    # print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")
    # print("模型参数量详情：")
    # summary(module, input_data=(x, ), mode="train")

    # # MSCB复杂度测试
    # module = MSCBLayer(512, 512, n=1, stride=1, kernel_sizes=[1, 3, 5],
    #                            expansion_factor=0.5, dw_parallel=True, add=True,
    #                            activation='relu').cuda()
    # x = torch.rand(1, 512, 189, device=device)
    # macs, params = profile(module, inputs=(x, ))
    # mb = 1000 * 1000
    # print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")
    # print("模型参数量详情：")
    # summary(module, input_data=(x, ), mode="train")

    # # LGAG复杂度测试
    # module = LGAG(F_g=512, F_l=512, F_int=256, kernel_size=3,
    #                       groups=256, activation='relu').cuda()
    # x = torch.rand(1, 512, 1505, device=device)
    # skip = torch.rand_like(x, device=device)
    # macs, params = profile(module, inputs=(x, skip))
    # mb = 1000 * 1000
    # print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")
    # print("模型参数量详情：")
    # summary(module, input_data=(x, skip), mode="train")

    # # EUCB测试
    # feat_len = 3010
    # module = EUCB(feat_len, 512, 512, kernel_size=3, stride=1, activation='relu').cuda()
    # x = torch.rand(1, 512, feat_len, device=device)
    # skip = torch.rand_like(x, device=device)
    # macs, params = profile(module, inputs=(x, ))
    # mb = 1000 * 1000
    # print(f"MACs: [{macs / mb / 1000}] Gb \nParams: [{params / mb}] Mb")
    # print("模型参数量详情：")
    # summary(module, input_data=(x, ), mode="train")