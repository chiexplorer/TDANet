import os
import torch
import torch.nn as nn

def get_feat_lens(feat_len, depth):
    feat_len_tmp = feat_len
    feat_lens = [feat_len]
    for i in range(depth - 1):
        feat_len_tmp = (feat_len_tmp + 1) // 2
        feat_lens.append(feat_len_tmp)
    feat_lens.reverse()  # 翻转
    return feat_lens

def get_feat_len(feat_len, depth):
    feat_len_tmp = feat_len
    feat_lens = [feat_len]
    for i in range(depth - 1):
        feat_len_tmp = (feat_len_tmp + 1) // 2
        feat_lens.append(feat_len_tmp)
    feat_lens.reverse()  # 翻转
    return feat_len_tmp


if __name__ == '__main__':
    def pad_input(input, window, stride, fixed_len=None):
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

    def pad_input_fixed(input, window, stride, fixed_len=None):
        batch_size, nsample = input.shape

        if fixed_len is not None:
            target_len = (fixed_len - 1) * stride
            rest = (target_len - nsample) // 2
            input = nn.functional.pad(input, (rest, target_len-nsample-rest), 'constant', 0)
        else:
            # pad the signals at the end for matching the window/stride size
            rest = window - (stride + nsample % window) % window
            if rest > 0:
                pad = torch.zeros(batch_size, rest).type(input.type())
                input = torch.cat([input, pad], 1)
            pad_aux = torch.zeros(batch_size, window - stride).type(input.type())
            input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    # L = 24000
    # sr = 8000
    # s = 6
    # k = 24
    # p = 12
    # x = torch.rand(1, L)
    # x, _ = pad_input_fixed(x, k, s, fixed_len=4096)
    # encoder = nn.Conv1d(in_channels=1,
    #         out_channels=1,
    #         kernel_size=k,
    #         stride=s,
    #         padding=p,
    #         bias=False)
    # print(encoder(x).shape)
    print(get_feat_len(3010, 5))
