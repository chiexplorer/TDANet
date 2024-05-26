import os
import torch
import torch.nn as nn


if __name__ == '__main__':
    depth = 5
    x = torch.rand(2, 128, 2010)
    feat_len = x.size(2)
    print(feat_len)
    feat_len_tmp = feat_len
    feat_lens = [feat_len]
    for i in range(depth - 2):
        feat_len_tmp = (feat_len_tmp + 1) // 2
        feat_lens.append(feat_len_tmp)
    feat_lens.reverse()  # 翻转
    print("长度：", feat_lens)