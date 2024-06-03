import os
import torch
import torch.nn as nn

def get_feat_len(feat_len, depth):
    feat_len_tmp = feat_len
    feat_lens = [feat_len]
    for i in range(depth - 1):
        feat_len_tmp = (feat_len_tmp + 1) // 2
        feat_lens.append(feat_len_tmp)
    feat_lens.reverse()  # 翻转
    return feat_lens



if __name__ == '__main__':
    depth = 5
    x = torch.rand(2, 128, 3010)
    feat_len = x.size(2)
    print(feat_len)
    print(get_feat_len(feat_len, depth))
