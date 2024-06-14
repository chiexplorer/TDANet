import torch
import torch.nn as nn
import os
import numpy

def static_at_dim(x, dim=1):
    """ show mean and var of x at specific dim """
    for i in range(x.shape[dim]):
        if dim == 0:
            print(f"[{i}]th channel——mean:{x[i, ...].mean()}, var:{x[i, ...].var()}")
        elif dim == 1:
            print(f"[{i}]th channel——mean:{x[:, i, ...].mean()}, var:{x[:, i, ...].var()}")
        elif dim == 2:
            print(f"[{i}]th channel——mean:{x[:, :, i, ...].mean()}, var:{x[:, :, i, ...].var()}")


if __name__ == '__main__':
    x = torch.randint(50, (1, 24), dtype=torch.float).view(2, 3, 4)
    print("origin x: \n", x)
    static_at_dim(x, dim=1)
    bn = nn.BatchNorm1d(num_features=3)

    x_bn = bn(x)
    print("batch normed: \n", x_bn)
    # 验证batch norm效果
    static_at_dim(x_bn, dim=1)

    ln = nn.LayerNorm(normalized_shape=[3, 4])
    x_ln = ln(x)
    print("layer normed: \n", x_ln)
    # 验证layer norm效果
    static_at_dim(x_ln, dim=0)



