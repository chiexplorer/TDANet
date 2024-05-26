import torch
from torchvision.models import resnet50
from thop import profile


if __name__ == '__main__':
    model = resnet50()
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,))
    mb = 1024*1024
    print(f"MACs: {macs/mb}Mb, Params: {params/mb}Mb")
