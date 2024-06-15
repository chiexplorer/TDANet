###
# Author: Kai Li
# Date: 2022-02-12 15:16:35
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-10-04 16:24:53
###
from .base_model import BaseModel
from .TDANet import TDANet
from .TDANet_best import TDANetBest
from .TDANet_yang import TDANetYang
from .SeBlock import SEBasicBlock1D
from .TDANet_origin import TDANetOrigin
from .TDANet_mult_tes import TDANetMultRes
from .TDANet_attn import TDANetAttn
from .TDANet_chunk import TDANetChunk
from .TDANet_transxnet import TDANetTranXNet
from .TDANet_MSFFN import TDANetMSFFN

__all__ = [
    "BaseModel",
    "TDANet",
    "TDANetBest",
    "SEBasicBlock1D",
    "TDANetYang",
    "TDANetOrigin",
    "TDANetMultRes",
    "TDANetAttn",
    "TDANetChunk",
    "TDANetTranXNet",
    "TDANetMSFFN"
]


def register_model(custom_model):
    """Register a custom model, gettable with `models.get`.

    Args:
        custom_model: Custom model to register.

    """
    if (
        custom_model.__name__ in globals().keys()
        or custom_model.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Model {custom_model.__name__} already exists. Choose another name."
        )
    globals().update({custom_model.__name__: custom_model})


def get(identifier):
    """Returns an model class from a string (case-insensitive).

    Args:
        identifier (str): the model name.

    Returns:
        :class:`torch.nn.Module`
    """
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")
