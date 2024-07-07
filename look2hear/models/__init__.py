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
from .TDANet_gate_variant import TDANetGateVariant
from .TDANet_OSRA_gated import TDANetGateOSRA
from .TDANet_dynamic_down import TDANetDynamicDownsample
from .TDANet_ULayer_num import TDANetULayerNum
from .TDANet_EMCAD import TDANetEMCAD
from .TDANet_intergral import TDANetEMCAD_v1
from .TDANet_intergral_v1_3 import TDANetEMCADv1_3
from .TDANet_intergral_v1_4 import TDANetEMCADv1_4
from .TDANet_intergral_v1_5 import TDANetEMCADv1_5
from .TDANet_intergral_v1_6 import TDANetEMCADv1_6
from .TDANet_EMCAD_f1 import TDANetEMCADF1
from .TDANet_no_drop import TDANetNoDrop

from .TDANet_intergralV1_6_noIDConv import TDANetEMCADv1_6_noIDConv
from .TDANet_intergral_v1_6_noASG import TDANetEMCADv1_6_noASG
from .TDANet_intergral_v1_6_noMMLP import TDANetEMCADv1_6_noMMLP


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
    "TDANetMSFFN",
    "TDANetGateVariant",
    "TDANetGateOSRA",
    "TDANetDynamicDownsample",
    "TDANetULayerNum",
    "TDANetEMCAD",
    "TDANetEMCADF1",
    "TDANetNoDrop",
    "TDANetEMCADv1_3",
    "TDANetEMCADv1_4",
    "TDANetEMCADv1_5",
    "TDANetEMCADv1_6",
    "TDANetEMCADv1_6_noIDConv",
    "TDANetEMCADv1_6_noASG",
    "TDANetEMCADv1_6_noMMLP"
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
