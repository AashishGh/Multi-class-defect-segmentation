from .unet_model import UNet
from .double_unet_model import DoubleUnet
from .transnetR_model import TransNetR
from .resunetplusplus import ResunetPlusplus
from .transrupnet_model import TransRUPNet
from .pvtformer import PVTFormer
# from .vmunet import VMUNet
# from .vmunetv2 import VMUNetV2
from .rma_mamba_s import RMAMamba_S


models_dict= {
    "unet": UNet(in_ch=3, out_ch=4),
    "doubleunet": DoubleUnet(),
    "transnetr": TransNetR(),
    "resunetplusplus": ResunetPlusplus(),
    "transrupnet": TransRUPNet(),
    "pvtformer": PVTFormer(),
    # "vmunet": VMUNet(num_classes=4),
    "rmamambas": RMAMamba_S(),
    # "vmunetv2": VMUNetV2(num_classes=4)
}

