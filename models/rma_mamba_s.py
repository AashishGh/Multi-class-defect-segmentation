import torch
import torch.nn as nn
import torch.nn.functional as F
from .vmamba_rma import VSSM, LayerNorm2d, VSSBlock, Permute, Backbone_VSSM
# from .data_process import get_data_augmentation, get_transforms

def DiceBCELoss(inputs, targets, smooth=1):
    inputs = torch.sigmoid(inputs)

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    Dice_BCE = BCE + dice_loss

    return Dice_BCE

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RMAttention(nn.Module):
    def __init__(self, channel, channel_first, **cfg):
        super(RMAttention, self).__init__()

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        # norm_layer: nn.Module = _NORMLAYERS.get(cfg['norm_layer'].lower(), None)
        # ssm_act_layer: nn.Module = _ACTLAYERS.get(cfg['ssm_act_layer'].lower(), None)
        # mlp_act_layer: nn.Module = _ACTLAYERS.get(cfg['mlp_act_layer'].lower(), None)

        # ---- defaults + user overrides ----
        defaults = dict(
            norm_layer="ln2d",
            ssm_act_layer="silu",
            mlp_act_layer="gelu",
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_drop_rate=0.0,
            gmlp=False,
            use_checkpoint=False,
        )
        defaults.update(cfg)   # cfg here is the user-passed kwargs
        cfg = defaults

        norm_layer = _NORMLAYERS.get(str(cfg["norm_layer"]).lower(), LayerNorm2d)
        ssm_act_layer = _ACTLAYERS.get(str(cfg["ssm_act_layer"]).lower(), nn.SiLU)
        mlp_act_layer = _ACTLAYERS.get(str(cfg["mlp_act_layer"]).lower(), nn.GELU)


        self.st_block1 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=channel, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=cfg['ssm_d_state'], ssm_ratio=cfg['ssm_ratio'], ssm_dt_rank=cfg['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=cfg['ssm_conv'], ssm_conv_bias=cfg['ssm_conv_bias'], ssm_drop_rate=cfg['ssm_drop_rate'], ssm_init=cfg['ssm_init'],
                forward_type=cfg['forward_type'], mlp_ratio=cfg['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=cfg['mlp_drop_rate'],
                gmlp=cfg['gmlp'], use_checkpoint=cfg['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block2 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=channel, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=cfg['ssm_d_state'], ssm_ratio=cfg['ssm_ratio'], ssm_dt_rank=cfg['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=cfg['ssm_conv'], ssm_conv_bias=cfg['ssm_conv_bias'], ssm_drop_rate=cfg['ssm_drop_rate'], ssm_init=cfg['ssm_init'],
                forward_type=cfg['forward_type'], mlp_ratio=cfg['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=cfg['mlp_drop_rate'],
                gmlp=cfg['gmlp'], use_checkpoint=cfg['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.ra_conv1 = BasicConv2d(channel, channel, 3, 1, padding=1)
        self.ra_conv2 = BasicConv2d(channel, channel, 3, 1, padding=1)
        self.map_head = nn.Conv2d(channel, 1, 1)   # 1-channel map for next stage


    def forward(self, x, map):
        b, c, h, w = x.shape
        base_size = (h, w)
        map = self.res(map, base_size)
        attn = -1*(torch.sigmoid(map)) + 1

        identity = x
        x = self.st_block1(x)
        x = attn.expand(-1, c, -1, -1).mul(x)
        x = self.ra_conv1(x)
        x = self.st_block2(x)
        x += identity
        feat = self.ra_conv2(x)          # (B, channel, h, w)
        new_map = self.map_head(feat)    # (B, 1, h, w)
        res_map = new_map + map          # residual refine
        return feat, res_map



class RMAMamba_T(nn.Module):
    def __init__(self, pretrained=None, channel=32, **kwargs):
        super(RMAMamba_T, self).__init__()
        defaults = dict(
            norm_layer="ln2d",
            ssm_act_layer="silu",
            mlp_act_layer="gelu",
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_drop_rate=0.0,
            gmlp=False,
            use_checkpoint=False,
        )
        defaults.update(cfg)   # cfg here is the user-passed kwargs
        cfg = defaults

        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **cfg)
        self.channel_first = self.encoder.channel_first

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        self.translayer1_st = BasicConv2d(96, channel, 1)
        self.translayer2_st = BasicConv2d(192, channel, 1)
        self.translayer3_st = BasicConv2d(384, channel, 1)
        self.translayer4_st = BasicConv2d(768, channel, 1)

        self.out_conv1 = nn.Conv2d(channel, 1, 1)

        self.attention1 = RMAttention(channel, self.channel_first, **cfg)
        self.attention2 = RMAttention(channel, self.channel_first, **cfg)
        self.attention3 = RMAttention(channel, self.channel_first, **cfg)

        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        self.loss_fn = DiceBCELoss

    def forward(self, sample):
        if torch.is_tensor(sample):
            x = sample
            y = None
        else:
            x = sample["images"]
            y = sample.get("masks", None)
        base_size = x.shape[-2:]

        features = self.encoder(x)
        x1 = features[0]  # 8, 96, 64, 64
        x2 = features[1]  # 8, 192, 32, 32
        x3 = features[2]  # 8, 384, 16, 16
        x4 = features[3]  # 8, 768, 8, 8

        x4_st = self.translayer4_st(x4)
        a4 = self.out_conv1(x4_st)
        out4 = self.res(a4, base_size)

        x1_st = self.translayer1_st(x1)
        x2_st = self.translayer2_st(x2)
        x3_st = self.translayer3_st(x3)

        a3 = self.attention3(x3_st, a4)
        out3 = self.res(a3, base_size)

        a2 = self.attention2(x2_st, a3)
        out2 = self.res(a2, base_size)

        a1 = self.attention1(x1_st, a2)
        out1 = self.res(a1, base_size)
        
        if y is None:
            return out1
        
        loss4 = self.loss_fn(out4, y)
        loss3 = self.loss_fn(out3, y)
        loss2 = self.loss_fn(out2, y)
        loss1 = self.loss_fn(out1, y)
        loss = loss1 + loss2 + loss3 + loss4

        return {'prediction': out1, 'loss': loss}


class RMAMamba_S(nn.Module):
    def __init__(self, pretrained=None, channel=32, num_classes=4,**cfg):
        super(RMAMamba_S, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **cfg)
        self.channel_first = self.encoder.channel_first
        self.num_classes = num_classes
        self.cls_head = nn.Conv2d(channel, num_classes, 1)
        self.map4_head = nn.Conv2d(channel, 1, 1)




        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        # norm_layer: nn.Module = _NORMLAYERS.get(cfg['norm_layer'].lower(), None)
        # ssm_act_layer: nn.Module = _ACTLAYERS.get(cfg['ssm_act_layer'].lower(), None)
        # mlp_act_layer: nn.Module = _ACTLAYERS.get(cfg['mlp_act_layer'].lower(), None)
        # ---- defaults + user overrides ----
        defaults = dict(
            norm_layer="ln2d",
            ssm_act_layer="silu",
            mlp_act_layer="gelu",
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_drop_rate=0.0,
            gmlp=False,
            use_checkpoint=False,
        )
        defaults.update(cfg)   # cfg here is the user-passed kwargs
        cfg = defaults

        norm_layer = _NORMLAYERS.get(str(cfg["norm_layer"]).lower(), LayerNorm2d)
        ssm_act_layer = _ACTLAYERS.get(str(cfg["ssm_act_layer"]).lower(), nn.SiLU)
        mlp_act_layer = _ACTLAYERS.get(str(cfg["mlp_act_layer"]).lower(), nn.GELU)


        self.st_block1 = nn.Sequential(
            Permute(0, 2, 3, 1) if not self.channel_first else nn.Identity(),
            VSSBlock(hidden_dim=96, drop_path=0.1, norm_layer=norm_layer, channel_first=self.channel_first,
                ssm_d_state=cfg['ssm_d_state'], ssm_ratio=cfg['ssm_ratio'], ssm_dt_rank=cfg['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=cfg['ssm_conv'], ssm_conv_bias=cfg['ssm_conv_bias'], ssm_drop_rate=cfg['ssm_drop_rate'], ssm_init=cfg['ssm_init'],
                forward_type=cfg['forward_type'], mlp_ratio=cfg['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=cfg['mlp_drop_rate'],
                gmlp=cfg['gmlp'], use_checkpoint=cfg['use_checkpoint']),
            Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity(),
        )
        self.translayer1_st = BasicConv2d(96, channel, 1)

        self.st_block2 = nn.Sequential(
            Permute(0, 2, 3, 1) if not self.channel_first else nn.Identity(),
            VSSBlock(hidden_dim=192, drop_path=0.1, norm_layer=norm_layer, channel_first=self.channel_first,
                ssm_d_state=cfg['ssm_d_state'], ssm_ratio=cfg['ssm_ratio'], ssm_dt_rank=cfg['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=cfg['ssm_conv'], ssm_conv_bias=cfg['ssm_conv_bias'], ssm_drop_rate=cfg['ssm_drop_rate'], ssm_init=cfg['ssm_init'],
                forward_type=cfg['forward_type'], mlp_ratio=cfg['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=cfg['mlp_drop_rate'],
                gmlp=cfg['gmlp'], use_checkpoint=cfg['use_checkpoint']),
            Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity(),
        )
        self.translayer2_st = BasicConv2d(192, channel, 1)

        self.st_block3 = nn.Sequential(
            Permute(0, 2, 3, 1) if not self.channel_first else nn.Identity(),
            VSSBlock(hidden_dim=384, drop_path=0.1, norm_layer=norm_layer, channel_first=self.channel_first,
                     ssm_d_state=cfg['ssm_d_state'], ssm_ratio=cfg['ssm_ratio'],
                     ssm_dt_rank=cfg['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=cfg['ssm_conv'], ssm_conv_bias=cfg['ssm_conv_bias'],
                     ssm_drop_rate=cfg['ssm_drop_rate'], ssm_init=cfg['ssm_init'],
                     forward_type=cfg['forward_type'], mlp_ratio=cfg['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=cfg['mlp_drop_rate'],
                     gmlp=cfg['gmlp'], use_checkpoint=cfg['use_checkpoint']),
            Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity(),
        )
        self.translayer3_st = BasicConv2d(384, channel, 1)

        self.st_block4 = nn.Sequential(
            Permute(0, 2, 3, 1) if not self.channel_first else nn.Identity(),
            VSSBlock(hidden_dim=768, drop_path=0.1, norm_layer=norm_layer, channel_first=self.channel_first,
                     ssm_d_state=cfg['ssm_d_state'], ssm_ratio=cfg['ssm_ratio'],
                     ssm_dt_rank=cfg['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=cfg['ssm_conv'], ssm_conv_bias=cfg['ssm_conv_bias'],
                     ssm_drop_rate=cfg['ssm_drop_rate'], ssm_init=cfg['ssm_init'],
                     forward_type=cfg['forward_type'], mlp_ratio=cfg['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=cfg['mlp_drop_rate'],
                     gmlp=cfg['gmlp'], use_checkpoint=cfg['use_checkpoint']),
            Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity(),
        )
        self.translayer4_st = BasicConv2d(768, channel, 1)
        self.out_conv1 = nn.Conv2d(channel, 1, 1)

        self.attention1 = RMAttention(channel, self.channel_first, **cfg)
        self.attention2 = RMAttention(channel, self.channel_first, **cfg)
        self.attention3 = RMAttention(channel, self.channel_first, **cfg)

        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        self.loss_fn = DiceBCELoss

    def forward(self, sample):
        # x = sample['images']
        # y = sample['masks']
        if torch.is_tensor(sample):
            x = sample
            y = None
        else:
            x = sample["images"]
            y = sample.get("masks", None)
        
        base_size = x.shape[-2:]

        features = self.encoder(x)
        x1 = features[0]  # 8, 96, 64, 64
        x2 = features[1]  # 8, 192, 32, 32
        x3 = features[2]  # 8, 384, 16, 16
        x4 = features[3]  # 8, 768, 8, 8

        x4_st = self.st_block4(x4)
        x4_st = self.translayer4_st(x4_st)
        a4 = self.out_conv1(x4_st)
        out4 = self.res(a4, base_size)

        x1_st = self.st_block1(x1)
        x1_st = self.translayer1_st(x1_st)
        x2_st = self.st_block2(x2)
        x2_st = self.translayer2_st(x2_st)
        x3_st = self.st_block3(x3)
        x3_st = self.translayer3_st(x3_st)

        # a3 = self.attention3(x3_st, a4)
        # out3 = self.res(a3, base_size)

        # a2 = self.attention2(x2_st, a3)
        # out2 = self.res(a2, base_size)

        # a1 = self.attention1(x1_st, a2)
        # out1 = self.res(a1, base_size)

        # if y is None:
        #     return out1

        # loss4 = self.loss_fn(out4, y)
        # loss3 = self.loss_fn(out3, y)
        # loss2 = self.loss_fn(out2, y)
        # loss1 = self.loss_fn(out1, y)
        # loss = loss1 + loss2 + loss3 + loss4

        # return {'prediction': out1, 'loss': loss}
        # stage-4: build map4 from features
        x4_st = self.st_block4(x4)
        x4_st = self.translayer4_st(x4_st)
        map4 = self.map4_head(x4_st)                 # (B,1,h4,w4)

        # stage-3/2/1: attention returns (feat, map)
        feat3, map3 = self.attention3(x3_st, map4)
        feat2, map2 = self.attention2(x2_st, map3)
        feat1, map1 = self.attention1(x1_st, map2)

        # final multi-class logits from features
        logits = self.cls_head(feat1)                # (B,4,h1,w1)
        logits = self.res(logits, base_size)         # (B,4,H,W)
        return logits




if __name__ == '__main__':
    ras = RMAMamba_T().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    sample = {
        "images": input_tensor,
        "masks": torch.zeros(1, 1, 352, 352, device=input_tensor.device),
    }
    out = ras(sample)
    print(out["prediction"].shape, out["loss"].item())
