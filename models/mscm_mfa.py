import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.pvtv2 import pvt_v2_b2

from mamba_ssm import Mamba
import copy


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        f1 = self.conv1(x)
        f = self.conv2(f1)

        return f

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BiPixelMambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.mamba_forw = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )

        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)

        self.mamba_backw = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )


        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C = x.shape[:2]

        assert C == self.dim
        img_dims = x.shape[2:]

        x_div = x
        NB = x_div.shape[0]

        n_tokens = x_div.shape[2:].numel()

        x_flat = x_div.reshape(NB, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))
        x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        out = x_out + x

        return out


class MSFF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, num_class,bias):
        super(MSFF, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.num_class = num_class

        self.project_in = nn.Conv3d(dim, hidden_features*3, kernel_size=(1,1,1), bias=bias)
        self.project_in2 = nn.Conv2d(dim, hidden_features * 3, kernel_size=1, bias=bias)

        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=1, padding=1, groups=hidden_features, bias=bias)
   
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(1,1,1), bias=bias)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.project_in(x)
        x1,x2,x3 = x.chunk(3, dim=1)
        x1 = self.dwconv1(x1).squeeze(2)
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))
        x = F.gelu(x1)*x2*x3
        x = x.unsqueeze(2)
        x = self.project_out(x)
        x = x.squeeze(2)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_class, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.B_SS2D = BiPixelMambaLayer(dim)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.msff = MSFF(dim, 2, num_class, False)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.B_SS2D(self.norm1(x)))

        input = x
        x = self.norm1(x)
        x = self.msff(x)
        x = x.permute(0, 2, 3, 1)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class Bmoudle(nn.Module):
    def __init__(self, in_channels, out_channels,num_class):
        super().__init__()
        self.block = Block(in_channels,num_class)
        self.Multi = MultiScaleFusion(in_channels, out_channels)

    def forward(self, x):
        x1 = self.block(x)
        x2 = self.Multi(x)
        x = x1 + x2

        return x


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
    else:
        raise NotImplementedError
    return all_top_indices_x[:num_freq], all_top_indices_y[:num_freq]


class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self, in_channels, dct_h=7, dct_w=7, frequency_branches=16, frequency_selection='top', reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()
        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)
        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        for freq_idx in range(frequency_branches):
            self.register_buffer(f'dct_weight_{freq_idx}',
                                 self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // 2, kernel_size=1, bias=False))
        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)
        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y,
                                                                                                            mapper_y,
                                                                                                            tile_size_y)
        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        return result * math.sqrt(2) if freq != 0 else result

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x_pooled = x if (H == self.dct_h and W == self.dct_w) else F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg /= self.num_freq
        multi_spectral_feature_max /= self.num_freq
        multi_spectral_feature_min /= self.num_freq
        multi_spectral_attention_map = torch.sigmoid(
            self.fc(multi_spectral_feature_avg + multi_spectral_feature_max + multi_spectral_feature_min)).view(
            batch_size, C // 2, 1, 1)
        return multi_spectral_attention_map


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_avg = torch.mean(x, dim=1, keepdim=True)
        channel_max, _ = torch.max(x, dim=1, keepdim=True)
        x_attention = torch.cat([channel_avg, channel_max], dim=1)
        x_attention = self.conv(x_attention)
        x_attention = self.sigmoid(x_attention)
        return x_attention


class MFA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.multi_spectral_attention_map = MultiFrequencyChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x_g, x_m):
        ori_xg = x_g
        x_cat = torch.cat([x_g, x_m], dim=1)
        x_f = self.multi_spectral_attention_map(x_cat)
        y_g = x_f * self.alpha
        y_m = x_f * (1 - self.alpha)

        x1 = y_g * x_g
        x2 = y_m * x_m
        x_mg = x1 + x2
        x = x_mg + ori_xg
        x_att = self.sa(x)
        x1 = x_att * ori_xg
        x2 = x_att * x_mg
        output = x1 + x2
        return output

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pw1 = nn.Conv2d(64, 1, kernel_size=1)
        self.pw2 = nn.Conv2d(128, 1, kernel_size=1)
        self.pw3 = nn.Conv2d(320, 1, kernel_size=1)
        self.pw4 = nn.Conv2d(512, 1, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2)

        self.backbone = pvt_v2_b2()
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.B_SSD1 = Bmoudle(64, 64, 1)
        self.B_SSD2 = Bmoudle(128, 128, 1)
        self.B_SSD3 = Bmoudle(320, 320, 1)
        self.B_SSD4 = Bmoudle(512, 512, 1)

        self.conv_out = nn.Conv2d(4, 9, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Conv2d(512, 320, 1)
        self.conv2 = nn.Conv2d(320, 128, 1)
        self.conv3 = nn.Conv2d(128, 64, 1)

        self.mfa1 = MFA(128)
        self.mfa2 = MFA(256)
        self.mfa3 = MFA(640)

    def forward(self, x):
        if x.size()[1] == 1:
            x = self.conv(x)
        H, W = x.size()[2:]

        x1, x2, x3, x4 = self.backbone(x)
        skip3 = x3
        skip2 = x2
        skip1 = x1

        x4_ = self.up(x4)
        x4 = self.B_SSD4(x4_)
        x41 = self.conv1(x4)
        skip3 = self.mfa3(skip3, x41)

        x3_ = self.up(skip3)
        x3 = self.B_SSD3(x3_)
        x31 = self.conv2(x3)
        skip2 = self.mfa2(skip2, x31)

        x2_ = self.up(skip2)
        x2 = self.B_SSD2(x2_)
        x21 = self.conv3(x2)
        skip1 = self.mfa1(skip1, x21)

        x1_ = self.up(skip1)
        x1 = self.B_SSD1(x1_)

        x_in4 = self.pw4(x4)
        x_in3 = self.pw3(x3)
        x_in2 = self.pw2(x2)
        x_in1 = self.pw1(x1)

        x_in4 = F.interpolate(x_in4, size=(H, W), mode="bilinear", align_corners=False)
        x_in3 = F.interpolate(x_in3, size=(H, W), mode="bilinear", align_corners=False)
        x_in2 = F.interpolate(x_in2, size=(H, W), mode="bilinear", align_corners=False)
        x_in1 = F.interpolate(x_in1, size=(H, W), mode="bilinear", align_corners=False)
        x = torch.cat([x_in4, x_in3, x_in2, x_in1], dim=1)
        x = self.conv_out(x)

        return self.sigmoid(x)
