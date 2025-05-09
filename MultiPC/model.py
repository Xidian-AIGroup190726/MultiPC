import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import matplotlib.pyplot as plt
import numpy as np

def get_hp(data):
    r""" get the high-frequency of input images,
    first calculate the avg_filter of the input as low-frequency,
    subtract the low-frequency to get the high-frequency

    Args:
    data (torch.Tensor): image matrix, shape of [N, C, H, W]
    Returns:
    torch.Tensor: high-frequency part of input, shape of [N, C, H, W]
    """
    rs = F.avg_pool2d(data, kernel_size=5, stride=1, padding=2)
    rs = data - rs
    return rs


def visualize_feature_maps(p1, pans1, hp1, p2, pans2, hp2, channel=0):
    """
    可视化两组特征图（p1, pans1, hp1）和（p2, pans2, hp2）在指定通道上的等高线图，每张图加 colorbar。

    Parameters:
    - p1, pans1, hp1: 第一组特征图 (tensor)，形状为 [B, C, H, W]
    - p2, pans2, hp2: 第二组特征图 (tensor)，形状为 [B, C, H, W]
    - channel: 要可视化的通道索引（默认为0）
    """

    # 提取指定batch和通道的特征
    p1_feature = p1[0, channel].detach().cpu().numpy()
    pans1_feature = pans1[0, channel].detach().cpu().numpy()
    hp1_feature = hp1[0, channel].detach().cpu().numpy()

    p2_feature = p2[0, channel].detach().cpu().numpy()
    pans2_feature = pans2[0, channel].detach().cpu().numpy()
    hp2_feature = hp2[0, channel].detach().cpu().numpy()

    features = [
        (p1_feature, 'P1 Input'),
        (pans1_feature, 'Pansharpening Input 1'),
        (hp1_feature, 'HP1 Output'),
        (p2_feature, 'P2 Input'),
        (pans2_feature, 'Pansharpening Input 2'),
        (hp2_feature, 'HP2 Output')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for idx, (feature, title) in enumerate(features):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        # 绘制等高线
        contour = ax.contourf(feature, levels=50, cmap='viridis')
        ax.set_title(f'{title} - Channel {channel}')
        ax.axis('off')

        # 加 colorbar
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
class MultiTaskModel(nn.Module):
    def __init__(self,num_classes=7,Wavelet='haar'):
        super(MultiTaskModel, self).__init__()

        self.conv_ms=nn.Conv2d(4,64,kernel_size=3, stride=1, padding=1)
        self.conv_pan=nn.Conv2d(1,64,kernel_size=3, stride=1, padding=1)

        # 分类分支的前半部分
        self.dwt_ms=DWT_ms(wavelet=Wavelet)
        self.dwt_pan=DWT_pan(wavelet=Wavelet)
        # 可学习的加权系数
        self.weight_low_ms = nn.Parameter(torch.ones(1))
        self.weight_high_ms = nn.Parameter(torch.ones(1))
        self.weight_low_pan = nn.Parameter(torch.ones(1))
        self.weight_high_pan = nn.Parameter(torch.ones(1))

        # 在初始化时自定义权重#👉又train上了，头不疼了？👈
        nn.init.constant_(self.weight_low_ms, 0.1)  # 低频 MS 权重初始化为 0.1
        nn.init.constant_(self.weight_high_ms, 10)  # 高频 MS 权重初始化为 10.0
        nn.init.constant_(self.weight_low_pan, 0.1)  # 低频 PAN 权重初始化为 0.1
        nn.init.constant_(self.weight_high_pan, 10)  # 高频 PAN 权重初始化为 10.0

        # 共享的cross fusion模块
        self.shared_cross_fusion1 = CollaborativeLearningModule(128,8,16)
        self.shared_cross_fusion2 = CollaborativeLearningModule(256,4,8)

        # Pansharpening分支的后续层
        self.residualblock_f1 = ResidualBlock(128)
        # self.downsample_pans_1 = downsample(64, 128)
        self.residualblock_f2 = ResidualBlock(128)
        self.downsample_pans_2 = downsample(128, 256)
        self.residualblock_f3 = ResidualBlock(256)
        self.downsample_pans_3 = downsample(256, 512)

        self.fextra = FeatureExtraction()

        # 分类分支的后续层
        self.conv_low = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1)
        self.conv_high = nn.Conv2d(15, 64, kernel_size=3, stride=1, padding=1)

        self.residualblock_low_11 = ResidualBlock(64)
        self.downsample_low_1 = downsample(64, 128)
        self.residualblock_low_12 = ResidualBlock(128)

        self.residualblock_high_11 = ResidualBlock(64)
        self.downsample_high_1 = downsample(64, 128)
        self.residualblock_high_12 = ResidualBlock(128)

        #cross

        self.residualblock_low_21 = ResidualBlock(128)
        self.downsample_low_2 = downsample(128, 256)
        self.residualblock_low_22 = ResidualBlock(256)

        self.residualblock_high_21 = ResidualBlock(128)
        self.downsample_high_2 = downsample(128, 256)
        self.residualblock_high_22 = ResidualBlock(256)

        #cross

        self.residualblock_low_31 = ResidualBlock(256)
        self.downsample_low_3 = downsample(256, 512)
        self.residualblock_low_32 = ResidualBlock(512)
        self.avgpool_low_3 = nn.AdaptiveAvgPool2d((2, 2))

        self.residualblock_high_31 = ResidualBlock(256)
        self.downsample_high_3 = downsample(256, 512)
        self.residualblock_high_32 = ResidualBlock(512)
        self.maxpool_high_3 = nn.AdaptiveMaxPool2d((2, 2))

        self.cross_attention_3 = Cross_attention(512)
        # self.cross_attention_3 = Overlapping_Cross_Attention(512,)
        self.fusion_map_3 = Fusion_map()

        self.linear_1 = nn.Linear(512, 128)
        self.relu = nn.ReLU(True)
        self.linear_2 = nn.Linear(128, num_classes)

        self.hp_adapter1 = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),  # 调整通道数
            nn.AdaptiveAvgPool2d(output_size=(16, 16))  # 强制匹配目标尺寸
        )
        self.hp_adapter2 = nn.Sequential(
            nn.Conv2d(256, 4, kernel_size=1),  # 调整通道数
            nn.AdaptiveAvgPool2d(output_size=(16, 16))  # 强制匹配目标尺寸
        )
    def forward(self, ms, panc,device):
        lrms = nn.functional.interpolate(ms, scale_factor=0.25, mode='bilinear', align_corners=True)
        upscaled_lrms = nn.functional.interpolate(lrms, scale_factor=4, mode='bilinear', align_corners=True)
        panp = nn.functional.interpolate(panc, scale_factor=0.25, mode='bilinear', align_corners=True)

        upscaled_ms = nn.functional.interpolate(ms, scale_factor=4, mode='bilinear', align_corners=True)
        # upscaled_ms = nn.functional.interpolate(upscaled_lrms, scale_factor=4, mode='bilinear', align_corners=True)

        #pansharping部分
        f_m = get_hp(lrms)
        f_m = nn.functional.interpolate(f_m, scale_factor=4, mode='bilinear', align_corners=True)
        f_pan = get_hp(panp)

        f_m = self.conv_ms(f_m)
        f_pan = self.conv_pan(f_pan)
        # print(f_pan.shape,f_m.shape)
        p = torch.cat([f_m, f_pan], dim=1)
        # print(p.shape)

        # 分类部分
        lowms,highms=self.dwt_ms(upscaled_ms)
        lowpan,highpan=self.dwt_pan(panc)
        # print(f"low and high{lowms.shape}{lowpan.shape}")
        # print(f"low and high{highms.shape}{highpan.shape}")

        # # 拼接加权后的低频和高频特征
        low_freq=torch.cat([lowms,lowpan],dim=1) #5*32*32
        high_freq=torch.cat([highms,highpan],dim=1) #15*32*32
        # print(f"low and high{low_freq.shape}{high_freq.shape}")

        f_low = self.conv_low(low_freq.to(device))
        f_high = self.conv_high(high_freq.to(device))
        # print(f"low and high{f_low.shape}{f_high.shape}")

        # 第一组处理
        # low path 1
        l1 = self.residualblock_low_11(f_low)
        l1 = self.downsample_low_1(l1)
        l1 = self.residualblock_low_12(l1)

        # high path 1
        h1 = self.residualblock_high_11(f_high)
        h1 = self.downsample_high_1(h1)
        h1 = self.residualblock_high_12(h1)
        # print(f"low and high{h1.shape}")
        # print(p.shape,h1.shape)

        # pansharping部分
        p1 = self.residualblock_f1(p)
        # p1 = self.residualblock_f1(p)
        # p1 = self.patt1(p1)
        # p1 = self.downsample_pans_1(p1)

        # print(p1.shape,h1.shape)
        hp1, hc1, fp1, fc1,cla1,pans1 = self.shared_cross_fusion1(p1, h1)
        # 调用函数可视化第0个通道的特征图
        # visualize_feature_maps(p1, hp1, channel=0)
        # print(hp1.shape, hc1.shape)
        # 分类第二组处理
        # low path 2
        l2 = self.residualblock_low_21(l1)
        l2 = self.downsample_low_2(l2)
        l2 = self.residualblock_low_22(l2)

        # high path 2
        h2 = self.residualblock_high_21(hc1)
        # h2 = self.residualblock_high_21(hc1)
        h2 = self.downsample_high_2(h2)
        h2 = self.residualblock_high_22(h2)
        # print(p.shape,h2.shape)

        # pansharping部分
        p2 = self.residualblock_f2(hp1)
        # p2 = self.patt2(p2)
        p2 = self.downsample_pans_2(p2)

        hp2, hc2, fp2, fc2,cla2,pans2 = self.shared_cross_fusion2(p2, h2)
        # 调用函数可视化 p1, hp1, p2 和 hp2
        # visualize_feature_maps(p1, pans1,hp1, p2, pans2, hp2, channel=0)

        p3 = self.residualblock_f3(hp2)
        # p3 = self.patt3(p3)
        p3 = self.downsample_pans_3(p3)
        outp = self.fextra(p3)

        # print(outp.shape,upscaled_lrms.shape,hp2.shape)
        outp = outp + upscaled_lrms

        # 第三组处理
        # low path 3
        l3 = self.residualblock_low_31(l2)
        l3 = self.downsample_low_3(l3)
        l3 = self.residualblock_low_32(l3)
        l3 = self.avgpool_low_3(l3)

        # high path 3
        # h3 = self.residualblock_high_31(hc2)
        # h3 = self.residualblock_high_31(h2)
        h3 = self.residualblock_high_31(hc2)
        h3 = self.downsample_high_3(h3)
        h3 = self.residualblock_high_32(h3)
        h3 = self.maxpool_high_3(h3)
        # print(l3.shape,h3.shape)

        # 注意力和融合
        low_feature_map_fusion,high_feature_map_fusion = self.cross_attention_3(l3, h3)
        fused = self.fusion_map_3(low_feature_map_fusion,high_feature_map_fusion)

        # 展平并通过全连接层
        fused = fused.view(fused.size(0), -1)  # 展平操作
        out = self.linear_1(fused)
        out = self.relu(out)
        outc = self.linear_2(out)

        hp1 = self.hp_adapter1(p1)
        hp2 = self.hp_adapter2(p2)
        # print(l1.shape,h1.shape)
        return {
            'outc': outc,  # 分类输出
            'outp': outp,  # 全色锐化输出
            'features': {
                'low': {'l1': l1, 'l2': l2},  # 低频路径特征
                'high': {'hc1': h1, 'hc2': h2},  # 高频路径特征
                'pan': {'hp1': hp1, 'hp2': hp2},  # 全色锐化特征
            }
        }

class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()

        self.up1 = nn.ConvTranspose2d(512, 64, kernel_size=4, stride=2, padding=1)  # 空间尺寸变为 2 倍
        self.up2 = nn.ConvTranspose2d(64, 4, kernel_size=4, stride=2, padding=1)  # 空间尺寸变为 2 倍

    def forward(self, x):
        x = self.up1(x)  # 第一次上采样
        x = self.up2(x)  # 第二次上采样
        # x = self.up3(x)  # 第三次上采样
        return x


class upsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(upsample, self).__init__()

        # self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=3, stride=2, padding=1)
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.down= nn.PixelUnshuffle(2)
        # self.bn=nn.BatchNorm2d(out_channels)
        # self.relu=nn.ReLU(True)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class downsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(downsample, self).__init__()

        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=3, stride=2, padding=1)
        # self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        # self.down= nn.PixelUnshuffle(2)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(True)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels//2, 1, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(in_channels//2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 多层次的空间特征提取
        feat = self.conv1(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        attention = self.conv2(feat)
        attention = self.sigmoid(attention)
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFusion, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.fusion = nn.Conv2d(in_channels * 3, in_channels, 1)

        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        # 添加最终的归一化和激活
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 多尺度特征提取
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)

        # 特征拼接
        feat_concat = torch.cat([feat1, feat2, feat3], dim=1)

        # 特征融合
        feat_fused = self.fusion(feat_concat)

        # 注意力加权
        attention_weights = self.attention(feat_fused)
        feat_refined = feat_fused * attention_weights

        # 残差连接
        out = feat_refined + x

        # 最终处理
        out = self.norm(out)
        out = self.relu(out)

        return out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = ChannelAttention(in_channels)
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_weights

class DynamicFeatureBias(nn.Module):
    def __init__(self, dim, num_heads, residual, size):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.size = size
        self.pos_dim = dim // 4

        # 位置偏置生成网络
        self.pos_proj = nn.Linear(dim, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, 1)  # 输出通道数为 1，生成注意力权重
        )

    def forward(self, biases):
        # 生成注意力权重矩阵
        biases = biases.flatten(2).transpose(1, 2)  # (B, H*W, pos_dim)
        # print(biases.shape)
        pos = self.pos_proj(biases)  # (B, H*W, pos_dim)
        if self.residual:
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)  # (B, H*W, 1)
        else:
            pos = self.pos3(self.pos2(self.pos1(pos)))

        # 调整形状为 (B, 1, H, W)
        attention_weights = pos.transpose(1, 2).view(biases.size(0), 1, self.size, self.size)
        return attention_weights

class CollaborativeLearningModule(nn.Module):
    def __init__(self, in_channels,window_size,size):
        super(CollaborativeLearningModule, self).__init__()

        self.cross_attention1 = Overlapping_Cross_Attention(in_channels,window_size,0.3,1)
        self.cross_attention2 = Overlapping_Cross_Attention(in_channels,window_size,0.3,1)

        # self.dynamic_pos_bias1 = DynamicFeatureBias(dim=in_channels, num_heads=1, residual=True,size=size)
        # 用于全色锐化任务的特征提取
        self.fc_to_fp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            # 添加空洞卷积以扩大感受野，捕捉更多的边缘和纹理信息
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            # 添加注意力机制以关注重要的光谱特性
            nn.Conv2d(in_channels//2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.fp_to_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 计算 offset（Δp_k）
        self.offset_conv1 = nn.Conv2d(in_channels * 2, 18, kernel_size=3, padding=1)
        self.offset_conv2 = nn.Conv2d(in_channels * 2, 18, kernel_size=3, padding=1)
        # 可变形卷积
        self.deform_conv1 = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.deform_conv2 = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # 融合特征 F = Conv([F_p, F_c'])
        self.fusion_conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.fusion_conv2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)

        self.last1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.last2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)

    def forward(self, U_pansharp, U_classify):
        # print(U_pansharp.shape,U_classify.shape)
        # 1. 特征提取：从FC提取对FP有利的特征,从FP提取对FC有利的特征
        FP_features = self.fc_to_fp(U_classify)
        FC_features = self.fp_to_fc(U_pansharp)

        # 2. 计算偏移量（用于分类任务增强）
        offset_classify = self.offset_conv1(torch.cat([FP_features, FC_features], dim=1))
        FC_deform = self.deform_conv1(FC_features, offset_classify)

        # 3. 特征融合（增强分类任务）
        classfi_fused_features = self.fusion_conv1(torch.cat([FP_features, FC_deform], dim=1))

        # 4. 计算偏移量（用于锐化任务增强）
        offset_pansharp = self.offset_conv2(torch.cat([FP_features, FC_features], dim=1))  # 可以使用相同的 offset_conv
        FP_deform = self.deform_conv2(FP_features, offset_pansharp)

        # 5. 特征融合（增强锐化任务）
        pansharp_fused_features = self.fusion_conv2(torch.cat([FC_features, FP_deform], dim=1))

        # 6. 交叉注意力（增强分类任务）
        U_classify1, U_classify2 = self.cross_attention1(classfi_fused_features, U_classify)
        U_classify = torch.cat((U_classify1, U_classify2), dim=1)
        # U_classify = torch.cat((classfi_fused_features, U_classify), dim=1)
        # 7. 交叉注意力（增强锐化任务）
        U_pansharp1, U_pansharp2 = self.cross_attention2(pansharp_fused_features, U_pansharp)
        U_pansharp = torch.cat((U_pansharp1, U_pansharp2), dim=1)
        # U_pansharp = torch.cat((pansharp_fused_features, U_pansharp), dim=1)

        U_classify = self.last1(U_classify)
        U_pansharp = self.last2(U_pansharp)

        return U_pansharp,U_classify,FP_features,FC_features,classfi_fused_features,pansharp_fused_features

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1)  # B, H, W, C
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, c, h, w)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    c = windows.shape[-1]

    # 重塑为原始窗口排列
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, c)

    # 调整维度顺序并合并窗口
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # B, C, H//ws, ws, W//ws, ws
    x = x.view(b, c, h, w)  # B, C, H, W

    return x


class Overlapping_Cross_Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入通道数
                 window_size,  # 窗口大小
                 overlap_ratio,  # 重叠比例
                 num_heads=1,  # 注意力头数
                 qkv_bias=False,
                 qk_scale=None,
                 ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # QKV projections for both branches
        self.qkv_A = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.qkv_B = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)

        # Output projections
        self.proj_A = nn.Conv2d(dim, dim, 1)
        self.proj_B = nn.Conv2d(dim, dim, 1)

        # Unfold operation for overlapping windows
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2
        )

    def forward(self, x_A, x_B):
        b, c, h, w = x_A.shape

        # Save original inputs for residual connection
        shortcut_A = x_A
        shortcut_B = x_B

        # QKV projections
        qkv_A = self.qkv_A(x_A)  # B, 3*C, H, W
        qkv_B = self.qkv_B(x_B)  # B, 3*C, H, W

        # Split qkv
        q_A, k_A, v_A = qkv_A.chunk(3, dim=1)  # Each is B, C, H, W
        q_B, k_B, v_B = qkv_B.chunk(3, dim=1)  # Each is B, C, H, W

        # Reshape for multi-head attention
        head_dim = c // self.num_heads
        q_A = q_A.view(b, self.num_heads, head_dim, h, w)
        k_A = k_A.view(b, self.num_heads, head_dim, h, w)
        v_A = v_A.view(b, self.num_heads, head_dim, h, w)

        q_B = q_B.view(b, self.num_heads, head_dim, h, w)
        k_B = k_B.view(b, self.num_heads, head_dim, h, w)
        v_B = v_B.view(b, self.num_heads, head_dim, h, w)

        # Reshape for window partition
        q_A = q_A.permute(0, 1, 3, 4, 2).contiguous()  # B, num_heads, H, W, head_dim
        k_A = k_A.permute(0, 1, 3, 4, 2).contiguous()
        v_A = v_A.permute(0, 1, 3, 4, 2).contiguous()

        q_B = q_B.permute(0, 1, 3, 4, 2).contiguous()
        k_B = k_B.permute(0, 1, 3, 4, 2).contiguous()
        v_B = v_B.permute(0, 1, 3, 4, 2).contiguous()

        # Combine B and num_heads
        q_A = q_A.view(b * self.num_heads, h, w, head_dim)
        k_A = k_A.view(b * self.num_heads, h, w, head_dim)
        v_A = v_A.view(b * self.num_heads, h, w, head_dim)

        q_B = q_B.view(b * self.num_heads, h, w, head_dim)
        k_B = k_B.view(b * self.num_heads, h, w, head_dim)
        v_B = v_B.view(b * self.num_heads, h, w, head_dim)

        # Window partition
        q_A_windows = window_partition(q_A,
                                       self.window_size)  # (num_windows*B*num_heads, window_size, window_size, head_dim)
        k_A_windows = window_partition(k_A, self.window_size)
        v_A_windows = window_partition(v_A, self.window_size)

        q_B_windows = window_partition(q_B, self.window_size)
        k_B_windows = window_partition(k_B, self.window_size)
        v_B_windows = window_partition(v_B, self.window_size)

        # Reshape for attention computation
        num_windows = (h // self.window_size) * (w // self.window_size)
        q_A_windows = q_A_windows.view(b * num_windows * self.num_heads, self.window_size * self.window_size, head_dim)
        k_A_windows = k_A_windows.view(b * num_windows * self.num_heads, self.window_size * self.window_size, head_dim)
        v_A_windows = v_A_windows.view(b * num_windows * self.num_heads, self.window_size * self.window_size, head_dim)

        q_B_windows = q_B_windows.view(b * num_windows * self.num_heads, self.window_size * self.window_size, head_dim)
        k_B_windows = k_B_windows.view(b * num_windows * self.num_heads, self.window_size * self.window_size, head_dim)
        v_B_windows = v_B_windows.view(b * num_windows * self.num_heads, self.window_size * self.window_size, head_dim)

        # Cross Attention A->B
        attn_A = (q_B_windows @ k_A_windows.transpose(-2, -1)) * self.scale
        attn_A = F.softmax(attn_A, dim=-1)
        out_A = attn_A @ v_A_windows

        # Cross Attention B->A
        attn_B = (q_A_windows @ k_B_windows.transpose(-2, -1)) * self.scale
        attn_B = F.softmax(attn_B, dim=-1)
        out_B = attn_B @ v_B_windows

        # Reshape back
        out_A = out_A.view(-1, self.window_size, self.window_size, head_dim)
        out_B = out_B.view(-1, self.window_size, self.window_size, head_dim)

        # Merge windows
        out_A = window_reverse(out_A, self.window_size, h, w)  # B*num_heads, head_dim, H, W
        out_B = window_reverse(out_B, self.window_size, h, w)

        # Reshape back to original format
        out_A = out_A.view(b, self.num_heads * head_dim, h, w)
        out_B = out_B.view(b, self.num_heads * head_dim, h, w)

        # Final projection
        out_A = self.proj_A(out_A)
        out_B = self.proj_B(out_B)

        # Add residual connection
        out_A = out_A + shortcut_A
        out_B = out_B + shortcut_B

        return out_A, out_B
class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)


    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B


class Fusion_map(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_conv1 = nn.Conv2d(1024,512,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(512)
        self.fusion_conv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.fusion_conv3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self,f1,f2,is_pansharp=False):
        x = torch.cat([f1,f2], dim=1)
        x = self.fusion_conv1(x)
        x = self.bn1(x)
        x = self.fusion_conv2(x)
        x = self.bn2(x)
        x = self.fusion_conv3(x)
        x = self.bn3(x)
        if not is_pansharp:
            x = torch.flatten(x, start_dim=1)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        result=x+y

        result=self.relu(result)

        return result


import torch
import torch.nn as nn
import pywt

class DWT_ms(nn.Module):
    def __init__(self, wavelet='haar'):
        """
        初始化类，指定小波类型。
        :param wavelet: 小波类型，默认为 'haar'。
        """
        super(DWT_ms, self).__init__()
        self.wavelet = wavelet

    def forward(self, image_tensor):
        """
        在前向传播中执行小波变换，将输入的四通道图像进行小波分解为低频和高频部分。
        :param image_tensor: 输入的批次四通道图像张量，形状为 (batch_size, 4, H, W)。
        :return: 低频部分 (batch_size, 4, H/2, W/2)，高频部分 (batch_size, 12, H/2, W/2)。
        """
        return self.dwt_batch(image_tensor)

    def dwt_batch(self, image_tensor):
        """
        对批次四通道图像进行离散小波变换（DWT），并将低频和高频部分分开。
        :param image_tensor: 输入的批次四通道图像张量，形状为 (batch_size, 4, H, W)。
        :return: 低频部分 (batch_size, 4, H/2, W/2)，高频部分 (batch_size, 12, H/2, W/2)。
        """
        batch_size, channels, height, width = image_tensor.shape
        assert channels == 4, "输入图像必须为四通道图像（RGBA）"

        low_freq_batch = []
        high_freq_batch = []

        # 对每个批次的图像进行小波变换
        for b in range(batch_size):
            low_freq_channels = []
            high_freq_channels_LH = []
            high_freq_channels_HL = []
            high_freq_channels_HH = []

            # 对每个通道分别进行小波变换
            for c in range(channels):  # 4 表示 RGBA 四个通道
                coeffs2 = pywt.dwt2(image_tensor[b, c].cpu().numpy(), self.wavelet)
                LL, (LH, HL, HH) = coeffs2

                # 将 numpy 数组转换为 PyTorch 张量并放到相同设备上
                low_freq_channels.append(torch.tensor(LL, device=image_tensor.device))  # 低频成分
                high_freq_channels_LH.append(torch.tensor(LH, device=image_tensor.device))  # 高频成分
                high_freq_channels_HL.append(torch.tensor(HL, device=image_tensor.device))  # 高频成分
                high_freq_channels_HH.append(torch.tensor(HH, device=image_tensor.device))  # 高频成分

            # 拼接四个通道的低频部分 (4, H/2, W/2)
            low_freq = torch.stack(low_freq_channels, dim=0)  # (4, H/2, W/2)
            low_freq_batch.append(low_freq)

            # 拼接高频部分 (12, H/2, W/2)
            high_freq = torch.cat(
                [torch.stack(high_freq_channels_LH, dim=0),
                 torch.stack(high_freq_channels_HL, dim=0),
                 torch.stack(high_freq_channels_HH, dim=0)],
                dim=0
            )  # (12, H/2, W/2)
            high_freq_batch.append(high_freq)

        # 将低频和高频部分转换为批次形式
        low_freq_batch = torch.stack(low_freq_batch).to(image_tensor.device)  # (batch_size, 4, H/2, W/2)
        high_freq_batch = torch.stack(high_freq_batch).to(image_tensor.device)  # (batch_size, 12, H/2, W/2)

        return low_freq_batch, high_freq_batch

    def idwt_batch(self, low_freq_batch, high_freq_batch):
        """
        根据低频和高频部分重构四通道图像。
        :param low_freq_batch: 低频部分张量，形状为 (batch_size, 4, H/2, W/2)。
        :param high_freq_batch: 高频部分张量，形状为 (batch_size, 12, H/2, W/2)。
        :return: 重构后的四通道图像张量，形状为 (batch_size, 4, H, W)。
        """
        batch_size = low_freq_batch.shape[0]
        reconstructed_batch = []

        for b in range(batch_size):
            # 提取低频部分 (4, H/2, W/2)
            LL_tensors = [low_freq_batch[b, c] for c in range(4)]  # 四个通道的低频部分

            # 提取高频部分 (12, H/2, W/2)
            LH_tensors = [high_freq_batch[b, c] for c in range(4)]  # 四个通道的LH
            HL_tensors = [high_freq_batch[b, c + 4] for c in range(4)]  # 四个通道的HL
            HH_tensors = [high_freq_batch[b, c + 8] for c in range(4)]  # 四个通道的HH

            # 对每个通道进行逆小波变换
            reconstructed_img = []
            for c in range(4):  # RGBA 四个通道
                coeffs2 = (LL_tensors[c].cpu().numpy(), (LH_tensors[c].cpu().numpy(),
                                                          HL_tensors[c].cpu().numpy(),
                                                          HH_tensors[c].cpu().numpy()))
                reconstructed = pywt.idwt2(coeffs2, self.wavelet)
                reconstructed_img.append(torch.tensor(reconstructed, device=low_freq_batch.device))

            # 将重建的通道拼接为图像
            reconstructed_img = torch.stack(reconstructed_img, dim=0)  # (4, H, W)
            reconstructed_batch.append(reconstructed_img)

        # 返回批次化的重构图像
        reconstructed_batch = torch.stack(reconstructed_batch).to(low_freq_batch.device)  # (batch_size, 4, H, W)

        return reconstructed_batch


class DWT_pan(nn.Module):
    def __init__(self, wavelet='haar'):
        """
        初始化类，指定小波类型。
        :param wavelet: 小波类型，默认为 'haar'。
        """
        super(DWT_pan, self).__init__()
        self.wavelet = wavelet

    def forward(self, image_tensor):
        """
        在前向传播中执行小波变换，将输入的单通道图像进行小波分解为低频和高频部分。
        :param image_tensor: 输入的批次单通道图像张量，形状为 (batch_size, 1, H, W)。
        :return: 低频部分 (batch_size, 1, H/2, W/2)，高频部分 (batch_size, 3, H/2, W/2)。
        """
        return self.dwt_batch(image_tensor)

    def dwt_batch(self, image_tensor):
        """
        对批次单通道图像进行离散小波变换（DWT），并将低频和高频部分分开。
        :param image_tensor: 输入的批次单通道图像张量，形状为 (batch_size, 1, H, W)。
        :return: 低频部分 (batch_size, 1, H/2, W/2)，高频部分 (batch_size, 3, H/2, W/2)。
        """
        batch_size, channels, height, width = image_tensor.shape
        assert channels == 1, "输入图像必须为单通道图像"

        low_freq_batch = []
        high_freq_batch = []

        # 对每个批次的图像进行小波变换
        for b in range(batch_size):
            # 使用 squeeze 将单通道图像转换为二维张量
            image_2d = image_tensor[b, 0].cpu().numpy()  # 转为 NumPy 数组

            # 执行小波变换
            coeffs2 = pywt.dwt2(image_2d, self.wavelet)
            LL, (LH, HL, HH) = coeffs2

            # 将结果转换为 PyTorch 张量，并保持在原来的设备上
            low_freq_batch.append(torch.tensor(LL, device=image_tensor.device))  # 低频成分
            high_freq_batch.append(torch.stack([
                torch.tensor(LH, device=image_tensor.device),  # 高频成分
                torch.tensor(HL, device=image_tensor.device),  # 高频成分
                torch.tensor(HH, device=image_tensor.device)   # 高频成分
            ], dim=0))  # (3, H/2, W/2)

        # 将低频和高频部分转换为批次形式
        low_freq_batch = torch.stack(low_freq_batch).unsqueeze(1)  # (batch_size, 1, H/2, W/2)
        high_freq_batch = torch.stack(high_freq_batch)  # (batch_size, 3, H/2, W/2)

        return low_freq_batch, high_freq_batch

    def idwt_batch(self, low_freq_batch, high_freq_batch):
        """
        根据低频和高频部分重构单通道图像。
        :param low_freq_batch: 低频部分张量，形状为 (batch_size, 1, H/2, W/2)。
        :param high_freq_batch: 高频部分张量，形状为 (batch_size, 3, H/2, W/2)。
        :return: 重构后的单通道图像张量，形状为 (batch_size, 1, H, W)。
        """
        batch_size = low_freq_batch.shape[0]
        reconstructed_batch = []

        for b in range(batch_size):
            # 提取低频部分
            LL_tensor = low_freq_batch[b, 0]  # (H/2, W/2)

            # 提取高频部分
            LH_tensor = high_freq_batch[b, 0]  # (H/2, W/2)
            HL_tensor = high_freq_batch[b, 1]  # (H/2, W/2)
            HH_tensor = high_freq_batch[b, 2]  # (H/2, W/2)

            # 合并小波系数
            coeffs2 = (LL_tensor.cpu().numpy(), (LH_tensor.cpu().numpy(), HL_tensor.cpu().numpy(), HH_tensor.cpu().numpy()))

            # 对图像进行逆小波变换
            reconstructed = pywt.idwt2(coeffs2, self.wavelet)

            # 将结果转换为 PyTorch 张量，并保持在原来的设备上
            reconstructed_batch.append(torch.tensor(reconstructed, device=low_freq_batch.device))

        # 将重建的图像转换为批次形式
        reconstructed_batch = torch.stack(reconstructed_batch).unsqueeze(1)  # (batch_size, 1, H, W)

        return reconstructed_batch
# class DWT_pan(nn.Module):
#     def __init__(self, wavelet):
#         """
#         初始化类，指定小波类型。
#         :param wavelet: 小波类型，默认为 'haar'。
#         """
#         super(DWT_pan, self).__init__()
#         self.wavelet = wavelet
#
#     def forward(self, image_tensor):
#         """
#         在前向传播中执行小波变换，将输入的单通道图像进行小波分解为低频和高频部分。
#         :param image_tensor: 输入的批次单通道图像张量，形状为 (batch_size, 1, H, W)。
#         :return: 低频部分 (batch_size, 1, H/2, W/2)，高频部分 (batch_size, 3, H/2, W/2)。
#         """
#         with torch.no_grad():
#             return self.dwt_batch(image_tensor)
#         # return self.dwt_batch(image_tensor)
#
#     def dwt_batch(self, image_tensor):
#         """
#         对批次单通道图像进行离散小波变换（DWT），并将低频和高频部分分开。
#         :param image_tensor: 输入的批次单通道图像张量，形状为 (batch_size, 1, H, W)。
#         :return: 低频部分 (batch_size, 1, H/2, W/2)，高频部分 (batch_size, 3, H/2, W/2)。
#         """
#         batch_size, channels, height, width = image_tensor.shape
#         assert channels == 1, "输入图像必须为单通道图像"
#
#         # 将 PyTorch Tensor 转换为 NumPy 数组
#         image_np = image_tensor.squeeze(1).cpu().numpy()  # 形状变为 (batch_size, H, W)
#
#         # 创建空列表以存储每个图像的小波变换结果
#         low_freq_batch = []
#         high_freq_batch = []
#
#         # 对每个批次的图像进行小波变换
#         for b in range(batch_size):
#             # 对单通道图像进行小波变换
#             coeffs2 = pywt.dwt2(image_np[b], self.wavelet)
#             LL, (LH, HL, HH) = coeffs2
#
#             # 转换回 PyTorch 张量
#             LL_tensor = torch.tensor(LL)  # 低频成分
#             LH_tensor = torch.tensor(LH)  # 高频成分
#             HL_tensor = torch.tensor(HL)  # 高频成分
#             HH_tensor = torch.tensor(HH)  # 高频成分
#
#             # 存储低频和高频部分
#             low_freq_batch.append(LL_tensor)  # (H/2, W/2)
#             high_freq_batch.append(torch.stack([LH_tensor, HL_tensor, HH_tensor], dim=0))  # (3, H/2, W/2)
#
#         # 将低频和高频部分转换为批次形式
#         low_freq_batch = torch.stack(low_freq_batch).unsqueeze(1)  # (batch_size, 1, H/2, W/2)
#         high_freq_batch = torch.stack(high_freq_batch)  # (batch_size, 3, H/2, W/2)
#
#         return low_freq_batch, high_freq_batch
#
#     def idwt_batch(self, low_freq_batch, high_freq_batch):
#         """
#         根据低频和高频部分重构单通道图像。
#         :param low_freq_batch: 低频部分张量，形状为 (batch_size, 1, H/2, W/2)。
#         :param high_freq_batch: 高频部分张量，形状为 (batch_size, 3, H/2, W/2)。
#         :return: 重构后的单通道图像张量，形状为 (batch_size, 1, H, W)。
#         """
#         batch_size = low_freq_batch.shape[0]
#         reconstructed_batch = []
#
#         for b in range(batch_size):
#             # 提取低频部分
#             LL_tensor = low_freq_batch[b, 0]  # (H/2, W/2)
#
#             # 提取高频部分
#             LH_tensor = high_freq_batch[b, 0]  # (H/2, W/2)
#             HL_tensor = high_freq_batch[b, 1]  # (H/2, W/2)
#             HH_tensor = high_freq_batch[b, 2]  # (H/2, W/2)
#
#             # 将小波系数合并
#             coeffs2 = (LL_tensor.numpy(), (LH_tensor.numpy(), HL_tensor.numpy(), HH_tensor.numpy()))
#
#             # 对图像进行逆小波变换
#             reconstructed = pywt.idwt2(coeffs2, self.wavelet)
#
#             # 将结果转换为 PyTorch 张量
#             reconstructed_batch.append(torch.tensor(reconstructed))
#
#         # 将重建的图像转换为批次形式
#         reconstructed_batch = torch.stack(reconstructed_batch).unsqueeze(1)  # (batch_size, 1, H, W)
#
#         return reconstructed_batch

# model=Model()
# print(model)
# ms=torch.rand((128, 4, 16, 16))
# pan=torch.rand((128,1,64,64))
# out=model(ms,pan)
# print(out.shape)
# print(out)