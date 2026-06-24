# 这个文件需要包含两个核心类：一个是负责展平图像并调用 Mamba 的 VSSBlock，
# 另一个是兼容 YOLO 结构的 C2f_Mamba

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class VSSBlock(nn.Module):
    """Visual State Space Block (Mamba 核心封装) 作用：将 2D 图像特征图展平为 1D 序列，送入 Mamba 进行全局序列建模，再还原为 2D。.
    """

    def __init__(self, c):
        super().__init__()
        # 这里的 mamba_ssm 是我们在云端配环境时安装的底层依赖
        try:
            from mamba_ssm import Mamba
        except ImportError:
            raise ImportError("Please install mamba_ssm and causal-conv1d first.")

        # d_model 为输入通道数，d_state 为状态空间维度(通常设为16)
        self.mamba = Mamba(d_model=c, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        # x 的原始形状: (Batch, Channels, Height, Width)
        b, c, h, w = x.shape

        # 1. 展平并转置: (B, C, H*W) -> (B, H*W, C) 以适应 Mamba 的输入格式
        x_flat = x.view(b, c, -1).transpose(1, 2)

        # 2. 归一化后过 Mamba 层
        x_norm = self.norm(x_flat)
        y = self.mamba(x_norm)

        # 3. 转置并还原回 2D 图像形状: (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        y = y.transpose(1, 2).view(b, c, h, w)

        # 4. 残差连接
        return y + x


class C2f_Mamba(nn.Module):
    """魔改版 C2f: 将内部的 Bottleneck 替换为 Mamba VSSBlock 保持了 C2f 的跨层特征拼接逻辑(CSP结构)，确保梯度流的稳定性.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏层通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 1x1 卷积调整通道并一分为二
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 最终特征融合

        # 核心替换：使用 VSSBlock 替代原来的 Bottleneck
        self.m = nn.ModuleList(VSSBlock(self.c) for _ in range(n))

    def forward(self, x):
        # 经过 cv1 后，将特征图在通道维度切分为两半
        y = list(self.cv1(x).chunk(2, 1))
        # 依次通过 Mamba 模块，并将每一步的输出收集起来
        y.extend(m(y[-1]) for m in self.m)
        # 将所有分支的特征在通道维度级联，最后过一个 1x1 卷积
        return self.cv2(torch.cat(y, 1))
