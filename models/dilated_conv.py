import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import math

class SamePadConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super(SamePadConv, self).__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1D(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-math.sqrt(groups/ float(in_channels*kernel_size)), math.sqrt(groups/ float(in_channels*kernel_size)))),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-math.sqrt(groups/ float(in_channels*kernel_size)), math.sqrt(groups/ float(in_channels*kernel_size)))),
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super(ConvBlock, self).__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1D(
            in_channels,
            out_channels,
            1,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-math.sqrt(1./(in_channels*1)), math.sqrt(1./(in_channels*1)))),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-math.sqrt(1./(in_channels*1)), math.sqrt(1./(in_channels*1))))
        ) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Layer):
    def __init__(self, in_channels, channels, kernel_size):
        super(DilatedConvEncoder, self).__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
