"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, BatchNorm2d, ReLU, Sequential


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.padding = (kernel_size - 1) // 2
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_channels * kernel_size * kernel_size,
                                                     fan_out=out_channels * kernel_size * kernel_size,
                                                     shape=(kernel_size, kernel_size, in_channels, out_channels),
                                                     dtype=dtype, device=device))
        if bias:
            bound = 1 / (in_channels * kernel_size ** 2) ** 0.5
            self.bias = Parameter(init.rand(self.out_channels, low = -bound, high = bound, device = device))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        _x = x.transpose((1, 2)).transpose((2, 3))
        out = ops.conv(_x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            out += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(out.shape)
        return out.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION


class ConvBN(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, device=None):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, device=device)
        self.bn = BatchNorm2d(out_channels, device=device)
        self.relu = ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))