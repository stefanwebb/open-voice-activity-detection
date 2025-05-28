"""
© 2025, Stefan Webb. Some Rights Reserved.

Except where otherwise noted, this work is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-sa/4.0/deed.en

"""
import torch
import torch.nn as nn

class EncoderBlock(nn.module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(EncoderBlock, self).__init__()

        # TODO: Revise 1D convolution
        self.reparam_conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        
        self.act_fn = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.reparam_conv(x)
        return self.act_fn(x)
