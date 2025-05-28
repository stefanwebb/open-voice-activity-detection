"""
© 2025, Stefan Webb. Some Rights Reserved.

Except where otherwise noted, this work is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-sa/4.0/deed.en

"""

import torch

class SileroVADBlock(torch.nn.module):
    def __init__(self):
        # TODO: Revise 1D convolution
        self.reparam_conv = torch.nn.Conv1d(
            in_channels=129,
            out_channels=128,
            kernel_size=(3,),
            stride=1,
            padding=1,
            dilation=1)
        
        self.activation = torch.nn.ReLU()

        # TODO: Figure out why this is in original model
        self.se = torch.nn.Identity()

    def forward(self, input: torch.Tensor):
        return self.activation(self.se(self.reparam_conv(input)))
