"""
© 2025, Stefan Webb. Some Rights Reserved.

Except where otherwise noted, this work is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-sa/4.0/deed.en

"""
import torch
import torch.nn as nn

from open_vad.encoder_block import EncoderBlock

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(EncoderBlock(in_channels=129, out_channels=128, kernel_size=3, stride=1, padding=1),
                                     EncoderBlock(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
                                     EncoderBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                                     EncoderBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        return self.encoder(x)
