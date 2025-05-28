"""
© 2025, Stefan Webb. Some Rights Reserved.

Except where otherwise noted, this work is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-sa/4.0/deed.en

"""
import torch
import torch.nn as nn

class STFT(nn.Module):

    def __init__(self, ):
        super(STFT, self).__init__()
        self.filter_length = 256
        self.padding = nn.ReflectionPad1d((0, 64))
        self.forward_basis_buffer = nn.Conv1d(in_channels=1, out_channels=258, kernel_size=256, stride=128, padding=0,
                                              bias=False)

    def transform_(self, input_data):
        x = self.padding(input_data).unsqueeze(1)
        x = self.forward_basis_buffer(x)
        cutoff = int(self.filter_length / 2 + 1)
        real_part = x[:, :cutoff, :6]
        imag_part = x[:, cutoff:, :6]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        return magnitude

    def forward(self, x):
        return self.transform_(x)
