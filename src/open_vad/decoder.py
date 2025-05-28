"""
© 2025, Stefan Webb. Some Rights Reserved.

Except where otherwise noted, this work is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-sa/4.0/deed.en

"""
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTMCell(128, 128)
        self.decoder = nn.Sequential(nn.Dropout(0.1),
                                     nn.ReLU(),
                                     nn.Conv1d(128, 1, kernel_size=1, padding=0, dilation=1),
                                     nn.Sigmoid())

    def forward(self, x, state=torch.zeros(0)):
        x = x.squeeze(-1)
        if len(state):
            h, c = self.rnn(x, (state[0], state[1]))
        else:
            h, c = self.rnn(x)

        x = h.unsqueeze(-1).float()
        state = torch.stack([h, c])
        x = self.decoder(x)
        return x, state