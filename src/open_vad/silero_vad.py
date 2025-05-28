"""
© 2025, Stefan Webb. Some Rights Reserved.

Except where otherwise noted, this work is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-sa/4.0/deed.en

"""
import torch
import torch.nn as nn

from open_vad.encoder_block import Decoder, Encoder, STFT

class SileroVAD(nn.Module):
    def __init__(self):
        super(SileroVAD, self).__init__()
        self.stft = STFT()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reset_states(self, batch_size=1):
        self._state = torch.zeros([0])
        self._context = torch.zeros([0])
        self._last_sr = 0
        self._last_batch_size = 0

    def forward(self, data, sr):

        num_samples = 512 if sr == 16000 else 256

        if data.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {data.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)")

        batch_size = data.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = torch.zeros(batch_size, context_size)

        x = torch.cat([self._context, data], dim=1)

        x = self.stft(x)
        x = self.encoder(x)
        x, self._state = self.decoder(x, self._state)


        self._context = data[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        return x