import time

import torch
from torch import nn
from utilities import get_model_stats

class Decoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=1):
        super().__init__()

        oc = 24
        in_ch = 6
        self.consolidate_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=oc, kernel_size=1, bias=False),
            nn.GroupNorm(2, oc),
            nn.GELU(),
        )

        self.upsample_layer = nn.PixelShuffle(upscale_factor=2)

        self.last_layer = nn.Conv2d(in_channels=in_ch, out_channels=out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.consolidate_layer(x)
        x = self.upsample_layer(x)
        x = self.last_layer(x)
        x = self.sigmoid(x)

        return x



if __name__ == '__main__':
    input = torch.randn(1, 96, 256, 256)
    decoder = Decoder(in_channels=96)
    start_time = time.time()
    output = decoder(input)
    print(output.size(), time.time() - start_time)

    get_model_stats(decoder, dummy_input=input)