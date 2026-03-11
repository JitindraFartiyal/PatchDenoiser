import time

import torch

from torch import nn
import torch.nn.functional as F
from utilities import get_model_stats

class L32PatchEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, latent_dim=32):
        super(L32PatchEncoder, self).__init__()

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=latent_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(2, latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.init_layer(x)
        x = F.gelu(x + self.block1(x))
        x = F.gelu(x + self.block2(x))
        x = F.gelu(x + self.block3(x))
        x = F.gelu(x + self.block4(x))
        x = F.gelu(x + self.block5(x))
        x = F.gelu(x + self.block6(x))
        x = F.gelu(x + self.block7(x))
        x = F.gelu(x + self.block8(x))
        x = self.last_layer(x)

        return x

class L128PatchEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, latent_dim=64):
        super(L128PatchEncoder, self).__init__()
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(2, latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.init_layer(x)
        x = F.gelu(x + self.block1(x))
        x = F.gelu(x + self.block2(x))
        x = F.gelu(x + self.block3(x))
        x = F.gelu(x + self.block4(x))
        x = F.gelu(x + self.block5(x))
        x = F.gelu(x + self.block6(x))
        x = F.gelu(x + self.block7(x))
        x = F.gelu(x + self.block8(x))
        x = self.last_layer(x)

        return x

class L512PatchEncoder(nn.Module):

    def __init__(self, in_channels=1, out_channels=96, latent_dim=96):

        super(L512PatchEncoder, self).__init__()
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=11, stride=2, padding=5, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels * 2),
            nn.GELU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(2, latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.init_layer(x)
        x = F.gelu(x + self.block1(x))
        x = F.gelu(x + self.block2(x))
        x = F.gelu(x + self.block3(x))
        x = F.gelu(x + self.block4(x))
        x = self.last_layer(x)
        # x = x * self.channel_attention(x)

        return x


if __name__ == '__main__':
    input32 = torch.randn(1 * 256, 1, 32, 32)
    input64 = torch.randn(1 * 64, 1, 64, 64)
    input128 = torch.randn(1 * 26, 1, 128, 128)
    input256 = torch.randn(1 * 2, 1, 256, 256)
    input512 = torch.randn(1, 1, 512, 512)

    l32_pe = L32PatchEncoder()
    l128_pe = L128PatchEncoder()
    l512_pe = L512PatchEncoder()

    start_time = time.time()
    embed32 = l32_pe(input32)
    embed128 = l128_pe(input128)
    embed512 = l512_pe(input512)

    print(embed32.shape, embed128.shape,  embed512.shape, time.time() - start_time)

    get_model_stats(l32_pe, input32)
    get_model_stats(l128_pe, input128)
    get_model_stats(l512_pe, input512)
