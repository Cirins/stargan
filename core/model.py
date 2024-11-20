import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, num_classes=5, repeat_num=6):
        super(Generator, self).__init__()

        self.layers = nn.ModuleDict()

        self.layers['initial'] = nn.Sequential(
            nn.Conv1d(3+num_classes, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm1d(conv_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            self.layers[f'downsample_{i}'] = nn.Sequential(
                nn.Conv1d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm1d(curr_dim*2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            self.layers[f'bottleneck_{i}'] = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)

        # Up-sampling layers.
        for i in range(2):
            self.layers[f'upsample_{i}'] = nn.Sequential(
                nn.ConvTranspose1d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm1d(curr_dim//2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )
            curr_dim = curr_dim // 2

        self.layers['final'] = nn.Sequential(
            nn.Conv1d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

    def forward(self, x, c):
        # Replicate spatially and concatenate class information.
        c = c.view(c.size(0), c.size(1), 1)
        c = c.repeat(1, 1, x.size(2))
        x = torch.cat([x, c], dim=1)
        for layer in self.layers.values():
            x = layer(x)
        return x


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, num_timesteps=128, conv_dim=64, num_classes=5, num_domains=10, repeat_num=6):
        super(Discriminator, self).__init__()

        self.layers = nn.ModuleDict()

        self.layers['initial'] = nn.Sequential(
            nn.Conv1d(3, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01)
        )

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            self.layers[f'downsample_{i}'] = nn.Sequential(
                nn.Conv1d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.01)
            )
            curr_dim = curr_dim * 2

        kernel_size = int(num_timesteps / np.power(2, repeat_num))
        self.layers['main'] = nn.Sequential(*self.layers.values())
        self.layers['src'] = nn.Conv1d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers['cls'] = nn.Conv1d(curr_dim, num_classes, kernel_size=kernel_size, bias=False)
        self.layers['dom'] = nn.Conv1d(curr_dim, num_domains, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.layers['main'](x)
        out_src = self.layers['src'](h)
        out_cls = self.layers['cls'](h)
        out_dom = self.layers['dom'](h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1)), out_dom.view(out_dom.size(0), out_dom.size(1))
