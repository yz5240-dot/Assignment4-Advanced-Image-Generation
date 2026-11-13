import math

import torch
from torch import nn
from torch.nn import functional as F

TIMESTEPS = 1000
IMAGE_SHAPE = (3, 32, 32)


def timestep_embedding(t, dim=32, max_period=10000):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    angles = t.float()[:, None] * frequencies[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=1)


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(4, out_channels)
        self.time = nn.Linear(32, out_channels)

    def forward(self, x, time):
        return F.silu(self.norm(self.conv(x)) + self.time(time)[:, :, None, None])


class DiffusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))
        self.encoder1 = TimeBlock(3, 24)
        self.encoder2 = TimeBlock(24, 32)
        self.down1 = nn.Conv2d(24, 24, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.middle = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1), nn.SiLU())
        self.bottleneck = TimeBlock(32, 48)
        self.decoder2 = TimeBlock(48 + 32, 32)
        self.decoder1 = TimeBlock(32 + 24, 24)
        self.output = nn.Conv2d(24, 3, 3, padding=1)

    def forward(self, x, t):
        time = self.time_mlp(timestep_embedding(t))
        first = self.encoder1(x, time)
        second = self.encoder2(self.down1(first), time)
        x = self.middle(self.bottleneck(self.down2(second), time))
        x = self.decoder2(torch.cat((F.interpolate(x, scale_factor=2), second), 1), time)
        x = self.decoder1(torch.cat((F.interpolate(x, scale_factor=2), first), 1), time)
        return self.output(x)


def alpha_bar(device):
    return (1 - torch.linspace(1e-4, 0.02, TIMESTEPS, device=device)).cumprod(0)


def diffusion_loss(model, clean, schedule):
    t = torch.randint(TIMESTEPS, (len(clean),), device=clean.device)
    noise = torch.randn_like(clean)
    alpha = schedule[t, None, None, None]
    noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
    return F.mse_loss(model(noisy, t), noise)


@torch.inference_mode()
def sample_diffusion(model, count, steps, rng):
    # Deterministic DDIM updates conditional on the initial Gaussian noise.
    device = next(model.parameters()).device
    schedule = alpha_bar(device)
    timesteps = torch.linspace(TIMESTEPS - 1, 0, steps).round().long().tolist()
    x = torch.randn((count, *IMAGE_SHAPE), device=device, generator=rng)
    for index, t in enumerate(timesteps):
        alpha = schedule[t]
        epsilon = model(x, torch.full((count,), t, device=device, dtype=torch.long))
        clean = ((x - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1, 1)
        if index + 1 < steps:
            next_alpha = schedule[timesteps[index + 1]]
            epsilon = (x - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            x = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * epsilon
        else:
            x = clean
    return x
