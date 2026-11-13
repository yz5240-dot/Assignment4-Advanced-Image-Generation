import torch
from torch import nn


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, images):
        return self.features(images).squeeze(1)


def langevin_sample(model, initial, steps, rng=None, step_size=8.0, noise_std=0.025):
    x = initial.detach()
    for _ in range(steps):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            (gradient,) = torch.autograd.grad(model(x).sum(), x)
        with torch.no_grad():
            noise = torch.randn(x.shape, device=x.device, generator=rng)
            x = (x - step_size * gradient + noise_std * noise).clamp(-1, 1)
    return x


def energy_loss(model, real, negative):
    positive_energy = model(real)
    negative_energy = model(negative.detach())
    regularizer = (positive_energy.square() + negative_energy.square()).mean()
    return (positive_energy - negative_energy).mean() + 0.1 * regularizer
