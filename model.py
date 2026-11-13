import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 7 * 7 * 128)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, noise):
        x = self.fc(noise).reshape(-1, 128, 7, 7)
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        return self.layers(images)


class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, noise):
        return self.generator(noise)


def get_model(model_name):
    from .diffusion import DiffusionUNet
    from .energy import EnergyModel

    models = {'GAN': GAN, 'DIFFUSION': DiffusionUNet, 'EBM': EnergyModel}
    try:
        model = models[model_name.upper()]
    except KeyError as error:
        raise ValueError(f'Unknown model: {model_name}') from error
    return model()
