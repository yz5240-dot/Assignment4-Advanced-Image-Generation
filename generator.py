import math
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


def generate_samples(model, device='cpu', num_samples=10, seed=None,
                     save_path=None, show=True):
    """Return generated images in [0, 1]; optionally save and display a grid."""
    if not 1 <= num_samples <= 100:
        raise ValueError('num_samples must be between 1 and 100.')
    generator = model.generator if hasattr(model, 'generator') else model
    generator.to(device)
    was_training = generator.training
    generator.eval()
    # A local CPU RNG makes a supplied seed repeatable without changing global RNG.
    rng = torch.Generator(device='cpu')
    if seed is None:
        rng.seed()
    else:
        rng.manual_seed(seed)
    noise = torch.randn(num_samples, 100, generator=rng).to(device)
    try:
        with torch.inference_mode():
            images = ((generator(noise).cpu() + 1) / 2).clamp(0, 1)
    finally:
        generator.train(was_training)
    columns = min(10, math.ceil(math.sqrt(num_samples)))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        save_image(images, save_path, nrow=columns)
    if show:
        import matplotlib.pyplot as plt
        grid = make_grid(images, nrow=columns)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title('Generated MNIST samples')
        plt.show()
    return images

