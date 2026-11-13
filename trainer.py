import csv
import json
from pathlib import Path

import torch
from torchvision.utils import save_image


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=20,
              output_dir='outputs', resume=False):
    """epochs is the total target epoch count; optimizer contains G and D."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / 'training_checkpoint.pt'
    model.to(device)
    opt_g, opt_d = optimizer['G'], optimizer['D']
    history = []
    start_epoch = 0
    fixed_noise = torch.randn(64, 100, device=device)
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'No checkpoint at {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        opt_g.load_state_dict(checkpoint['optimizer_G'])
        opt_d.load_state_dict(checkpoint['optimizer_D'])
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        fixed_noise = checkpoint['fixed_noise'].to(device)
    for epoch in range(start_epoch, epochs):
        model.train()
        sum_g = sum_d = 0.0
        count = 0
        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train D on real images and detached generated images.
            opt_d.zero_grad(set_to_none=True)
            fake_images = model.generator(torch.randn(batch_size, 100, device=device))
            loss_d = (criterion(model.discriminator(real_images), real_labels)
                      + criterion(model.discriminator(fake_images.detach()), fake_labels)) / 2
            loss_d.backward()
            opt_d.step()

            # Freeze D's parameters, but keep BatchNorm in training mode.
            # Both adversarial updates must use the same batch-statistics rule.
            model.discriminator.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            loss_g = criterion(model.discriminator(fake_images), real_labels)
            loss_g.backward()
            opt_g.step()
            model.discriminator.requires_grad_(True)
            sum_g += loss_g.item() * batch_size
            sum_d += loss_d.item() * batch_size
            count += batch_size
        if count == 0:
            raise ValueError('The data loader is empty.')
        row = {'epoch': epoch + 1, 'generator_loss': sum_g / count,
               'discriminator_loss': sum_d / count}
        history.append(row)
        print(f"Epoch {epoch+1}/{epochs} | G: {row['generator_loss']:.4f} | D: {row['discriminator_loss']:.4f}")
        model.generator.eval()
        with torch.inference_mode():
            samples = (model.generator(fixed_noise) + 1) / 2
        save_image(samples, output / f'epoch_{epoch+1:03d}.png', nrow=8)
        torch.save(model.generator.state_dict(), output / 'generator.pt')
        torch.save({'epoch': epoch + 1, 'model': model.state_dict(),
                    'optimizer_G': opt_g.state_dict(), 'optimizer_D': opt_d.state_dict(),
                    'history': history, 'fixed_noise': fixed_noise}, checkpoint_path)
        with (output / 'loss_history.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        (output / 'training_summary.json').write_text(json.dumps({
            'dataset': 'MNIST training split', 'epochs_completed': epoch + 1,
            'device': str(device), 'last_losses': row,
            'note': 'Losses alone do not measure image quality; inspect the sample grids.'
        }, indent=2))
    model.history = history
    return model

