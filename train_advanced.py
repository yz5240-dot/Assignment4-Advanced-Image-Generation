import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from helper_lib.diffusion import DiffusionUNet, alpha_bar, diffusion_loss, sample_diffusion
from helper_lib.energy import EnergyModel, energy_loss, langevin_sample


def train(args):
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(args.cpu_threads)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    if args.parquet:
        from io import BytesIO

        from PIL import Image
        from pyarrow.parquet import ParquetFile

        table = ParquetFile(args.parquet).read(columns=['img', 'label'])
        if args.limit and args.limit < len(table):
            indices = torch.randperm(len(table), generator=torch.Generator().manual_seed(42))
            table = table.take(indices[:args.limit].numpy())
        images = torch.stack([
            transforms.functional.pil_to_tensor(Image.open(BytesIO(row.as_py()['bytes'])).convert('RGB'))
            for row in table['img']
        ])
        dataset = TensorDataset(images, torch.tensor(table['label'].to_pylist()))
    else:
        dataset = datasets.CIFAR10(args.data, train=True, download=True, transform=transform)
    if args.limit and args.limit < len(dataset):
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))
        dataset = Subset(dataset, indices[:args.limit].tolist())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=device.type == 'cuda')
    is_diffusion = args.model == 'diffusion'
    model = (DiffusionUNet() if is_diffusion else EnergyModel()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001 if is_diffusion else 0.0003)
    schedule = alpha_bar(device)
    # Store recent negatives so sampling can continue from earlier low-energy states.
    replay = torch.rand((2048, 3, 32, 32), device=device) * 2 - 1 if not is_diffusion else None
    history = []
    first_epoch = 0
    checkpoint_path = out / f'{args.model}_resume.pt'
    if args.resume:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        first_epoch = state['epoch']
        history = state['history']
        replay = state['replay']
    print(f'{args.model}: {len(dataset)} CIFAR-10 images on {device}', flush=True)
    for epoch in range(first_epoch, args.epochs):
        total = 0.0
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            if images.dtype == torch.uint8:
                images = images.float() / 127.5 - 1
            optimizer.zero_grad(set_to_none=True)
            if is_diffusion:
                loss = diffusion_loss(model, images, schedule)
            else:
                ids = torch.randint(len(replay), (len(images),), device=device)
                fresh = torch.rand_like(images) * 2 - 1
                restart = torch.rand((len(images), 1, 1, 1), device=device) < 0.10
                start = torch.where(restart, fresh, replay[ids])
                negative = langevin_sample(model, start, args.langevin_steps)
                replay[ids] = negative
                loss = energy_loss(model, images, negative)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(images)
        mean_loss = total / len(dataset)
        history.append({'epoch': epoch + 1, 'loss': mean_loss})
        print(f'Epoch {epoch + 1}/{args.epochs}: {mean_loss:.5f}', flush=True)
        torch.save(model.state_dict(), out / f'{args.model}.pt')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1, 'history': history, 'replay': replay}, checkpoint_path)
        (out / f'{args.model}_history.json').write_text(json.dumps(
            {'dataset': 'CIFAR-10 train', 'training_images': len(dataset), 'epochs': history}, indent=2))
    model.eval()
    rng = torch.Generator(device=device).manual_seed(42)
    if is_diffusion:
        samples = sample_diffusion(model, 16, 50, rng)
    else:
        initial = torch.rand((16, 3, 32, 32), device=device, generator=rng) * 2 - 1
        samples = langevin_sample(model.requires_grad_(False), initial, 100, rng)
    save_image((samples + 1) / 2, out / f'{args.model}_samples.png', nrow=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=('diffusion', 'ebm'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--langevin-steps', type=int, default=16)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--data', default='data')
    parser.add_argument('--parquet', help='Optional local CIFAR-10 training parquet mirror')
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--limit', type=int, default=0, help='Use a reproducible training subset')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.langevin_steps, args.cpu_threads) < 1 or args.limit < 0:
        parser.error('epochs, batch size, steps, and threads must be positive; limit must be nonnegative')
    train(args)
