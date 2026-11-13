import io
import os
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from torchvision.utils import save_image

from helper_lib.model import Generator, get_model
from helper_lib.diffusion import IMAGE_SHAPE, sample_diffusion
from helper_lib.energy import langevin_sample
from helper_lib.generator import generate_samples

app = FastAPI(title='Advanced Image Generation API', version='2.0')
BASE_DIR = Path(__file__).resolve().parent


def model_path():
    return Path(os.environ.get('GAN_MODEL_PATH', str(BASE_DIR / 'outputs/generator.pt')))


@lru_cache(maxsize=1)
def load_generator():
    path = model_path()
    if not path.is_file():
        raise HTTPException(503, 'Trained model missing. Run the training notebook first.')
    model = Generator()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval()


@app.get('/health')
def health():
    return {'status': 'ok', 'model_file_present': model_path().is_file(),
            'diffusion_ready': advanced_path('diffusion').is_file(),
            'ebm_ready': advanced_path('ebm').is_file()}


@app.get('/generate', responses={200: {'content': {'image/png': {}}}},
         response_class=Response)
def generate(num_samples: int = Query(16, ge=1, le=100),
             seed: int | None = Query(None, ge=0, le=2147483647)):
    images = generate_samples(load_generator(), num_samples=num_samples,
                              seed=seed, show=False)
    buffer = io.BytesIO()
    save_image(images, buffer, format='PNG', nrow=min(10, num_samples))
    return Response(content=buffer.getvalue(), media_type='image/png')


def advanced_path(name):
    return BASE_DIR / 'outputs' / f'{name}.pt'


@lru_cache(maxsize=2)
def load_advanced(name):
    path = advanced_path(name)
    if not path.is_file():
        raise HTTPException(503, f'Trained {name} weights are missing. Run train_advanced.py.')
    model = get_model(name)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval().requires_grad_(False)


def sampling_rng(seed):
    rng = torch.Generator()
    if seed is None:
        rng.seed()
    else:
        rng.manual_seed(seed)
    return rng


def render_samples(images):
    buffer = io.BytesIO()
    save_image((images + 1) / 2, buffer, format='PNG', nrow=min(4, len(images)))
    return Response(buffer.getvalue(), media_type='image/png')


@app.get('/generate/diffusion', responses={200: {'content': {'image/png': {}}}},
         response_class=Response)
def generate_diffusion(num_samples: int = Query(4, ge=1, le=16),
                       steps: int = Query(50, ge=2, le=100),
                       seed: int | None = Query(None, ge=0, le=2147483647)):
    return render_samples(sample_diffusion(
        load_advanced('diffusion'), num_samples, steps, sampling_rng(seed)
    ))


@app.get('/generate/ebm', responses={200: {'content': {'image/png': {}}}},
         response_class=Response)
def generate_ebm(num_samples: int = Query(4, ge=1, le=16),
                 steps: int = Query(100, ge=1, le=200),
                 seed: int | None = Query(None, ge=0, le=2147483647)):
    model = load_advanced('ebm')
    rng = sampling_rng(seed)
    initial = torch.rand((num_samples, *IMAGE_SHAPE), generator=rng) * 2 - 1
    return render_samples(langevin_sample(model, initial, steps, rng))
