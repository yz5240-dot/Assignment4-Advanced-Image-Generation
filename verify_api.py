import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image


def verify(client):
    health = client.get('/health')
    health.raise_for_status()
    assert all(health.json()[key] for key in (
        'model_file_present', 'diffusion_ready', 'ebm_ready',
    )), health.json()
    for route, steps in (('/generate', None), ('/generate/diffusion', 50), ('/generate/ebm', 100)):
        params = {'num_samples': 4, 'seed': 42}
        if steps is not None:
            params['steps'] = steps
        response = client.get(route, params=params, timeout=120)
        response.raise_for_status()
        assert response.headers['content-type'] == 'image/png'
        with Image.open(BytesIO(response.content)) as image:
            image.verify()
        repeat = client.get(route, params=params, timeout=120)
        assert response.content == repeat.content, route
        stem = route.rsplit('/', 1)[-1]
        Path('outputs').mkdir(exist_ok=True)
        Path(f'outputs/api_{stem}.png').write_bytes(response.content)
        print(f'{route}: HTTP {response.status_code}, valid reproducible PNG')
    assert client.get('/generate/diffusion', params={'steps': 1}).status_code == 422
    assert client.get('/generate/ebm', params={'num_samples': 0}).status_code == 422


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help='Use a running server. Omit to test FastAPI in process.')
    args = parser.parse_args()
    if args.url:
        from httpx import Client

        with Client(base_url=args.url, trust_env=False) as client:
            verify(client)
    else:
        from fastapi.testclient import TestClient

        from api import app

        with TestClient(app) as client:
            verify(client)
