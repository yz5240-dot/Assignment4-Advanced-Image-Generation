# Assignment 4 · Advanced Image Generation

Yulin Zhang

This project extends the Assignment 3 FastAPI and helper library. It provides the original MNIST GAN route and two CIFAR-10 generation routes.

| Endpoint | Algorithm | Query parameters |
|---|---|---|
| `GET /generate/diffusion` | U-Net noise prediction and DDIM sampling | `num_samples` (1–16), `steps` (2–100), optional `seed` |
| `GET /generate/ebm` | ConvNet energy function and Langevin sampling | `num_samples` (1–16), `steps` (1–200), optional `seed` |
| `GET /generate` | Original MNIST GAN | `num_samples` (1–100), optional `seed` |
| `GET /health` | Check for the three weight files | None |

The diffusion model learns to predict Gaussian noise on CIFAR-10 images at random timesteps. Its U-Net has two downsampling stages, skip connections, and sinusoidal timestep conditioning. The EBM gives one scalar energy per image. Training lowers the energy of real images relative to negative samples and regularizes both energies. Langevin sampling differentiates energy with respect to the image and updates its pixels. The optimizer differentiates the contrastive training loss with respect to the model parameters.

## Included training run

Both checkpoints were trained on the full 50,000-image CIFAR-10 training split for five epochs on CPU. The loss histories record the per-epoch training objectives. The sample grids show that both samplers generate varied images. Many samples still look noisy or blurry after five epochs, and individual CIFAR-10 objects are difficult to identify.

The recorded run used batch size 96, random seed 42, and eight Langevin steps per training batch for the EBM. The data came from the [CIFAR-10 Parquet mirror](https://huggingface.co/datasets/uoft-cs/cifar10). With PyArrow installed and that file downloaded as `train.parquet`, the training commands were:

```bash
python train_advanced.py diffusion --parquet train.parquet --epochs 5 --batch-size 96 --cpu-threads 5
python train_advanced.py ebm --parquet train.parquet --epochs 5 --batch-size 96 --langevin-steps 8 --cpu-threads 4
```

## Run with Docker Desktop

From the extracted project folder, with Docker Desktop running:

```powershell
docker compose up --build
```

Open [Swagger UI](http://localhost:8000/docs) or visit the two URLs below. Both endpoints return PNG image grids.

- [Diffusion, four images](http://localhost:8000/generate/diffusion?num_samples=4&steps=50&seed=42)
- [EBM, four images](http://localhost:8000/generate/ebm?num_samples=4&steps=100&seed=42)

From another terminal, with the Python requirements installed, run `python verify_api.py --url http://localhost:8000`. The script checks health, valid PNG responses, reproducibility with a fixed seed, and invalid input handling. To test FastAPI in process, omit `--url`.

## Train with Colab

Open `Assignment4_Training.ipynb` in Colab, select a T4 GPU, and upload this ZIP when the notebook prompts. The notebook trains both models and exports a refreshed project ZIP. Alternatively, from a Python environment with the requirements installed:

```bash
python train_advanced.py diffusion --epochs 20
python train_advanced.py ebm --epochs 20
```

Both commands use the full 50,000-image CIFAR-10 training split by default. Training saves inference weights, loss histories, and sample grids to `outputs/`. Add `--resume` in the same environment to resume from its training checkpoint, using `--epochs` as the new total. Restart the API after replacing weights.

## Theory

`Assignment4_Theory_Yulin_Zhang.pdf` and `Theory_Answers.md` contain the nine answers with derivations and runnable PyTorch examples.

## GitHub submission

The project was prepared as a separate Assignment 4 project based on Assignment 3. If using the GitHub website, create the Assignment 4 repository and upload the extracted files and directories. Upload the weight files in `outputs/` with the source files so the Docker build can load all three models. Keep `Dockerfile`, `compose.yaml`, and `api.py` at the root of the repository. Use the committed repository URL in your submission.
