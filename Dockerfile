FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2

COPY requirements.txt .
RUN python -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY helper_lib/ ./helper_lib/
COPY api.py .
COPY outputs/generator.pt outputs/diffusion.pt outputs/ebm.pt ./outputs/
RUN python -c "from api import load_advanced, load_generator; load_generator(); load_advanced('diffusion'); load_advanced('ebm')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import json,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)); assert r['diffusion_ready'] and r['ebm_ready']"
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
