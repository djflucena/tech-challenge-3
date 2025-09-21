FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=9080
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Cache de deps primeiro (mais rápido em rebuild)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Agora copia o resto (respeita .dockerignore)
COPY . /app

EXPOSE 9080
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-9080} main:server"]
