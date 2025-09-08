FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY pyproject.toml .
RUN uv sync

RUN apt update && apt install -y --no-install-recommends \
	ffmpeg tesseract-ocr \
	&& rm -rf /var/lib/apt/lists/*

RUN uv pip install fastapi uvicorn python-multipart

COPY . .
