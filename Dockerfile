FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

# llama-cpp-python(QU LLM)은 C++ wheel을 직접 빌드하므로 build toolchain 필요
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ cmake make \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
