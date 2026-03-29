FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-dotenv \
    pymysql \
    sqlalchemy \
    numpy \
    pandas \
    pydantic \
    pydantic-settings \
    boto3 \
    cryptography \
    langchain \
    langchain-text-splitters \
    qdrant-client

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
