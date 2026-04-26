# Bite-Knowledge Recommendation Serving Repo

![image](https://github.com/user-attachments/assets/3d5e0b00-de94-4538-845a-f6504606ce8b)

FastAPI 기반의 추천/검색 서빙 서비스.

## 의존성 관리

`uv`를 사용한다 (pip / conda 사용 금지).

```bash
brew install uv          # 최초 1회
uv sync                  # 런타임 deps 설치
uv sync --all-groups     # dev deps 포함 (ONNX export 등)
```

## 로컬 실행

```bash
uv run python server.py
```

기본 포트는 `8000`. Cloudflare tunnel 접근을 위해 `0.0.0.0` 바인딩 필수.

## 검색 모듈 사전 준비

쿼리 임베딩에 사용할 ONNX 모델은 host에서 1회 export 필요.

```bash
uv run python scripts/export_query_encoder.py
```

산출물은 `models/qwen3-embed-onnx-int8/`에 저장되며 `.gitignore`에 포함되어 있다. Docker 빌드 시에는 host에 export된 결과가 image에 그대로 포함된다.

## Docker

```bash
docker build -t recsys-serving .
docker run -p 8001:8001 recsys-serving
```
