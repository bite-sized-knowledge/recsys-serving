# recsys-serving — Claude 작업 메모

검색 + 추천 서빙 (FastAPI). 모든 ML 컴포넌트 **CPU only** (Mac mini 가정).

## 아키텍처

- **Hybrid 검색**: BM25 (MySQL FULLTEXT ngram parser) + Dense (Qdrant + Qwen3-Embedding-0.6B ONNX) + RRF (k=60)
- **Optional cross-encoder rerank**: BAAI/bge-reranker-base (default OFF, opt-in via mode=`hybrid_rerank`)
- **Stateless cursor**: zlib + base64로 snapshot/offset/query_hash 인코딩. `query_id`도 cursor 안에 echo (`i` 필드).
- **Query Understanding**: 순수 regex sanitize (`app/services/query_understanding.py:analyze`). LLM 제거됨. API contract는 `POST /search/understand` 그대로 유지.

## ⚠️ 실제로 겪은 함정 (반복 금지)

### ONNX & 양자화
- **int8 dynamic quantization은 Qwen3 RoPE attention을 깨뜨림** (cos 0.86~0.93). fp32 유지 필수. `--quantize` 플래그는 opt-in으로만 두고 default 금지.
- **bge-reranker int8**도 top-1은 맞지만 2~4위가 흔들림. fp32 default.
- bge-reranker 로드 시 pre-converted ONNX는 `subfolder="onnx", file_name="model.onnx"` 명시 필수.
- ONNX export 시 `position_ids`를 입력에 명시적으로 추가해야 함. 생략하면 export는 통과해도 inference 결과가 틀어짐.

### 의존성 호환
- **transformers 5.x ↔ optimum 1.17 깨짐** (`is_tf_available` import error). 선택지: ① `transformers >=4.49,<5.0`, ② `optimum 2.x + optimum-onnx` 분리 패키지.
- **sentence-transformers를 production deps에 두면 PyTorch CUDA variant가 끌려와 디스크가 가득 참** ("No space left"). dev group 유지. drift monitoring은 pre-computed `.npz` 참조 vector로 회피.
- llama-cpp-python은 빌드에 gcc/g++/cmake 필요. Query Understanding LLM 제거 시 Dockerfile의 toolchain layer도 같이 제거.
- fastembed BM25는 한국어 미지원 → sparse vector 채널은 도입 보류.

### 검색 모드 dispatch (가장 흔한 실수)
- `mode in (SearchMode.HYBRID, ...)` 분기는 **BM25 채널 + Dense 채널 양쪽 모두에 추가**해야 함. 한쪽만 추가하면 빈 결과만 반환됨 (HYBRID_RERANK 추가 시 한 번 당함).

### Reranker latency
- 긴 passage 그대로 넣으면 2.5s+. SQL `SUBSTRING`으로 passage 자르고 `max_len=128`, `asyncio.wait_for(timeout=1.0~1.5s)` + 실패 시 RRF 원래 순서로 silent fallback.

## query_norm_hash (cross-service 동기화)

- 알고리즘: `lower + strip + sha1[:12]` (hex 12자).
- `recsys-serving/app/api/search/service.py:_query_hash` ↔ `bite-api/internal/event/service.go:normalizeAndHashQuery`. **둘 중 한쪽만 바꾸면 분석 join이 silent하게 깨짐.**

## Drift monitoring

- 색인 측(harvest_post SentenceTransformer) ↔ 쿼리 측(recsys ONNX) 인코더 silent drift 감지가 목적.
- production은 sentence-transformers 안 쓰고 `app/services/drift_reference.npz` (≈127KB) 동봉. 참조 갱신은 `scripts/build_drift_reference.py` (dev group).
- lifespan asyncio task로 24h 주기. 별도 cron 컨테이너 X.
- `GET /admin/diagnostics/drift`에서 마지막 측정 결과 조회.

## 사전 준비

- ONNX export는 host에서 1회: `uv run python scripts/export_query_encoder.py`. 산출물 `models/`는 .gitignore — Docker build 시 host의 export 결과가 image에 포함됨.
- 의존성: `uv sync` (런타임), `uv sync --all-groups` (export/drift reference 등 dev).

## 검증 명령

- 평가: `uv run python scripts/eval_search.py --samples 30 --k 20` (Recall@K, latency p50/p95).
- drift 수동 1회: `curl -H "X-API-Key:..." http://localhost:8001/admin/diagnostics/drift`.
