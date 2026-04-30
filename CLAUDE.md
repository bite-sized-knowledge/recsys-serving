# recsys-serving — Claude 작업 메모

검색 + 추천 서빙 (FastAPI). 모든 ML 컴포넌트 **CPU only** (Mac mini 가정).

## 아키텍처

- **Hybrid 검색**: BM25 (MySQL FULLTEXT ngram parser) + Dense (Qdrant + Qwen3-Embedding-0.6B ONNX) + RRF (k=60)
- **Optional cross-encoder rerank**: BAAI/bge-reranker-base (default OFF, opt-in via mode=`hybrid_rerank`)
- **Stateless cursor**: zlib + base64로 snapshot/offset/query_hash 인코딩. `query_id`도 cursor 안에 echo (`i` 필드).
- **Query Understanding**: 순수 regex sanitize (`app/services/query_understanding.py:analyze`). LLM 제거됨. API contract는 `POST /search/understand` 그대로 유지.
- **추천 서빙 (Phase 1+2)**: 카테고리 단위 Beta-Bernoulli Thompson Sampling + 글로벌 풀 quota fill + Phase 2 user_profile within-category rerank. 상태 (recommendation_global, member_category_bandit, Qdrant user_profile) 는 `recommender` 배치가 채움. 응답 contract `{"articles":[...]}` 유지.

## 추천 흐름 (`/feeds` GET)

1. `bandit.load_or_init` — 풀 카테고리 set 에 대해 (member_id, category_id) state 보장. 없으면 `member_interest` 기반 prior `Beta(4,1)` (선택), `Beta(1,2)` (미선택) 채워 INSERT.
2. `bandit.sample_thetas` — Beta(α,β) sample.
3. `bandit.allocate_quota` — softmax(theta) × 10, top-K=6 카테고리 quota 분배.
4. `recommendation_global` 에서 카테고리별 후보 풀 `quota * 4` 가져옴 (`rank_global ASC`).
5. **Phase 2**: `user_vector_lookup.get_user_vector(member_id)` 있으면 within-category cosine rerank (가중 0.7 글로벌 + 0.3 유사도). 없으면 글로벌 score 그대로.
6. dedup, quota 부족시 글로벌 풀 backfill, in-response shuffle.
7. `impression_logger.log_impressions` — `recommendation_impression` 동기 INSERT (10 row, latency 무시 가능).

## Feedback (`/feeds/feedback` POST)

- bite-api 가 `user_events` insert 후 fire-and-forget 으로 호출 (실패 silent).
- `bandit.apply_reward`: incremental UPDATE α/β. 이벤트 가중치는 `services/bandit.py` 상수.
- Phase 2 positive event (`article_in/like/archive/share`) 는 `user_vector_lookup.push_event_to_profile` 로 user_profile EMA push (decay=0.9, 첫 클릭이면 article_vec 그대로 init).
- 배치 reconcile 이 매일 ground-truth 로 정정 — 실시간 race 무시.

## ⚠️ 추천 함정

- **`recommendation_global` 비어있으면** 빈 응답. recommender 배치가 한 번도 안 돌면 이 상태.
- **응답 contract `{"articles":[...]}`** 변경 금지. bite-api 호출자에 영향. Phase 2 rerank 도입해도 contract 그대로.
- **Qdrant `user_profile` collection** 은 recommender 배치가 만든다 (없으면 Phase 2 코드는 graceful skip 하고 글로벌 score fallback). 차원 1024.
- **bandit incremental ↔ batch ground-truth race**: 둘 다 같은 row UPSERT. 실시간 update 가 잠깐 sweep 될 수 있지만 다음 reconcile (매일) 이 ground-truth 로 덮어씀 — 의도된 동작.

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
