"""Cross-encoder reranker for hybrid search top-N refinement.

BAAI/bge-reranker-base를 ONNX(int8 weight-only)로 CPU 추론. (query, passage) pair에
점수를 매겨 top-N을 정밀 재정렬한다. 일반 dense embedding은 query/passage를 독립적으로
encoding하지만, cross-encoder는 두 입력의 attention을 직접 계산하므로 ranking 품질이
훨씬 높다.

Latency 추정:
  - 30 pair 배치(batch=16, fp32 fallback): ~150ms (M-series CPU)
  - int8 quantize + per-channel: ~80ms
이는 hybrid_search 100ms와 합쳐도 p95 < 300ms 목표 안에 들어간다.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from app.core.config import config

log = logging.getLogger(__name__)


class Reranker:
    _instance: "Reranker | None" = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> "Reranker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def __init__(self) -> None:
        model_dir = Path(config.RERANKER_MODEL_DIR)
        candidate = model_dir / config.RERANKER_MODEL_FILENAME
        # int8 파일이 없으면 fp32(model.onnx)로 fallback
        if not candidate.exists():
            fallback = model_dir / "model.onnx"
            if fallback.exists():
                log.warning(
                    "Reranker int8 model not found at %s, falling back to fp32 %s",
                    candidate, fallback,
                )
                candidate = fallback
            else:
                raise FileNotFoundError(
                    f"Reranker ONNX model not found at {model_dir}. "
                    "Run `uv run python scripts/export_reranker.py [--quantize]`."
                )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0

        self._session = ort.InferenceSession(
            str(candidate),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._max_len = config.RERANKER_MAX_LEN
        self._batch_size = max(config.RERANKER_BATCH_SIZE, 1)

    def warmup(self) -> None:
        self.score([("warmup", "warmup passage")])

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        scores: list[float] = []
        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i : i + self._batch_size]
            scores.extend(self._score_batch(batch))
        return scores

    def _score_batch(self, batch: Sequence[Tuple[str, str]]) -> List[float]:
        enc = self._tokenizer(
            [q for q, _ in batch],
            [p for _, p in batch],
            padding=True,
            truncation=True,
            max_length=self._max_len,
            return_tensors="np",
        )
        feed = {k: v for k, v in enc.items() if k in self._input_names}
        logits = self._session.run(None, feed)[0]
        return logits.squeeze(-1).astype(np.float32).tolist()


def rerank(
    query: str,
    candidates: List[Tuple[str, str]],
    top_k: int | None = None,
) -> List[str]:
    """candidates: [(article_id, passage_text)] → reranked article_id list desc by score."""
    if not candidates:
        return []
    reranker = Reranker.instance()
    pairs = [(query, passage) for _, passage in candidates]
    scores = reranker.score(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    ids = [c[0] for c, _ in ranked]
    return ids[:top_k] if top_k else ids
