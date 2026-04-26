"""Query encoder for hybrid search.

색인 측(harvest_post `src/embedder.py`)이 SentenceTransformer로 Qwen3-Embedding-0.6B를
last token pooling + L2 normalize 방식으로 인코딩한다. 본 모듈은 동일한 결과를
ONNX Runtime(int8 quantized)로 ~5x 빠르게 산출한다.

`scripts/export_query_encoder.py`로 모델을 사전 export 해야 한다.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from app.core.config import config

log = logging.getLogger(__name__)


def _last_token_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    seq_lens = attention_mask.sum(axis=1) - 1
    batch_idx = np.arange(last_hidden.shape[0])
    return last_hidden[batch_idx, seq_lens]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class QueryEncoder:
    """Singleton-style ONNX Runtime encoder for search queries."""

    _instance: "QueryEncoder | None" = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> "QueryEncoder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        model_dir = Path(config.EMBEDDING_MODEL_DIR)
        model_path = model_dir / config.EMBEDDING_MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Run `uv run python scripts/export_query_encoder.py`."
            )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # let ORT pick

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self._max_len = config.EMBEDDING_MAX_LEN
        self._dim = config.EMBEDDING_DIM

    def warmup(self) -> None:
        self.encode("warmup")

    def encode(self, text: str) -> np.ndarray:
        enc = self._tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self._max_len,
        )
        feed = {k: v for k, v in enc.items() if k in self._input_names}
        if "position_ids" in self._input_names and "position_ids" not in feed:
            seq_len = enc["input_ids"].shape[1]
            feed["position_ids"] = np.arange(seq_len, dtype=np.int64)[np.newaxis, :].repeat(
                enc["input_ids"].shape[0], axis=0
            )
        last_hidden = self._session.run(None, feed)[0]
        pooled = _last_token_pool(last_hidden, enc["attention_mask"])[0]
        if pooled.shape[0] > self._dim:
            pooled = pooled[: self._dim]
        return _l2_normalize(pooled.astype(np.float32))
