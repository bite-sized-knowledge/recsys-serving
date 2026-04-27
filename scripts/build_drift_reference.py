"""Drift 참조 vector 사전 계산.

색인 측(harvest_post)이 SentenceTransformer로 인코딩한 결과와 동일한 출력을
SAMPLE_QUERIES에 대해 미리 계산하여 .npz로 저장한다. drift.measure()는
런타임에 sentence-transformers 의존 없이 이 파일과 ONNX 출력을 비교만 하면 됨.

production image에 sentence-transformers + torch (CUDA) 변종이 들어가는 것을
회피하기 위한 설계. 모델 변경 시 이 스크립트를 재실행하여 참조를 갱신.

산출물: app/services/drift_reference.npz
  - keys: 각 query → fp32 normalized vector (color_size = config.EMBEDDING_DIM)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import config
from app.services.drift import SAMPLE_QUERIES


def main() -> int:
    from sentence_transformers import SentenceTransformer

    print(f"Loading reference encoder: {config.DRIFT_REFERENCE_MODEL_ID}")
    st = SentenceTransformer(
        config.DRIFT_REFERENCE_MODEL_ID, device="cpu", trust_remote_code=True
    )

    print(f"Encoding {len(SAMPLE_QUERIES)} sample queries...")
    vectors = {}
    for q in SAMPLE_QUERIES:
        raw = np.asarray(st.encode(q, normalize_embeddings=False), dtype=np.float32)
        if raw.shape[0] > config.EMBEDDING_DIM:
            raw = raw[: config.EMBEDDING_DIM]
        norm = float(np.linalg.norm(raw))
        vectors[q] = raw / norm if norm > 0 else raw

    out_path = ROOT / "app" / "services" / "drift_reference.npz"
    np.savez(out_path, **{f"q_{i}": v for i, v in enumerate(vectors.values())})
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"Shape per vector: ({config.EMBEDDING_DIM},)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
