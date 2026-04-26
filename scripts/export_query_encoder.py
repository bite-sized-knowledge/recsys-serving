"""Qwen3-Embedding-0.6B을 ONNX(fp32)로 변환.

산출물:
  models/qwen3-embed-onnx/
    model.onnx
    tokenizer.json (+ tokenizer 관련 파일)

색인 측(harvest_post)이 SentenceTransformer + last token pooling + L2 normalize를
사용하므로, 동일 입력에 대한 cosine similarity가 ≥ 0.99인지 검증한다.

양자화(int8)는 1차에서 제외한다. dynamic int8 quantization은 Qwen3 RoPE attention과
호환되지 않아 cos parity가 ~0.86까지 떨어졌음. 후속 PR에서 calibrated/QAT 시도 예정.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForFeatureExtraction
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "qwen3-embed-onnx"
DIM = 1024
PARITY_THRESHOLD = 0.99
SAMPLES = [
    "LLM 추천 시스템",
    "kubernetes operator pattern",
    "벡터 검색 데이터베이스",
    "검색 시스템 개편",
]


def _last_token_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    seq_lens = attention_mask.sum(axis=1) - 1
    batch_idx = np.arange(last_hidden.shape[0])
    return last_hidden[batch_idx, seq_lens]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _onnx_encode(
    text: str,
    session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
) -> np.ndarray:
    enc = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    input_names = {i.name for i in session.get_inputs()}
    feed = {k: v for k, v in enc.items() if k in input_names}
    if "position_ids" in input_names and "position_ids" not in feed:
        seq_len = enc["input_ids"].shape[1]
        feed["position_ids"] = np.arange(seq_len, dtype=np.int64)[np.newaxis, :].repeat(
            enc["input_ids"].shape[0], axis=0
        )
    last_hidden = session.run(None, feed)[0]
    pooled = _last_token_pool(last_hidden, enc["attention_mask"])[0]
    if pooled.shape[0] > DIM:
        pooled = pooled[:DIM]
    return _l2_normalize(pooled.astype(np.float32))


def _st_encode(text: str, model: SentenceTransformer) -> np.ndarray:
    vec = np.asarray(model.encode(text, normalize_embeddings=False), dtype=np.float32)
    if vec.shape[0] > DIM:
        vec = vec[:DIM]
    return _l2_normalize(vec)


def export_and_verify(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.onnx"

    if model_path.exists() and (output_dir / "tokenizer.json").exists():
        print(f"[1/2] Skip export: {model_path} already exists.")
    else:
        print(f"[1/2] Export: {MODEL_ID} → {output_dir}", flush=True)
        model = ORTModelForFeatureExtraction.from_pretrained(
            MODEL_ID,
            export=True,
            trust_remote_code=True,
        )
        model.save_pretrained(output_dir)
        AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True).save_pretrained(output_dir)

    print("[2/2] Parity check (SentenceTransformer vs ONNX fp32)", flush=True)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), trust_remote_code=True)
    st_model = SentenceTransformer(MODEL_ID, device="cpu", trust_remote_code=True)

    failures = []
    for text in SAMPLES:
        st_vec = _st_encode(text, st_model)
        onnx_vec = _onnx_encode(text, session, tokenizer)
        cos = float(np.dot(st_vec, onnx_vec))
        flag = "OK" if cos >= PARITY_THRESHOLD else "FAIL"
        print(f"  [{flag}] cos={cos:.4f} | {text[:40]}")
        if cos < PARITY_THRESHOLD:
            failures.append((text, cos))

    if failures:
        print(f"\nParity check failed for {len(failures)} sample(s):")
        for text, cos in failures:
            print(f"  cos={cos:.4f} threshold={PARITY_THRESHOLD} | {text}")
        return 1

    print(f"\nDone. ONNX 모델이 SentenceTransformer 결과와 cos ≥ {PARITY_THRESHOLD}로 일치.")
    print(f"Path: {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Qwen3-Embedding-0.6B → ONNX fp32")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    return export_and_verify(args.output)


if __name__ == "__main__":
    sys.exit(main())
