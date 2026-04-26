"""Qwen3-Embedding-0.6B을 ONNX(fp32 또는 weight-only int8)로 변환.

산출물:
  models/qwen3-embed-onnx/
    model.onnx
    tokenizer.json (+ tokenizer 관련 파일)

색인 측(harvest_post)이 SentenceTransformer + last token pooling + L2 normalize를
사용하므로, 동일 입력에 대한 cosine similarity가 ≥ 0.99인지 검증한다.

`--quantize` 플래그를 주면 MatMul 노드만 weight-only int8로 양자화한다.
일반 dynamic int8(LayerNorm·RoPE 포함)은 Qwen3 RoPE attention과 호환되지 않아
parity가 ~0.86까지 떨어지므로, op_types_to_quantize=["MatMul"]로 제한하여
attention 내부는 fp32를 유지한다.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
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


def _quantize_weight_only(fp32_path: Path, int8_path: Path) -> None:
    """MatMul 노드만 weight-only int8 양자화.

    ONNX Runtime의 dynamic quantization은 op_types_to_quantize로 양자화 대상을
    제한할 수 있다. Qwen3 RoPE attention(LayerNorm/Sin/Cos/Add 등)을 양자화하면
    parity가 깨지므로 MatMul에만 적용 → 모델 크기 ~25% 감소, attention 손실 0.
    """
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        reduce_range=False,
        extra_options={"WeightSymmetric": True},
    )


def export_and_verify(output_dir: Path, quantize: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "model.onnx"
    int8_path = output_dir / "model_int8.onnx"

    if fp32_path.exists() and (output_dir / "tokenizer.json").exists():
        print(f"[1/3] Skip export: {fp32_path} already exists.")
    else:
        print(f"[1/3] Export fp32: {MODEL_ID} → {output_dir}", flush=True)
        model = ORTModelForFeatureExtraction.from_pretrained(
            MODEL_ID,
            export=True,
            trust_remote_code=True,
        )
        model.save_pretrained(output_dir)
        AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True).save_pretrained(output_dir)

    if quantize:
        if int8_path.exists():
            print(f"[2/3] Skip int8 quantize: {int8_path} already exists.")
        else:
            print(f"[2/3] Quantize MatMul → int8: {int8_path}", flush=True)
            _quantize_weight_only(fp32_path, int8_path)
            # external data 파일이 있으면 함께 복사 (large model 양자화 시)
            ext_data = fp32_path.with_name(fp32_path.name + "_data")
            if ext_data.exists():
                shutil.copy2(ext_data, int8_path.with_name(int8_path.name + "_data"))
        target_path = int8_path
        flavor = "int8"
    else:
        print("[2/3] Skip quantize (use --quantize to enable).")
        target_path = fp32_path
        flavor = "fp32"

    print(f"[3/3] Parity check (SentenceTransformer vs ONNX {flavor})", flush=True)
    session = ort.InferenceSession(str(target_path), providers=["CPUExecutionProvider"])
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
    parser = argparse.ArgumentParser(description="Export Qwen3-Embedding-0.6B → ONNX")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="MatMul 노드만 weight-only int8 양자화 (parity ≥ 0.99 검증)",
    )
    args = parser.parse_args()
    return export_and_verify(args.output, args.quantize)


if __name__ == "__main__":
    sys.exit(main())
