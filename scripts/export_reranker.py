"""BAAI/bge-reranker-base 가져와 ONNX 모델로 저장 (선택적 int8).

bge-reranker는 HuggingFace Hub에 사전 변환된 ONNX 가 있어 그대로 다운로드한다
(Optimum이 직접 변환하면 sigmoid 전후 차이로 ranking이 미세하게 깨짐).

cross-encoder는 (query, passage) pair를 입력 받아 relevance score(raw logit)을
산출한다. Higher score = more relevant. hybrid_search top-N 정밀 재정렬 용도.

산출물:
  models/bge-reranker-base-onnx/
    model.onnx (fp32, ~400MB)
    model_int8.onnx (--quantize 시, ~110MB)
    tokenizer 관련 파일

CrossEncoder(sigmoid 적용)와 ONNX(raw logit)는 score scale은 다르지만 ranking은 동일.
검증은 argsort 일치 여부로 한다.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForSequenceClassification
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

MODEL_ID = "BAAI/bge-reranker-base"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "bge-reranker-base-onnx"
SAMPLES = [
    ("LLM 추천 시스템", "OpenAI GPT를 활용한 영화 추천 엔진을 만들었습니다."),
    ("LLM 추천 시스템", "리액트로 만든 todo 앱입니다."),
    ("kubernetes operator pattern", "Operator SDK로 CRD를 처리하는 패턴을 정리했다."),
    ("kubernetes operator pattern", "오늘 점심 메뉴는 김치찌개"),
    ("벡터 검색 데이터베이스", "Qdrant로 1억 개 임베딩을 색인하고 latency를 측정했다."),
    ("벡터 검색 데이터베이스", "리액트 hooks 사용법"),
]


def _quantize(fp32_path: Path, int8_path: Path) -> None:
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    ext_data = fp32_path.with_name(fp32_path.name + "_data")
    if ext_data.exists():
        shutil.copy2(ext_data, int8_path.with_name(int8_path.name + "_data"))


def _onnx_score_pairs(
    pairs: list[tuple[str, str]],
    session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
) -> np.ndarray:
    enc = tokenizer(
        [q for q, _ in pairs],
        [p for _, p in pairs],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    input_names = {i.name for i in session.get_inputs()}
    feed = {k: v for k, v in enc.items() if k in input_names}
    logits = session.run(None, feed)[0]
    return logits.squeeze(-1).astype(np.float32)


def export_and_verify(output_dir: Path, quantize: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "model.onnx"
    int8_path = output_dir / "model_int8.onnx"

    if fp32_path.exists() and (output_dir / "tokenizer.json").exists():
        print(f"[1/3] Skip download: {fp32_path}")
    else:
        print(f"[1/3] Fetch pre-converted ONNX: {MODEL_ID}/onnx → {output_dir}", flush=True)
        model = ORTModelForSequenceClassification.from_pretrained(
            MODEL_ID, subfolder="onnx", file_name="model.onnx", export=False
        )
        model.save_pretrained(output_dir)
        AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(output_dir)

    if quantize:
        if int8_path.exists():
            print(f"[2/3] Skip quantize: {int8_path}")
        else:
            print(f"[2/3] Quantize int8 (per-channel) → {int8_path}", flush=True)
            _quantize(fp32_path, int8_path)
        target = int8_path
        flavor = "int8"
    else:
        print("[2/3] Skip quantize (use --quantize to enable).")
        target = fp32_path
        flavor = "fp32"

    print(f"[3/3] Parity check (CrossEncoder vs ONNX {flavor})", flush=True)
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    ce = CrossEncoder(MODEL_ID, max_length=512, device="cpu")

    ce_scores = ce.predict(SAMPLES, convert_to_numpy=True)
    onnx_scores = _onnx_score_pairs(SAMPLES, session, tokenizer)

    ce_rank = np.argsort(-ce_scores)
    onnx_rank = np.argsort(-onnx_scores)
    rank_match = bool(np.array_equal(ce_rank, onnx_rank))

    print(f"  CrossEncoder ranks: {ce_rank.tolist()}")
    print(f"  ONNX ranks        : {onnx_rank.tolist()}")
    print(f"  ranking 일치       : {rank_match}")

    pairs_50 = SAMPLES * 9
    start = time.perf_counter()
    _onnx_score_pairs(pairs_50, session, tokenizer)
    ms = (time.perf_counter() - start) * 1000
    print(f"  ONNX rerank ({len(pairs_50)} pairs): {ms:.0f}ms")

    if not rank_match:
        print("\nFAIL: ranking이 일치하지 않음 (양자화 손실 가능).")
        return 1
    print(f"\nOK: ONNX {flavor} reranker가 CrossEncoder ranking과 일치.")
    print(f"Path: {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch BAAI/bge-reranker-base ONNX")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="weight-only int8 양자화 (BERT 아키텍처는 보통 ranking 보존됨)",
    )
    args = parser.parse_args()
    return export_and_verify(args.output, args.quantize)


if __name__ == "__main__":
    sys.exit(main())
