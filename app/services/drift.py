"""Encoder parity drift monitoring (CPU, no sentence-transformers dependency).

색인 측(harvest_post의 SentenceTransformer)과 쿼리 측(recsys ONNX)이 라이브러리
업그레이드/모델 교체로 silent drift하면 cosine similarity가 1에서 멀어진다.

production image에 sentence-transformers + torch가 들어가는 부담을 회피하기 위해
참조 vector를 사전 계산해 `drift_reference.npz`로 동봉한다 (≈127KB). drift.measure는
런타임에 그 참조 vector와 ONNX 출력만 비교한다. 참조 갱신은 `scripts/build_drift_reference.py`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import config

log = logging.getLogger(__name__)

SAMPLE_QUERIES: tuple[str, ...] = (
    "LLM 추천 시스템",
    "kubernetes operator pattern",
    "벡터 검색 데이터베이스",
    "검색 시스템 개편",
    "rate limiter 구현",
    "GPT 함수 호출",
    "테스트 자동화",
    "서비스 메시 istio",
    "데이터 파이프라인 airflow",
    "graphql 페이지네이션",
    "react server components",
    "FastAPI 배포",
    "PostgreSQL 인덱스 튜닝",
    "Redis 캐시 전략",
    "kafka consumer group",
    "마이크로서비스 분리",
    "CI/CD 파이프라인",
    "오픈소스 기여 방법",
    "TypeScript 제네릭",
    "node.js 메모리 누수 디버깅",
    "딥러닝 모델 양자화",
    "JWT 토큰 보안",
    "SwiftUI 애니메이션",
    "rust async runtime",
    "go context propagation",
    "spring boot 설정",
    "WebRTC peer connection",
    "OAuth2 PKCE flow",
    "도커 컴포즈 네트워크",
    "터미널 단축키 모음",
)

REFERENCE_PATH = Path(__file__).with_name("drift_reference.npz")


@dataclass
class DriftReport:
    measured_at: float
    samples: int
    avg_cos: float
    min_cos: float
    violations: int
    threshold: float
    available: bool
    error: Optional[str] = None


_lock = threading.Lock()
_latest: Optional[DriftReport] = None
_reference: Optional[list[np.ndarray]] = None


def latest_report() -> Optional[dict]:
    with _lock:
        return asdict(_latest) if _latest is not None else None


def _set_latest(report: DriftReport) -> None:
    global _latest
    with _lock:
        _latest = report


def _load_reference() -> Optional[list[np.ndarray]]:
    global _reference
    if _reference is not None:
        return _reference
    if not REFERENCE_PATH.exists():
        return None
    data = np.load(REFERENCE_PATH)
    _reference = [data[f"q_{i}"].astype(np.float32) for i in range(len(SAMPLE_QUERIES))]
    return _reference


def measure() -> DriftReport:
    """참조 vector(.npz)와 현재 ONNX 인코더 출력의 cosine 비교.

    참조 파일이 없거나 ONNX 인코더 로드 실패면 available=False.
    """
    reference = _load_reference()
    if reference is None:
        msg = (
            f"reference vector file not found at {REFERENCE_PATH}. "
            "Run scripts/build_drift_reference.py to regenerate."
        )
        log.warning("Drift measurement unavailable: %s", msg)
        report = DriftReport(
            measured_at=time.time(),
            samples=0,
            avg_cos=0.0,
            min_cos=0.0,
            violations=0,
            threshold=config.DRIFT_COS_THRESHOLD,
            available=False,
            error=msg,
        )
        _set_latest(report)
        return report

    try:
        from app.services.embedder import QueryEncoder

        encoder = QueryEncoder.instance()
        cosines: list[float] = []
        for query, ref_vec in zip(SAMPLE_QUERIES, reference):
            onnx_vec = encoder.encode(query)
            if ref_vec.shape != onnx_vec.shape:
                log.warning(
                    "Shape mismatch for %r: reference=%s onnx=%s — rebuild reference?",
                    query, ref_vec.shape, onnx_vec.shape,
                )
                continue
            cosines.append(float(np.dot(ref_vec, onnx_vec)))

        if not cosines:
            raise RuntimeError("no cosines computed (all shape-mismatch)")

        threshold = config.DRIFT_COS_THRESHOLD
        violations = sum(1 for c in cosines if c < threshold)
        report = DriftReport(
            measured_at=time.time(),
            samples=len(cosines),
            avg_cos=float(np.mean(cosines)),
            min_cos=float(np.min(cosines)),
            violations=violations,
            threshold=threshold,
            available=True,
        )
        if violations > 0:
            log.warning(
                "drift parity: avg_cos=%.4f min_cos=%.4f violations=%d/%d threshold=%.2f",
                report.avg_cos, report.min_cos, violations, report.samples, threshold,
            )
        else:
            log.info(
                "drift parity: avg_cos=%.4f min_cos=%.4f violations=0/%d",
                report.avg_cos, report.min_cos, report.samples,
            )
        _set_latest(report)
        return report
    except Exception as exc:
        log.warning("Drift measurement failed: %s", exc)
        report = DriftReport(
            measured_at=time.time(),
            samples=0,
            avg_cos=0.0,
            min_cos=0.0,
            violations=0,
            threshold=config.DRIFT_COS_THRESHOLD,
            available=False,
            error=str(exc),
        )
        _set_latest(report)
        return report
