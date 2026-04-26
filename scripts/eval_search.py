"""검색 품질 평가 — fulltext vs hybrid 비교.

본 스크립트는 검색용 golden dataset(query → expected article_id)이 별도로 갖춰지기
전까지 사용하는 자동 평가다. DB에서 article을 랜덤 샘플링하여 title을 query로
넣고, 각 모드(fulltext/dense/hybrid)에서 자기 자신이 top-K에 포함되는지를 측정한다.

지표:
  Recall@K    : top-K에 정답이 있는 비율
  MRR         : 정답의 reciprocal rank 평균
  latency p50 : 각 mode당 응답 시간 중앙값
  latency p95 : 95-percentile

후속 PR에서 manual golden dataset(`data/search_golden.jsonl`, query×expected_ids)이
준비되면 본 스크립트는 두 모드를 모두 지원하도록 확장한다.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from sqlalchemy import select

from app.db.dependencies import get_db
from app.models.article import Article
from app.services.hybrid_search import SearchMode, hybrid_search
from app.services.qdrant_client import SearchFilters

DEFAULT_SAMPLES = 50
DEFAULT_K = 20


@dataclass
class ModeResult:
    mode: str
    recall_at_k: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((p / 100) * (len(s) - 1)))
    return s[k]


async def _evaluate_mode(
    db,
    samples: list[tuple[str, str]],
    mode: SearchMode,
    k: int,
) -> ModeResult:
    hits = 0
    rr_sum = 0.0
    latencies: List[float] = []

    for article_id, title in samples:
        start = time.perf_counter()
        try:
            results = await hybrid_search(
                db=db,
                query=title,
                filters=SearchFilters(),
                max_pool=max(k, 50),
                mode=mode,
            )
        except Exception as exc:
            print(f"  ! {mode.value} 실패 ({article_id}): {exc}")
            continue
        latencies.append((time.perf_counter() - start) * 1000)
        results_top_k = results[:k]
        if article_id in results_top_k:
            hits += 1
            rank = results_top_k.index(article_id) + 1
            rr_sum += 1.0 / rank

    n = max(len(samples), 1)
    return ModeResult(
        mode=mode.value,
        recall_at_k=hits / n,
        mrr=rr_sum / n,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
    )


async def main_async(samples_n: int, k: int) -> int:
    db_gen = get_db()
    db = next(db_gen)
    try:
        rows = db.execute(
            select(Article.article_id, Article.title)
            .where(Article.title.isnot(None))
            .order_by(Article.article_id)
            .limit(samples_n)
        ).all()
        samples = [(str(r[0]), str(r[1])) for r in rows if r[0] and r[1]]
        if not samples:
            print("샘플 article이 없습니다.")
            return 1
        print(f"샘플 {len(samples)}개로 평가 시작 (top-K={k})\n")

        modes = [
            SearchMode.FULLTEXT,
            SearchMode.DENSE,
            SearchMode.HYBRID,
            SearchMode.HYBRID_RERANK,
        ]
        results: dict[str, ModeResult] = {}
        for mode in modes:
            print(f"== {mode.value} ==")
            results[mode.value] = await _evaluate_mode(db, samples, mode, k)
            r = results[mode.value]
            print(f"  Recall@{k}: {r.recall_at_k:.3f}")
            print(f"  MRR      : {r.mrr:.3f}")
            print(f"  p50      : {r.latency_p50_ms:.1f} ms")
            print(f"  p95      : {r.latency_p95_ms:.1f} ms\n")

        print("== 비교표 ==")
        print(f"{'mode':<10} {'Recall@K':>10} {'MRR':>8} {'p50ms':>8} {'p95ms':>8}")
        for mode_name, r in results.items():
            print(
                f"{mode_name:<10} {r.recall_at_k:>10.3f} {r.mrr:>8.3f} "
                f"{r.latency_p50_ms:>8.1f} {r.latency_p95_ms:>8.1f}"
            )
        return 0
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid search 품질 평가 (self-retrieval)")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()
    return asyncio.run(main_async(args.samples, args.k))


if __name__ == "__main__":
    raise SystemExit(main())
